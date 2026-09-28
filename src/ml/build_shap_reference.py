"""
Reference SHAP statistics per rating band, so the live app can rank a single player.

Why this exists. `predict_and_explain` used to pick the weakness as the raw
`argmin` of the SHAP values. Three things made that answer wrong as *advice*:

  1. Features are not on a common scale. `opening_acpl` carries the largest
     xgb_gain (0.298), so its contributions are systematically the largest in
     magnitude -- argmin picked the loudest feature, not the worst one.
  2. SHAP is measured against the average player of the whole sample. A 900-rated
     player is below average on everything, so the most negative contribution goes
     to whichever feature has the steepest population gradient -- the same one for
     nearly everybody at that level. Hence `avg_think_time` for 27% of players.
  3. For some features the model links a WORSE value to a HIGHER contribution
     (collinearity inside the move-quality block: at a fixed overall ACPL a high
     endgame ACPL means the mistakes were not made in the opening). The minimum
     then lands on a player who is unusually good at that feature.

(1) and (2) are fixed by standardising each feature's contribution against players
of the SAME predicted-rating band; (3) by keeping only candidates whose sign passes
`candidate_signs.sign_test` on this pool's training players. On blitz (2160 players,
validate_xai protocol) agreement with the deviation-based reference over the same
candidate set rises 0.052 -> 0.508 (trivial baseline 0.170) and, measured against
the fixed 17-class reference, 0.052 -> 0.38.

A single player cannot supply those statistics, so they are computed once over the
training population and stored next to the model, together with the candidate list.

Usage:
    python -m src.ml.build_shap_reference data/dataset_blitz_2k.parquet --pool blitz
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from src.features.extractor import THEMED_CANDIDATES, TRAINABLE_CANDIDATES, FEATURE_ORIENTATION

REF_PATH_TMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models", "shap_reference_{pool}.pkl")


def build(parquet, pool, n_bands=8, sign_filter=True, all_themed=False, out_path=None):
    from src.ml.model import ELOModel      # локально: model.py імпортує цей модуль
    from src.ml.candidate_signs import sign_test
    df = pd.read_parquet(parquet)
    df = df[df["target_elo"].between(400, 3000)]
    model = ELOModel(pool=pool)
    feats = model.feature_names

    # one aggregated vector per player, exactly as app.py builds it
    agg = df.groupby("player")[feats].mean()
    pred = pd.Series(model.model.predict(agg), index=agg.index)
    sv = model.explainer.shap_values(agg)

    # Condition 2 of the candidate pool: the sign test, evaluated on THIS pool's
    # model. Falls back to the documented blitz list only when asked to.
    if all_themed:                       # diagnostic reference: all 17 themed features, no sign test
        candidates, signs = list(THEMED_CANDIDATES), {}
    elif sign_filter:
        candidates, signs = sign_test(agg, sv, feats, THEMED_CANDIDATES, FEATURE_ORIENTATION)
    else:
        candidates, signs = list(TRAINABLE_CANDIDATES), {}
    excluded = [c for c in THEMED_CANDIDATES if c not in candidates]
    shap = pd.DataFrame(sv, index=agg.index, columns=feats)[candidates]

    band = pd.qcut(pred, n_bands, labels=False, duplicates="drop")
    edges = [float(x) for x in pd.qcut(pred, n_bands, retbins=True,
                                       duplicates="drop")[1]]
    mean = shap.groupby(band).mean()
    std = shap.groupby(band).std(ddof=0).replace(0, np.nan)
    # Knots for continuous interpolation. Hard binning made the rule jump: two
    # halves of one player's games land in different bands 57% of the time (a
    # 4-game prediction is noisy), so they were normalised against different
    # statistics and the answer flipped. Interpolating between band centres makes
    # a small change in the predicted rating produce a small change in the score.
    centres = pred.groupby(band).mean().reindex(mean.index)

    ref = {"pool": pool, "candidates": list(candidates), "sign_excluded": excluded,
           "sign_test": signs,
           "band_edges": edges, "band_centres": [float(x) for x in centres],
           "mean": mean, "std": std,
           "n_players": int(len(agg)), "n_bands": int(mean.shape[0])}
    path = out_path or REF_PATH_TMPL.format(pool=pool)
    joblib.dump(ref, path)
    print(f"{pool}: {len(agg)} players, {mean.shape[0]} bands "
          f"[{edges[0]:.0f}…{edges[-1]:.0f}], {len(candidates)} candidates "
          f"(sign test excluded: {', '.join(excluded) or 'none'}) -> {path}")
    return ref


def load(pool):
    """Reference for `pool`, or None if it has not been built yet."""
    path = REF_PATH_TMPL.format(pool=pool)
    return joblib.load(path) if os.path.exists(path) else None


def band_of(ref, predicted_elo):
    """Band index for a rating, clamped to the range the reference covers."""
    edges = ref["band_edges"]
    idx = int(np.searchsorted(edges, predicted_elo, side="right") - 1)
    return max(0, min(idx, ref["n_bands"] - 1))


def stats_at(ref, predicted_elo):
    """Mean and std per candidate at `predicted_elo`, linearly interpolated
    between band centres (constant beyond the outermost knots)."""
    x = ref.get("band_centres")
    if not x:                                     # довідник старого формату
        b = band_of(ref, predicted_elo)
        return ref["mean"].iloc[b], ref["std"].iloc[b]
    # Only candidates the stored frames actually carry: a reference built before
    # a feature rename still lists the old name in `candidates`, and indexing
    # `mean` with it would raise instead of letting the caller fall back.
    cands = [c for c in ref["candidates"] if c in ref["mean"].columns]
    mu = pd.Series({c: float(np.interp(predicted_elo, x, ref["mean"][c].values))
                    for c in cands})
    sd = pd.Series({c: float(np.interp(predicted_elo, x, ref["std"][c].values))
                    for c in cands})
    return mu, sd


def _degenerate(sd):
    """Standard deviations that cannot standardise anything: zero variance
    (`build` stores it as NaN) or a statistic that interpolation left missing."""
    sd = np.asarray(sd, dtype=float)
    return ~np.isfinite(sd) | (sd == 0)


def standardised_scores(shap_dict, mu, sd, candidates):
    """
    (phi - mu) / sd per candidate -- the single definition of the
    band-standardised rule, shared by model.py and the validation harnesses so
    that they measure the same mechanism.

    Candidates whose band statistics are degenerate are DROPPED rather than
    scored: a contribution that cannot be standardised must not be namable.
    Keeping it would put a NaN into `sorted(..., key=score.get)`, where every
    comparison is False and the winner is decided by input order, silently.
    """
    scored = {}
    for c in candidates:
        if c not in shap_dict or c not in mu.index or _degenerate(sd[c]):
            continue
        scored[c] = (float(shap_dict[c]) - float(mu[c])) / float(sd[c])
    return scored


def standardise_matrix(values, mu, sd):
    """
    Array form of `standardised_scores` for the validation harnesses: a row of
    contributions per player against mu/sd broadcast to the same shape.

    Degenerate entries become NaN, which numpy's argsort places last -- the
    array equivalent of dropping the candidate from the pool.
    """
    values = np.asarray(values, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = (values - mu) / sd
    return np.where(_degenerate(sd), np.nan, scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--pool", required=True, choices=["bullet", "blitz", "rapid", "classical"])
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--no-sign-filter", action="store_true",
                    help="skip the sign test and store the documented blitz list (11)")
    ap.add_argument("--all-themed", action="store_true",
                    help="diagnostic: keep all 17 themed candidates (no sign test); use with --out so the "
                         "production reference is not overwritten (synthetic_validation --reference)")
    ap.add_argument("--out", default=None, help="write the reference here instead of models/shap_reference_<pool>.pkl")
    args = ap.parse_args()
    build(args.parquet, args.pool, args.bands, sign_filter=not args.no_sign_filter,
          all_themed=args.all_themed, out_path=args.out)


if __name__ == "__main__":
    main()
