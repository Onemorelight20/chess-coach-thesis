"""
Test-retest reliability of the weakness mechanism: modal order (per-game argmin,
as in validate_xai.experiment_a) vs aggregate order (app.py's), scored on
identical players, splits and seeds, with shap_norm_compare's standardisation
knob:

    alpha = 0  -> argmin phi_j
    alpha = 1  -> argmin (phi_j - mu_j) / sigma_j

Confidence intervals come from a cluster bootstrap over players.

Usage:
    python -m src.ml.detector_order --pool blitz --seeds 10 --boot 1000
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, WEAKNESS_CANDIDATES
from src.ml.validate_xai import fit_model_and_explainer


def build_splits(frame, seeds, min_games):
    """Half/half splits of every eligible player, one per seed."""
    splits = []
    for player, games in frame.groupby("player"):
        if len(games) < min_games:
            continue
        index = games.index.to_numpy()
        for seed in range(seeds):
            shuffled = np.random.RandomState(seed).permutation(index)
            cut = len(shuffled) // 2
            splits.append((player, shuffled[:cut], shuffled[cut:]))
    return splits


def score(first, second):
    labels = list(range(len(WEAKNESS_CANDIDATES)))
    share = pd.Series(first).value_counts(normalize=True)
    return {
        "agreement": float(np.mean(np.asarray(first) == np.asarray(second))),
        "chance": float((share ** 2).sum()),
        "kappa": float(cohen_kappa_score(first, second, labels=labels)),
        "categories": int(share.size),
        "top_category": WEAKNESS_CANDIDATES[int(share.index[0])],
        "top_share": float(share.iloc[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="blitz")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--min-games", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--alphas", default="0,0.5,1")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    frame = pd.read_parquet(os.path.join("data", f"dataset_{args.pool}_2k.parquet"))
    _, explainer, features = fit_model_and_explainer(frame[FEATURE_NAMES], frame["target_elo"])
    features = features.copy()
    features["player"] = frame["player"].values
    features = features.reset_index(drop=True)
    columns = [FEATURE_NAMES.index(c) for c in WEAKNESS_CANDIDATES]

    # per-game contributions: the modal estimator, and the population moments
    per_game = explainer.shap_values(features[FEATURE_NAMES])[:, columns]
    mu, sigma = per_game.mean(axis=0), per_game.std(axis=0)
    sigma[sigma == 0] = 1.0
    per_game_argmin = per_game.argmin(axis=1)

    splits = build_splits(features, args.seeds, args.min_games)
    print(f"splits: {len(splits)} | players: {len({s[0] for s in splits})}")

    # aggregate estimator: average the vectors first, explain once
    stacked = np.vstack([features.loc[half, FEATURE_NAMES].mean().values
                         for _, first, second in splits for half in (first, second)])
    aggregated = explainer.shap_values(pd.DataFrame(stacked, columns=FEATURE_NAMES))[:, columns]

    modal = lambda rows: Counter(per_game_argmin[rows]).most_common(1)[0][0]
    table = pd.DataFrame({"player": [s[0] for s in splits],
                          "modal_a": [modal(s[1]) for s in splits],
                          "modal_b": [modal(s[2]) for s in splits]})
    for alpha in alphas:
        z = ((aggregated - alpha * mu) / (sigma ** alpha)).argmin(axis=1)
        table[f"aggr{alpha:g}_a"], table[f"aggr{alpha:g}_b"] = z[0::2], z[1::2]

    variants = ["modal"] + [f"aggr{a:g}" for a in alphas]
    results = {v: score(table[f"{v}_a"], table[f"{v}_b"]) for v in variants}

    # cluster bootstrap over players
    groups = {p: d for p, d in table.groupby("player")}
    players = np.array(list(groups))
    rng = np.random.RandomState(0)
    draws = {v: [] for v in variants}
    deltas = []
    for _ in range(args.boot):
        sample = pd.concat([groups[p] for p in rng.choice(players, len(players), replace=True)])
        for v in variants:
            draws[v].append(cohen_kappa_score(sample[f"{v}_a"], sample[f"{v}_b"],
                                              labels=list(range(len(WEAKNESS_CANDIDATES)))))
        deltas.append(draws[variants[1]][-1] - draws["modal"][-1])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for v in variants:
        results[v]["ci95"] = ci(draws[v])

    headline = {
        "baseline": "modal", "candidate": variants[1],
        "delta_kappa": results[variants[1]]["kappa"] - results["modal"]["kappa"],
        "ci95": ci(deltas), "p_delta_gt_0": float(np.mean(np.array(deltas) > 0)),
    }

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "detector_order.json")
    with open(out, "w") as fh:
        json.dump({"pool": args.pool, "seeds": args.seeds, "min_games": args.min_games,
                   "n_splits": len(splits), "n_players": int(len(players)),
                   "bootstrap": args.boot, "results": results, "headline": headline}, fh, indent=2)

    print(f"\n{'variant':<10}{'agree':>8}{'chance':>9}{'kappa':>8}{'ci95':>18}{'cats':>6}")
    for v in variants:
        r = results[v]
        span = "[%.3f; %.3f]" % (r["ci95"][0], r["ci95"][1])
        print(f"{v:<10}{r['agreement']:>8.3f}{r['chance']:>9.3f}{r['kappa']:>8.3f}"
              f"{span:>18}{r['categories']:>6}")
    print(f"\ndelta = {headline['delta_kappa']:+.3f} "
          f"[{headline['ci95'][0]:.3f}; {headline['ci95'][1]:.3f}]  "
          f"P(delta>0) = {headline['p_delta_gt_0']:.3f}  ->  {out}")


if __name__ == "__main__":
    main()
