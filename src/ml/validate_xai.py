"""
Validation of the SHAP-based weakness detector, resolving the weakness the way
app.py does: average the N feature vectors, then explain the aggregated one.

Experiment A -- test-retest reliability: split each player's games in two and
    measure agreement (Cohen's kappa, cluster bootstrap over players).
Experiment B -- agreement with a behavioural ground truth built from the player's
    features over all their games (worst area by population z-score).

Usage:
    python -m src.ml.validate_xai data/dataset_blitz_1600p.parquet --outdir results
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, DISPLAY_NAMES, WEAKNESS_CANDIDATES, FEATURE_ORIENTATION
from src.ml import plotstyle
plotstyle.apply()

RNG = 42


def fit_model_and_explainer(X, y):
    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=RNG)
    model.fit(Xi, y)
    import shap
    return model, shap.TreeExplainer(model), Xi


def per_game_weakness(explainer, Xi):
    """For each game, the candidate feature whose SHAP value most lowers Elo."""
    sv = explainer.shap_values(Xi)
    cand_idx = [FEATURE_NAMES.index(c) for c in WEAKNESS_CANDIDATES]
    sub = sv[:, cand_idx]
    picks = np.argmin(sub, axis=1)  # most negative = strongest downward pull
    return np.array([WEAKNESS_CANDIDATES[i] for i in picks])


def modal(series):
    m = series.mode()
    return m.iloc[0] if len(m) else None


def aggregate_weakness(explainer, Xi, groups, k=1):
    """The k weakest candidates of each group of games, in app.py's order: average
    the feature vectors first, then explain the single aggregated vector."""
    stacked = pd.DataFrame([Xi.loc[rows].mean().values for rows in groups],
                           columns=Xi.columns)
    cand_idx = [FEATURE_NAMES.index(c) for c in WEAKNESS_CANDIDATES]
    ranked = explainer.shap_values(stacked)[:, cand_idx].argsort(axis=1)[:, :k]
    names = np.array(WEAKNESS_CANDIDATES)[ranked]
    return names[:, 0] if k == 1 else names


def _pair_scores(first, second):
    share = pd.Series(first).value_counts(normalize=True)
    return {"split_half_agreement": float(np.mean(np.asarray(first) == np.asarray(second))),
            "chance_agreement": float((share ** 2).sum()),
            "cohen_kappa": float(cohen_kappa_score(first, second, labels=WEAKNESS_CANDIDATES)),
            "n_categories": int(share.size)}


def experiment_a(df, Xi, explainer, min_games=4, seeds=range(10), boot=1000):
    splits = []
    for player, g in df.groupby("player"):
        if len(g) < min_games:
            continue
        idx = g.index.to_numpy()
        for seed in seeds:
            shuffled = np.random.RandomState(seed).permutation(idx)
            half = len(shuffled) // 2
            splits.append((player, shuffled[:half], shuffled[half:]))

    aggregated = aggregate_weakness(
        explainer, Xi, [h for _, a, b in splits for h in (a, b)])
    per_game = df["shap_weak"].to_numpy()
    rec = pd.DataFrame({
        "player": [s[0] for s in splits],
        "aggregate_a": aggregated[0::2], "aggregate_b": aggregated[1::2],
        "modal_a": [modal(pd.Series(per_game[s[1]])) for s in splits],
        "modal_b": [modal(pd.Series(per_game[s[2]])) for s in splits]})

    out = _pair_scores(rec["aggregate_a"], rec["aggregate_b"])
    out["n_players"] = int(rec["player"].nunique())
    out["n_splits"] = int(len(rec))
    out["modal_comparison"] = _pair_scores(rec["modal_a"], rec["modal_b"])

    # cluster bootstrap over players -- one player's splits are not independent
    groups = {p: d for p, d in rec.groupby("player")}
    players = np.array(list(groups))
    rng = np.random.RandomState(RNG)
    kap = lambda d, v: cohen_kappa_score(d[f"{v}_a"], d[f"{v}_b"], labels=WEAKNESS_CANDIDATES)
    drawn = []
    for _ in range(boot):
        sample = pd.concat([groups[p] for p in rng.choice(players, len(players), replace=True)])
        drawn.append((kap(sample, "aggregate"), kap(sample, "modal")))
    drawn = np.array(drawn)
    span = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out["kappa_ci95"] = span(drawn[:, 0])
    out["modal_comparison"]["kappa_ci95"] = span(drawn[:, 1])
    delta = drawn[:, 0] - drawn[:, 1]
    out["kappa_gain_over_modal"] = {
        "delta": out["cohen_kappa"] - out["modal_comparison"]["cohen_kappa"],
        "ci95": span(delta), "p_gt_0": float(np.mean(delta > 0))}
    return out


def experiment_b(df, Xi, explainer):
    # Ground truth: per-player z-score x FEATURE_ORIENTATION, so idxmax = weakest.
    pm = df.groupby("player")[WEAKNESS_CANDIDATES].mean()
    z = (pm - pm.mean()) / pm.std(ddof=0)
    orient = pd.Series(FEATURE_ORIENTATION)[WEAKNESS_CANDIDATES]
    gt = z.mul(orient, axis=1).idxmax(axis=1).rename("gt_weak")  # highest = weakest area
    # Prediction: SHAP over each player's games aggregated the way app.py does.
    rows = df.groupby("player").indices
    players = pd.Index(list(rows), name="player")
    ranked = aggregate_weakness(explainer, Xi, [rows[p] for p in players], k=2)
    pred = pd.Series(ranked[:, 0], index=players, name="pred_weak")
    # second guess: is GT among the two weakest candidates of the same vector?
    top2 = pd.Series(list(ranked), index=players)
    merged = pd.concat([gt, pred], axis=1).dropna()
    top1 = float((merged["gt_weak"] == merged["pred_weak"]).mean())
    in_top2 = float(np.mean([gt.loc[p] in top2.loc[p] for p in merged.index]))
    kappa = cohen_kappa_score(merged["gt_weak"], merged["pred_weak"],
                              labels=WEAKNESS_CANDIDATES)
    # baselines
    chance = 1.0 / len(WEAKNESS_CANDIDATES)
    majority = float(gt.value_counts(normalize=True).iloc[0])
    return ({"n_players": int(len(merged)),
             "top1_accuracy": top1, "top2_accuracy": in_top2,
             "cohen_kappa": float(kappa),
             "chance_baseline": chance, "majority_baseline": majority},
            merged, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    X, y = df[FEATURE_NAMES], df["target_elo"]
    print(f"Loaded {len(df):,} rows | {df['player'].nunique()} players\n")

    model, explainer, Xi = fit_model_and_explainer(X, y)
    df["shap_weak"] = per_game_weakness(explainer, Xi)

    print("== Distribution of single-game SHAP-identified weaknesses ==")
    print(df["shap_weak"].value_counts().to_string(), "\n")

    a = experiment_a(df, Xi, explainer)
    print("== Experiment A: test-retest reliability ==")
    for k, v in a.items():
        if isinstance(v, dict):
            continue
        print(f"  {k:24s}: {v:.3f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    lift_a = a["split_half_agreement"] / a["chance_agreement"]
    print(f"  -> agreement is {lift_a:.1f}x chance")
    m, gain = a["modal_comparison"], a["kappa_gain_over_modal"]
    print(f"  -> kappa {a['cohen_kappa']:.3f} "
          f"[{a['kappa_ci95'][0]:.3f}; {a['kappa_ci95'][1]:.3f}] aggregated vs "
          f"{m['cohen_kappa']:.3f} [{m['kappa_ci95'][0]:.3f}; {m['kappa_ci95'][1]:.3f}] modal; "
          f"delta {gain['delta']:+.3f} [{gain['ci95'][0]:.3f}; {gain['ci95'][1]:.3f}], "
          f"P(delta>0) = {gain['p_gt_0']:.3f}\n")

    b, merged, gt = experiment_b(df, Xi, explainer)
    print("== Experiment B: agreement with behavioural ground truth ==")
    for k, v in b.items():
        print(f"  {k:24s}: {v:.3f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    print(f"  -> top-1 is {b['top1_accuracy']/b['chance_baseline']:.1f}x chance\n")

    with open(os.path.join(args.outdir, "xai_validation.json"), "w") as f:
        json.dump({"experiment_a": a, "experiment_b": b}, f, indent=2)
    merged.to_csv(os.path.join(args.outdir, "xai_groundtruth_vs_pred.csv"))

    # figure: GT vs predicted weakness confusion
    labels = WEAKNESS_CANDIDATES
    conf = pd.crosstab(merged["gt_weak"], merged["pred_weak"]).reindex(
        index=labels, columns=labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(conf.values, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([DISPLAY_NAMES.get(l, l) for l in labels], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([DISPLAY_NAMES.get(l, l) for l in labels], fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, conf.values[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted (aggregated SHAP)"); ax.set_ylabel("Ground truth (aggregated)")
    ax.set_title("Weakness detection: ground truth vs aggregated SHAP")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_xai_confusion.png")); plt.close(fig)
    print(f"Saved XAI validation tables + figure to '{args.outdir}/'.")


if __name__ == "__main__":
    main()
