"""
Elo-prediction accuracy vs. number of games aggregated per player.

For each N, mean-aggregate N random games per player into one vector (the
deployed aggregator) and evaluate with player-level 5-fold CV.

Usage:
    python -m src.ml.aggregation_curve data/dataset_blitz_2k.parquet --outdir results
    python -m src.ml.aggregation_curve --pools data/dataset_bullet_2k.parquet,data/dataset_blitz_2k.parquet,data/dataset_rapid_2k.parquet,data/dataset_classical_2k.parquet --outdir results
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
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES
from src.features.aggregate import aggregate_features
from src.ml.train import PARAMS
from src.ml import plotstyle
plotstyle.apply()


def eval_at_N(df, N, repeats=3, base_seed=0):
    r2s, maes, npl = [], [], []
    for r in range(repeats):
        rng = np.random.default_rng(base_seed + r)
        rows, ys = [], []
        for _player, g in df.groupby("player"):
            if len(g) < N:
                continue
            idx = rng.choice(len(g), N, replace=False)
            sub = g.iloc[idx]
            dicts = sub[FEATURE_NAMES].to_dict("records")
            rows.append(aggregate_features(dicts, "mean"))
            ys.append(float(g["target_elo"].median()))
        X, y = pd.DataFrame(rows), np.array(ys)
        res = cross_validate(XGBRegressor(**PARAMS, n_jobs=1), X, y,
                             cv=KFold(5, shuffle=True, random_state=base_seed + r),
                             scoring=["r2", "neg_mean_absolute_error"], n_jobs=-1)
        r2s.append(res["test_r2"].mean())
        maes.append(-res["test_neg_mean_absolute_error"].mean())
        npl.append(len(y))
    return (float(np.mean(r2s)), float(np.std(r2s)),
            float(np.mean(maes)), int(np.mean(npl)))


def by_pool(paths, outdir, levels_str):
    """
    The same mean-aggregation curve, repeated per pool. Mirrors evaluate.py's
    --pools convention but consolidates output into a single JSON.
    """
    requested = [int(x) for x in levels_str.split(",")]
    curves = {}
    for path in paths:
        df = pd.read_parquet(path)
        df = df[df["target_elo"].between(400, 3000)]
        pool_name = df["pool"].mode().iloc[0] if "pool" in df.columns else os.path.basename(path)
        per_player = df.groupby("player").size()
        levels = [n for n in requested if (per_player >= n).sum() >= 30]
        curve = []
        for N in levels:
            r2, r2sd, mae, npl = eval_at_N(df, N)
            curve.append({"n_games": N, "n_players": npl, "r2": r2, "r2_std": r2sd, "mae": mae})
        curves[pool_name] = curve
        print(f"{pool_name:>10}: N={levels[0]} R2={curve[0]['r2']:.3f} -> "
              f"N={levels[-1]} R2={curve[-1]['r2']:.3f}")

    with open(os.path.join(outdir, "aggregation_curve_by_pool.json"), "w") as f:
        json.dump(curves, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for pool_name, curve in curves.items():
        ax.plot([c["n_games"] for c in curve], [c["r2"] for c in curve], "o-", label=pool_name)
    ax.set_xlabel("games aggregated per player (N)"); ax.set_ylabel("CV R²")
    ax.set_title("Aggregation curve by time-control pool"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_aggregation_by_pool.png"))
    plt.close(fig)
    print(f"Saved -> {outdir}/aggregation_curve_by_pool.json, fig_aggregation_by_pool.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="?", default=None)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--levels", default="1,2,3,5,8,12")
    ap.add_argument("--pools", default=None,
                    help="comma-separated single-pool parquet paths "
                         "(e.g. data/dataset_bullet_2k.parquet,data/dataset_blitz_2k.parquet,...) "
                         "-> per-pool aggregation-curve comparison only, skips the single-dataset run below")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.pools:
        by_pool([p.strip() for p in args.pools.split(",")], args.outdir, args.levels)
        return
    if not args.parquet:
        ap.error("either a parquet path or --pools is required")

    df = pd.read_parquet(args.parquet)
    df = df[df["target_elo"].between(400, 3000)]
    print(f"Loaded {len(df):,} rows | {df['player'].nunique()} players (mean aggregation)\n")

    per_player = df.groupby("player").size()
    requested = [int(x) for x in args.levels.split(",")]
    levels = [n for n in requested if (per_player >= n).sum() >= 30]
    skipped = [n for n in requested if n not in levels]
    if skipped:
        print(f"Skipping N={skipped}: fewer than 30 players have that many games "
              f"(this dataset was collected with --games-per-player={int(per_player.max())}).\n")
    curve = []
    print(f"{'N games':>8}{'players':>9}{'R2':>9}{'MAE':>8}")
    for N in levels:
        r2, r2sd, mae, npl = eval_at_N(df, N)
        curve.append({"n_games": N, "n_players": npl, "r2": r2, "r2_std": r2sd, "mae": mae})
        print(f"{N:>8}{npl:>9}{r2:>9.3f}{mae:>8.0f}")

    with open(os.path.join(args.outdir, "aggregation_curve.json"), "w") as f:
        json.dump(curve, f, indent=2)

    ns = [c["n_games"] for c in curve]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.errorbar(ns, [c["r2"] for c in curve], yerr=[c["r2_std"] for c in curve],
                 marker="o", capsize=3, color="#1f77b4")
    ax1.set_xlabel("games aggregated per player (N)"); ax1.set_ylabel("CV R²")
    ax1.set_title("Accuracy rises with more games/player"); ax1.grid(True, alpha=0.3)
    ax2.plot(ns, [c["mae"] for c in curve], marker="s", color="#d62728")
    ax2.set_xlabel("games aggregated per player (N)"); ax2.set_ylabel("CV MAE (Elo)")
    ax2.set_title("Error falls with more games/player"); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_aggregation_curve.png"))
    plt.close(fig)
    print(f"\nN=1 R²={curve[0]['r2']:.3f} -> N={ns[-1]} R²={curve[-1]['r2']:.3f}. "
          f"Saved -> {args.outdir}/fig_aggregation_curve.png")


if __name__ == "__main__":
    main()
