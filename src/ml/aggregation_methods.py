"""
Which feature-aggregation strategy predicts Elo best from a player's N games?

Each player's games are collapsed into ONE vector, then evaluated with
player-level 5-fold CV. Strategies: mean (the deployed default), median,
trimmed20, and mean+std (which doubles the vector to 58 dims).

Usage:
    python -m src.ml.aggregation_methods data/dataset_blitz_1600p.parquet --outdir results
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES
from src.ml.train import PARAMS


def aggregate(g: pd.DataFrame, method: str) -> dict:
    X = g[FEATURE_NAMES].to_numpy(dtype=float)
    if method == "mean":
        v = np.nanmean(X, axis=0)
        return dict(zip(FEATURE_NAMES, v))
    if method == "median":
        v = np.nanmedian(X, axis=0)
        return dict(zip(FEATURE_NAMES, v))
    if method == "trimmed20":
        v = [trim_mean(col[~np.isnan(col)], 0.2) if np.any(~np.isnan(col)) else np.nan
             for col in X.T]
        return dict(zip(FEATURE_NAMES, v))
    if method == "mean+std":
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        d = {f + "_mean": m for f, m in zip(FEATURE_NAMES, mean)}
        d.update({f + "_std": s for f, s in zip(FEATURE_NAMES, std)})
        return d
    raise ValueError(method)


def evaluate_method(df, method, min_games, seed=42):
    rows, ys = [], []
    for _player, g in df.groupby("player"):
        if len(g) < min_games:
            continue
        rows.append(aggregate(g, method))
        ys.append(float(g["target_elo"].median()))
    X = pd.DataFrame(rows)
    y = np.array(ys)
    res = cross_validate(
        XGBRegressor(**PARAMS, n_jobs=1), X, y,
        cv=KFold(5, shuffle=True, random_state=seed),
        scoring=["r2", "neg_mean_absolute_error"], n_jobs=-1,
    )
    return {"n_players": len(y), "n_features": X.shape[1],
            "r2": float(res["test_r2"].mean()), "r2_std": float(res["test_r2"].std()),
            "mae": float(-res["test_neg_mean_absolute_error"].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--min-games", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    df = df[df["target_elo"].between(400, 3000)]
    print(f"Loaded {len(df):,} rows | {df['player'].nunique()} players "
          f"(aggregating players with >= {args.min_games} games)\n")

    methods = ["mean", "median", "trimmed20", "mean+std"]
    results = {}
    print(f"{'method':12}{'dims':>6}{'players':>9}{'R2':>9}{'MAE':>8}")
    for m in methods:
        r = evaluate_method(df, m, args.min_games)
        results[m] = r
        print(f"{m:12}{r['n_features']:>6}{r['n_players']:>9}"
              f"{r['r2']:>9.3f}{r['mae']:>8.0f}")

    best = max(results, key=lambda m: results[m]["r2"])
    print(f"\nBest by R²: {best}  (R²={results[best]['r2']:.3f}, MAE={results[best]['mae']:.0f})")
    with open(os.path.join(args.outdir, "aggregation_methods.json"), "w") as f:
        json.dump({"best": best, "results": results, "min_games": args.min_games}, f, indent=2)
    print(f"Saved -> {args.outdir}/aggregation_methods.json")


if __name__ == "__main__":
    main()
