"""
Learning curve: is the Elo regressor data-limited or signal-limited?

Subsamples increasing numbers of players, retrains with train.py's 5-fold
GroupKFold, and plots R^2 / MAE against player count; a curve still rising at the
largest sample means data-limited. On a mixed-pool dataset it also overlays a
blitz-only curve at equal player counts. No engine needed.

Usage:
    python -m src.ml.learning_curve data/dataset_blitz_1600p.parquet --outdir results
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
from sklearn.model_selection import GroupKFold, cross_validate
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES
from src.ml.train import PARAMS  # identical model config to the deployed trainer
from src.ml import plotstyle
plotstyle.apply()


def curve(df, levels, n_repeats, base_seed=0):
    """R^2/MAE at each player-count level, averaged over random player subsamples."""
    players = df["player"].unique()
    rows = []
    for L in levels:
        L = min(L, len(players))
        reps = 1 if L >= len(players) else n_repeats
        r2s, maes, nrows = [], [], []
        for r in range(reps):
            rng = np.random.default_rng(base_seed + r)
            sel = rng.choice(players, L, replace=False)
            sub = df[df["player"].isin(sel)]
            res = cross_validate(
                XGBRegressor(**PARAMS, n_jobs=1),
                sub[FEATURE_NAMES], sub["target_elo"],
                cv=GroupKFold(5), groups=sub["player"],
                scoring=["r2", "neg_mean_absolute_error"], n_jobs=-1,
            )
            r2s.append(float(res["test_r2"].mean()))
            maes.append(float(-res["test_neg_mean_absolute_error"].mean()))
            nrows.append(len(sub))
        rows.append({"n_players": L, "n_rows": int(np.mean(nrows)),
                     "r2": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
                     "mae": float(np.mean(maes)), "mae_std": float(np.std(maes))})
        print(f"  {L:>5} players | rows~{int(np.mean(nrows)):>5} | "
              f"R2={np.mean(r2s):.3f}±{np.std(r2s):.3f} | MAE={np.mean(maes):.0f}", flush=True)
    return rows


def slope_verdict(rows):
    """Is the tail of the curve still rising?"""
    if len(rows) < 2:
        return "n/a"
    d_r2 = rows[-1]["r2"] - rows[-2]["r2"]
    d_players = rows[-1]["n_players"] - rows[-2]["n_players"]
    per1k = d_r2 / max(d_players, 1) * 1000
    if per1k > 0.02:
        return f"STILL RISING (+{per1k:.3f} R2 per +1000 players) -> more players will help"
    if per1k < -0.005:
        return f"NOISY/declining ({per1k:+.3f} R2 per +1000) -> not data-limited"
    return f"FLATTENED ({per1k:+.3f} R2 per +1000 players) -> signal-limited, not data-limited"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    n_all = df["player"].nunique()
    is_multi_pool = "pool" in df.columns and df["pool"].nunique() > 1
    print(f"Loaded {len(df):,} rows | {n_all} players\n")

    mixed_levels = sorted({100, 250, 500, 800, 1100, 1400, n_all})
    mixed_levels = [L for L in mixed_levels if L <= n_all]
    print("== Mixed-pool learning curve ==" if is_multi_pool else "== Learning curve ==")
    mixed = curve(df, mixed_levels, args.repeats)

    result = {"mixed": mixed, "mixed_verdict": slope_verdict(mixed)}

    blitz = None
    if is_multi_pool:
        # On a single-pool dataset blitz_df would equal df, making this a
        # self-vs-self duplicate of the curve above.
        blitz_df = df[df["pool"] == "blitz"]
        if blitz_df["player"].nunique() >= 50:
            nb = blitz_df["player"].nunique()
            blitz_levels = sorted({50, 100, 200, 300, nb})
            blitz_levels = [L for L in blitz_levels if L <= nb]
            print(f"\n== Blitz-only learning curve ({nb} players) ==")
            blitz = curve(blitz_df, blitz_levels, args.repeats)
            result["blitz"] = blitz
            result["blitz_verdict"] = slope_verdict(blitz)
    else:
        print("\n(Dataset is already single-pool -- skipping the blitz-only overlay, "
              "which would just duplicate the curve above.)")

    with open(os.path.join(args.outdir, "learning_curve.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def plot_on(ax, key, ylabel):
        mx = [r["n_players"] for r in mixed]
        ax.errorbar(mx, [r[key] for r in mixed], yerr=[r[key + "_std"] for r in mixed],
                    marker="o", capsize=3, label=("mixed pools" if is_multi_pool else "all players"))
        if blitz:
            bx = [r["n_players"] for r in blitz]
            ax.errorbar(bx, [r[key] for r in blitz], yerr=[r[key + "_std"] for r in blitz],
                        marker="s", capsize=3, label="blitz only")
        ax.set_xlabel("number of players")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plot_on(ax1, "r2", "CV R²")
    ax1.set_title("R² vs players (player-grouped 5-fold)")
    plot_on(ax2, "mae", "CV MAE (Elo)")
    ax2.set_title("MAE vs players")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_learning_curve.png"))
    plt.close(fig)

    print("\n=== Verdict ===")
    print(("mixed pools:" if is_multi_pool else "overall    :"), result["mixed_verdict"])
    if blitz:
        print("blitz only :", result["blitz_verdict"])
        # fair comparison at equal player count (~400)
        m_at = min(mixed, key=lambda r: abs(r["n_players"] - blitz[-1]["n_players"]))
        print(f"\nAt ~{blitz[-1]['n_players']} players: "
              f"blitz R²={blitz[-1]['r2']:.3f} vs mixed R²={m_at['r2']:.3f} "
              f"(@{m_at['n_players']}) -> "
              f"{'single-pool helps' if blitz[-1]['r2'] > m_at['r2'] else 'no single-pool gain'}")
    print(f"\nSaved -> {args.outdir}/fig_learning_curve.png + learning_curve.json")


if __name__ == "__main__":
    main()
