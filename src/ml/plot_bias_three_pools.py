"""
Рисунок 3.12: зміщення прогнозу за рейтингом, Lichess проти Chess.com, три пули.

Точки беруться з `results/<pool>/external_validation_bias.csv`, а лінії й
розриви між ними рахує та сама модель `bias ~ platform * rating`, що й
`external_validation.rating_adjusted_gap`.

Usage:
    python -m src.ml.plot_bias_three_pools --outdir results
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ml import plotstyle

POOLS = [("bullet", "Bullet"), ("blitz", "Blitz"), ("rapid", "Rapid")]
LEVELS = (800, 2200)


def fit(df):
    """bias ~ platform * rating; повертає дві лінії та розрив на рівні rating."""
    y = df["bias"].to_numpy(float)
    cc = (df["platform"] != "lichess").to_numpy(float)
    elo = df["actual_elo"].to_numpy(float)
    X = np.column_stack([np.ones(len(y)), cc, elo, cc * elo])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    line = lambda c, e: beta[0] + beta[1] * c + beta[2] * e + beta[3] * c * e
    return line, lambda e: beta[1] + beta[3] * e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    plotstyle.apply()

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, (pool, title) in zip(axes, POOLS):
        path = os.path.join(args.outdir, pool, "external_validation_bias.csv")
        df = pd.read_csv(path)
        line, gap = fit(df)

        for plat, colour, label in (("lichess", plotstyle.PRIMARY, "Lichess, control group"),
                                    ("chess.com", plotstyle.SECONDARY, "Chess.com")):
            sub = df[df["platform"] == plat]
            ax.scatter(sub["actual_elo"], sub["bias"], s=14, alpha=0.45,
                       color=colour, edgecolors="none", label=label)

        xs = np.linspace(df["actual_elo"].min(), df["actual_elo"].max(), 200)
        ax.plot(xs, line(0.0, xs), color=plotstyle.PRIMARY, lw=1.8)
        ax.plot(xs, line(1.0, xs), color=plotstyle.SECONDARY, lw=1.8)

        for e in LEVELS:
            lo, hi = line(0.0, e), line(1.0, e)
            ax.annotate("", xy=(e, hi), xytext=(e, lo),
                        arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4",
                                        color="black", lw=1.2))
            ax.text(e + 60, (lo + hi) / 2, f"{gap(e):+.0f}",
                    va="center", ha="left", fontsize=9, fontweight="bold")

        ax.axhline(0, color="grey", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Actual Elo")
    axes[0].set_ylabel("Prediction bias (predicted − actual)")

    h, l = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color="grey", lw=1.8))
    l.append("fitted lines and gap between them")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.09))
    out = os.path.join(args.outdir, "fig_bias_three_pools.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"->  {out}")


if __name__ == "__main__":
    main()
