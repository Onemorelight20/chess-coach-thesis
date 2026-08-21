"""
Static architecture diagram: five pipeline stages left-to-right, with a dashed
arrow showing that new games re-enter the same pipeline. The system persists no
per-player profile between runs, so that arrow is a usage cycle, not a profile
update. Reads no data; it lives here so the figure has a versioned source.

Usage:
    python -m src.ml.plot_architecture --outdir results
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STAGES = [
    ("1. Рівень даних", ["Стратиф. збір, 400–2800", "200 Elo/бін, 4 пули",
                         "Збір із контрольними точками"], "#DCE6F1"),
    ("2. Рівень аналізу", ["Stockfish, MultiPV=5", "0,05 с/хід (робочий режим)",
                           "оцінка ±1000 cp"], "#E2EFDA"),
    ("3. Вилучення ознак", ["29 ознак (21+3+5)", "фази за матеріалом",
                            "win-prob, engine-agreement"], "#FCE4D6"),
    ("4. Модель + XAI", ["XGBoost (окремо на пул)", "SHAP → внески 29 ознак",
                         "слабкість = argmin φⱼ"], "#E4DFEC"),
    ("5. Адаптація", ["тема ← слабкість", "пазл Lichess за рейтингом",
                      "рекалібровка (Chess.com)"], "#FCE4EC"),
]

CAPTION = ("цикл використання: нові партії гравця проходять той самий "
          "конвеєр заново — окремого збереженого профілю система не веде")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    n = len(STAGES)
    box_w, box_h, gap = 6.4, 3.4, 0.7
    total_w = n * box_w + (n - 1) * gap
    fig, ax = plt.subplots(figsize=(18, 6.6))

    centers = []
    for i, (title, bullets, color) in enumerate(STAGES):
        x0 = i * (box_w + gap)
        centers.append(x0 + box_w / 2)
        box = FancyBboxPatch((x0, 0), box_w, box_h,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             linewidth=1.4, edgecolor="black", facecolor=color)
        ax.add_patch(box)
        ax.text(x0 + box_w / 2, box_h - 0.55, title, ha="center", va="center",
                fontsize=13, fontweight="bold")
        for j, b in enumerate(bullets):
            ax.text(x0 + 0.35, box_h - 1.35 - j * 0.68, f"• {b}",
                    ha="left", va="center", fontsize=10.5)
        if i > 0:
            xprev = (i - 1) * (box_w + gap) + box_w
            ax.plot([xprev, x0], [box_h / 2, box_h / 2], color="black", lw=1.2)

    # Feedback arrow, stage 5 -> stage 1. Manual quadratic Bezier for the dip
    # depth; a separate FancyArrowPatch draws just the arrowhead.
    x_start, x_end = centers[-1], centers[0]
    y0, y_dip = -0.5, -3.2
    t = np.linspace(0, 1, 200)
    ctrl_x, ctrl_y = (x_start + x_end) / 2, y_dip
    bx = (1 - t) ** 2 * x_start + 2 * (1 - t) * t * ctrl_x + t ** 2 * x_end
    by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * ctrl_y + t ** 2 * y0

    ax.plot([x_start, x_start], [0, y0], "--", color="#C44E52", lw=1.6)
    ax.plot(bx[:-4], by[:-4], "--", color="#C44E52", lw=1.6)
    ax.add_patch(FancyArrowPatch((bx[-4], by[-4]), (x_end, 0),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color="#C44E52", lw=1.6, shrinkA=0, shrinkB=0))

    ax.text(total_w / 2, y_dip - 0.85, CAPTION, ha="center", va="center",
            fontsize=11.5, style="italic", color="#C44E52")

    ax.set_xlim(-0.3, total_w + 0.3)
    ax.set_ylim(y_dip - 1.7, box_h + 0.3)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_architecture.png"), dpi=200)
    plt.close(fig)
    print(f"Saved -> {args.outdir}/fig_architecture.png")


if __name__ == "__main__":
    main()
