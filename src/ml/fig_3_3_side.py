"""Рисунок 3.3 магістерської з двома панелями поруч (за зразком кафедри: одна картинка, під кожною
панеллю – її позначка, спільний підпис у тексті роботи). Дані ті самі, що для fig_3_3a/fig_3_3b
у figures_ua.py: results/pool_summary.csv і results/aggregation_mixed_by_pool.json.
Розмір – під ширину рядка роботи (≈16 см), шрифт 9 pt у натуральну величину.
Usage: python -m src.ml.fig_3_3_side --out results/fig_ua/fig_3_3_side.png"""
import argparse, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ml.figures_ua import POOL, CommaFormatter, X_N, Y_MAE  # noqa: E402  (імпорт застосовує plotstyle)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/fig_ua/fig_3_3_side.png")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8.5,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5})
    tab = pd.read_csv("results/pool_summary.csv")
    curves = json.load(open("results/aggregation_mixed_by_pool.json"))
    W_IN = 16.0 / 2.54
    fig = plt.figure(figsize=(W_IN, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.55, left=0.075, right=0.99, top=0.97, bottom=0.33)
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    names = [POOL[p] for p in tab["pool"]]
    ax1.bar(names, tab["MAE"], color="#C44E52"); ax1.set_ylabel(Y_MAE)
    ax2.bar(names, tab["R2"], color="#4C72B0"); ax2.set_ylabel("R²")
    ax2.yaxis.set_major_formatter(CommaFormatter())
    for ax in (ax1, ax2):
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for pool, curve in curves.items():
        ax3.plot([c["n_games"] for c in curve], [c["r2_mixed"] for c in curve], "o-", ms=3.5, label=POOL[pool])
    ax3.set_xlabel(X_N.replace("Кількість агрегованих партій гравця", "Кількість партій гравця")); ax3.set_ylabel("R²")
    ax3.set_xticks([1, 2, 3, 5, 8])
    ax3.yaxis.set_major_formatter(CommaFormatter()); ax3.legend(loc="lower right", frameon=True)
    # позначки панелей під ними
    b1, b2, b3 = ax1.get_position(), ax2.get_position(), ax3.get_position()
    fig.text((b1.x0 + b2.x1) / 2, 0.02, "а) окрема партія", ha="center", va="bottom", fontsize=10)
    fig.text((b3.x0 + b3.x1) / 2, 0.02, "б) криві агрегування", ha="center", va="bottom", fontsize=10)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches=None)
    print(a.out, fig.get_size_inches())


if __name__ == "__main__":
    main()
