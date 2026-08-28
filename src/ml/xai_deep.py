"""
Deepened XAI study: per-rating-band SHAP profiles, SHAP dependence plots for the
most important features, and detector reliability as a function of how many of a
player's games are used (single-game vs aggregated).

Usage:
    python -m src.ml.xai_deep data/dataset_big.parquet --outdir results
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
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, DISPLAY_NAMES, WEAKNESS_CANDIDATES, FEATURE_ORIENTATION
from src.ml import plotstyle
plotstyle.apply()

RNG = 42
BANDS = [0, 1400, 1800, 2200, 4000]
BAND_LABELS = ["<1400", "1400–1800", "1800–2200", ">2200"]


def fit_shap(X, y):
    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=RNG).fit(Xi, y)
    import shap
    sv = shap.TreeExplainer(model).shap_values(Xi)
    return Xi, sv


def modal(s):
    m = s.mode()
    return m.iloc[0] if len(m) else None


def band_profiles(df, sv, outdir):
    bands = pd.cut(df["target_elo"], BANDS, labels=BAND_LABELS)
    svdf = pd.DataFrame(sv, columns=FEATURE_NAMES)
    prof = svdf.groupby(bands.values, observed=False).mean().T  # features x bands
    prof.to_csv(os.path.join(outdir, "shap_band_profiles.csv"))
    # order features by overall importance
    order = svdf.abs().mean().sort_values(ascending=False).index
    prof = prof.loc[order]
    fig, ax = plt.subplots(figsize=(8, 9))
    vmax = np.nanmax(np.abs(prof.values))
    im = ax.imshow(prof.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(prof.shape[1])); ax.set_xticklabels(prof.columns, fontsize=9)
    ax.set_yticks(range(prof.shape[0]))
    ax.set_yticklabels([DISPLAY_NAMES.get(f, f) for f in prof.index], fontsize=8)
    ax.set_title("Mean SHAP contribution by rating band\n(red = raises Elo, blue = lowers)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_shap_bands.png")); plt.close(fig)
    print("== SHAP band profiles (top features) ==")
    print(prof.head(8).round(1).to_string(), "\n")


def dependence_plots(Xi, sv, y, outdir, feats=("opening_acpl", "avg_think_time", "hanging_rate")):
    feats = [f for f in feats if f in FEATURE_NAMES]
    fig, axes = plt.subplots(1, len(feats), figsize=(5 * len(feats), 4.2))
    if len(feats) == 1:
        axes = [axes]
    for ax, f in zip(axes, feats):
        j = FEATURE_NAMES.index(f)
        sc = ax.scatter(Xi[f], sv[:, j], c=y, cmap="viridis", s=8, alpha=0.5)
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_xlabel(DISPLAY_NAMES.get(f, f)); ax.set_ylabel("SHAP value")
        ax.set_title(f"Dependence: {DISPLAY_NAMES.get(f, f)}", fontsize=9)
    fig.colorbar(sc, ax=axes, label="Elo", fraction=0.025)
    fig.savefig(os.path.join(outdir, "fig_shap_dependence.png")); plt.close(fig)


def reliability_vs_n(df, outdir, ns=(1, 2, 3, 4, 5, 6), seeds=range(8)):
    """Reference weakness = aggregated population-z over half A. Two estimators are
    scored against it on N games of half B: the modal per-game SHAP answer, and the
    SAME population-z rule that built the reference. The second arm is the reference
    rule's own test-retest -- a ceiling for this sample size, not the SHAP mechanism.
    Neither arm varies the attribution order; that is detector_order.py."""
    cand = WEAKNESS_CANDIDATES
    pm = df[cand].mean(); ps = df[cand].std(ddof=0).replace(0, 1)
    orient = pd.Series(FEATURE_ORIENTATION)[cand]  # so idxmax always means "weakest"

    # Each player needs >= 2*n games (reference half A + estimate half B of size n);
    # skip N values too large for this dataset instead of averaging over nobody.
    per_player = df.groupby("player").size()
    usable_ns = [n for n in ns if (per_player >= 2 * n).sum() >= 30]
    skipped = [n for n in ns if n not in usable_ns]
    if skipped:
        print(f"Skipping N={skipped}: fewer than 30 players have >= 2N games "
              f"(this dataset was collected with --games-per-player={int(per_player.max())}).\n")

    rows = []
    for n in usable_ns:
        a_single, a_agg = [], []
        for seed in seeds:
            for _, g in df.groupby("player"):
                if len(g) < 2 * n:
                    continue
                sh = g.sample(frac=1.0, random_state=seed)
                half = len(sh) // 2
                A, B = sh.iloc[:half], sh.iloc[half:half + n]
                ref = (((A[cand].mean() - pm) / ps) * orient).idxmax()
                est_single = modal(B["shap_weak"])
                est_agg = (((B[cand].mean() - pm) / ps) * orient).idxmax()
                a_single.append(ref == est_single)
                a_agg.append(ref == est_agg)
        rows.append({"n_games": n,
                     "agreement_single_shap": float(np.mean(a_single)),
                     "agreement_aggregated_z": float(np.mean(a_agg))})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(outdir, "xai_reliability_vs_n.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tab["n_games"], tab["agreement_single_shap"], "o-", label="modal per-game SHAP")
    ax.plot(tab["n_games"], tab["agreement_aggregated_z"], "s--",
            label="reference rule on N games (ceiling)")
    ax.axhline(1 / len(cand), color="grey", ls=":", label="chance")
    ax.set_xlabel("games used to estimate weakness (N)"); ax.set_ylabel("agreement with reference")
    ax.set_title("Agreement with the reference weakness vs games used")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_xai_reliability.png")); plt.close(fig)
    print("== Reliability vs N games ==")
    print(tab.round(3).to_string(index=False), "\n")
    return tab


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

    Xi, sv = fit_shap(X, y)
    cand_idx = [FEATURE_NAMES.index(c) for c in WEAKNESS_CANDIDATES]
    df["shap_weak"] = [WEAKNESS_CANDIDATES[i] for i in np.argmin(sv[:, cand_idx], axis=1)]

    band_profiles(df, sv, args.outdir)
    dependence_plots(Xi, sv, y.values, args.outdir)
    rel = reliability_vs_n(df, args.outdir)

    with open(os.path.join(args.outdir, "xai_deep_summary.json"), "w") as f:
        json.dump({"reliability_vs_n": rel.to_dict("records")}, f, indent=2)
    print(f"Deep-XAI artifacts written to '{args.outdir}/'.")


if __name__ == "__main__":
    main()
