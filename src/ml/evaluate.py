"""
Leakage-free evaluation harness for chess Elo estimation. Writes CSV tables and
PNG figures to --outdir: leakage demonstration (KFold vs GroupKFold), model
comparison, VIF + correlation, feature-group ablation, noise-ceiling curve, and
global importance (XGBoost gain + SHAP).

Usage:
    python -m src.ml.evaluate data/dataset_blitz_1600p.parquet --outdir results
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
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES, DISPLAY_NAMES
from src.ml import plotstyle
plotstyle.apply()

RNG = 42
N_SPLITS = 5
SCORERS = {"MAE": "neg_mean_absolute_error",
           "RMSE": "neg_root_mean_squared_error",
           "R2": "r2"}

FEATURE_GROUPS = {
    "accuracy": ["acpl", "inaccuracy_rate", "mistake_rate", "blunder_rate"],
    "phase": ["opening_acpl", "middlegame_acpl", "endgame_acpl"],
    "situational": ["conversion_acpl", "defense_acpl"],
    "tactical": ["hanging_rate", "pin_exposure", "check_given_rate", "capture_rate"],
    "positional": ["king_danger", "mobility_avg", "center_control_avg", "material_volatility"],
    "style": ["engine_match_rate"],
    "time": ["think_time_consistency", "avg_think_time", "time_pressure_rate"],
    # no extra engine cost
    "winprob": ["wp_loss_mean", "wp_blunder_rate", "cpl_std"],
    # MultiPV; ~1.2-1.3x engine cost
    "agreement": ["top1_agree", "top3_agree", "avg_move_rank", "acpl_critical", "critical_agree"],
}
BASE21 = [f for f in FEATURE_NAMES
          if f not in FEATURE_GROUPS["winprob"] + FEATURE_GROUPS["agreement"]]


# ---------------------------------------------------------------- model zoo ---
def make_models():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline([("imp", imp()), ("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "RandomForest": Pipeline([("imp", imp()),
                                  ("m", RandomForestRegressor(n_estimators=300, random_state=RNG, n_jobs=-1))]),
        "MLP": Pipeline([("imp", imp()), ("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                                            early_stopping=True, random_state=RNG))]),
        "XGBoost": Pipeline([("imp", imp()),
                             ("m", xgb.XGBRegressor(n_estimators=300, max_depth=5,
                                                    learning_rate=0.05, subsample=0.8,
                                                    colsample_bytree=0.8, random_state=RNG))]),
    }


def cv_table(models, X, y, cv, groups=None):
    rows = []
    for name, model in models.items():
        cvres = cross_validate(model, X, y, cv=cv, groups=groups, scoring=SCORERS, n_jobs=-1)
        rows.append({
            "model": name,
            "MAE": -cvres["test_MAE"].mean(), "MAE_std": cvres["test_MAE"].std(),
            "RMSE": -cvres["test_RMSE"].mean(), "RMSE_std": cvres["test_RMSE"].std(),
            "R2": cvres["test_R2"].mean(), "R2_std": cvres["test_R2"].std(),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- VIF ------
def compute_vif(X):
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xi = StandardScaler().fit_transform(Xi)
    vifs = []
    for j in range(Xi.shape[1]):
        others = np.delete(Xi, j, axis=1)
        r2 = LinearRegression().fit(others, Xi[:, j]).score(others, Xi[:, j])
        vifs.append(1.0 / (1.0 - r2) if r2 < 1 else np.inf)
    return pd.DataFrame({"feature": list(X.columns), "VIF": vifs}).sort_values("VIF", ascending=False)


# ------------------------------------------------------------- ablation -------
def ablation(X, y, groups):
    cv = GroupKFold(n_splits=N_SPLITS)
    base_model = lambda cols: Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("m", xgb.XGBRegressor(n_estimators=300, max_depth=5,
                                                               learning_rate=0.05, subsample=0.8,
                                                               colsample_bytree=0.8, random_state=RNG))])
    full = -cross_validate(base_model(FEATURE_NAMES), X, y, cv=cv, groups=groups,
                           scoring="neg_mean_absolute_error", n_jobs=-1)["test_score"].mean()
    rows = [{"removed_group": "(none) full model", "MAE": full, "delta_MAE": 0.0}]
    for gname, cols in FEATURE_GROUPS.items():
        keep = [c for c in FEATURE_NAMES if c not in cols]
        mae = -cross_validate(base_model(keep), X[keep], y, cv=cv, groups=groups,
                              scoring="neg_mean_absolute_error", n_jobs=-1)["test_score"].mean()
        rows.append({"removed_group": gname, "MAE": mae, "delta_MAE": mae - full})
    return pd.DataFrame(rows)


# ------------------------------------------------------- additive ablation ----
def additive_ablation(X, y, groups, outdir):
    """
    Incremental value of the two groups added beyond the original 21:
    base (21) -> +win-probability (24) -> +engine agreement (29, full).
    """
    cv = GroupKFold(n_splits=N_SPLITS)
    model = lambda cols: Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("m", xgb.XGBRegressor(n_estimators=300, max_depth=5,
                                                          learning_rate=0.05, subsample=0.8,
                                                          colsample_bytree=0.8, random_state=RNG))])
    stages = [
        ("base (21 features)", BASE21),
        ("+ win-probability (24)", BASE21 + FEATURE_GROUPS["winprob"]),
        ("+ engine agreement (29, full)", FEATURE_NAMES),
    ]
    rows, base_mae = [], None
    for name, cols in stages:
        res = cross_validate(model(cols), X[cols], y, cv=cv, groups=groups,
                             scoring=["neg_mean_absolute_error", "r2"], n_jobs=-1)
        mae = -res["test_neg_mean_absolute_error"].mean()
        r2 = res["test_r2"].mean()
        base_mae = mae if base_mae is None else base_mae
        rows.append({"stage": name, "n_features": len(cols), "MAE": mae, "R2": r2,
                     "delta_MAE_vs_base": mae - base_mae})
    tab = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(range(len(tab)), tab["MAE"], color="#C44E52")
    ax1.set_xticks(range(len(tab))); ax1.set_xticklabels(tab["stage"], rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("MAE (Elo)"); ax1.set_title("Error vs feature groups added")
    ax2.bar(range(len(tab)), tab["R2"], color="#4C72B0")
    ax2.set_xticks(range(len(tab))); ax2.set_xticklabels(tab["stage"], rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("R2"); ax2.set_title("Accuracy vs feature groups added")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_feature_groups.png"))
    plt.close(fig)
    return tab


# ------------------------------------------------- noise-ceiling experiment ---
def aggregation_curve(df, ns, seeds=(1, 2, 3)):
    """
    Aggregate N games per player (MEDIAN features, median Elo) and 5-fold CV.
    This is the robustness variant; the curve cited by the thesis is the MEAN one
    from ml/aggregation_curve.py, which matches the aggregator app.py deploys.
    """
    out = []
    model = lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("m", xgb.XGBRegressor(n_estimators=300, max_depth=5,
                                                     learning_rate=0.05, subsample=0.8,
                                                     colsample_bytree=0.8, random_state=RNG))])
    for n in ns:
        maes, r2s, npl = [], [], []
        for seed in seeds:
            rows = []
            for _, g in df.groupby("player"):
                if len(g) < n:
                    continue
                s = g.sample(n, random_state=seed)
                agg = s[FEATURE_NAMES].median().to_dict()
                agg["target_elo"] = s["target_elo"].median()
                rows.append(agg)
            agg_df = pd.DataFrame(rows)
            npl.append(len(agg_df))
            Xa, ya = agg_df[FEATURE_NAMES], agg_df["target_elo"]
            cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            cvres = cross_validate(model(), Xa, ya, cv=cv,
                                   scoring=SCORERS, n_jobs=-1)
            maes.append(-cvres["test_MAE"].mean())
            r2s.append(cvres["test_R2"].mean())
        out.append({"games_per_player": n, "n_players": int(np.mean(npl)),
                    "MAE": np.mean(maes), "R2": np.mean(r2s)})
    return pd.DataFrame(out)


# ----------------------------------------------------------- importance -------
def importances(X, y, outdir):
    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=RNG)
    model.fit(Xi, y)
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xi)
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"feature": list(X.columns),
                        "xgb_gain": model.feature_importances_,
                        "shap_mean_abs": mean_abs}).sort_values("shap_mean_abs", ascending=False)
    # figure
    fig, ax = plt.subplots(figsize=(8, 5))
    d = imp.sort_values("shap_mean_abs")
    ax.barh([DISPLAY_NAMES.get(f, f) for f in d["feature"]], d["shap_mean_abs"], color="#4C72B0")
    ax.set_xlabel("mean(|SHAP value|)  — average impact on predicted Elo")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_shap_global.png")); plt.close(fig)
    return imp


# --------------------------------------------------- per-pool (time control) --
def per_pool_analysis(df, outdir):
    """Time-control study: per-pool best-model quality and aggregation curve."""
    order = ["bullet", "blitz", "rapid", "classical"]
    pools = [p for p in order if p in set(df["pool"])]
    rows, curves = [], {}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for pool in pools:
        sub = df[df["pool"] == pool]
        Xp, yp, gp = sub[FEATURE_NAMES], sub["target_elo"], sub["player"]
        cv = GroupKFold(N_SPLITS)
        model = make_models()["XGBoost"]
        res = cross_validate(model, Xp, yp, cv=cv, groups=gp, scoring=SCORERS, n_jobs=-1)
        ns = [n for n in [1, 3, 5, 8, 10] if (sub.groupby("player").size() >= n).sum() >= 30]
        cur = aggregation_curve(sub, ns) if ns else None
        rows.append({"pool": pool, "n_rows": len(sub), "n_players": gp.nunique(),
                     "elo_mean": round(yp.mean()), "elo_std": round(yp.std()),
                     "MAE_pergame": -res["test_MAE"].mean(), "R2_pergame": res["test_R2"].mean(),
                     "R2_aggregated": (cur["R2"].max() if cur is not None else float("nan"))})
        if cur is not None:
            curves[pool] = cur
            ax.plot(cur["games_per_player"], cur["R2"], "o-", label=pool)
    ax.set_xlabel("games aggregated per player"); ax.set_ylabel("R2")
    ax.set_title("Aggregation curve by time-control pool"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_pool_aggregation.png")); plt.close(fig)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(outdir, "pool_comparison.csv"), index=False)
    print("== Per-pool comparison (time-control study) ==")
    print(tab.round(3).to_string(index=False), "\n")
    return tab


# --------------------------------------------- cross-pool summary (--pools) ---
def pool_summary(paths, outdir):
    """
    One row per separately-collected single-pool dataset, using the same
    GroupKFold XGBoost as everywhere else in this module.
    """
    rows = []
    for path in paths:
        d = pd.read_parquet(path)
        d = d[d["target_elo"].between(400, 3000)]
        Xp, yp, gp = d[FEATURE_NAMES], d["target_elo"], d["player"]
        pool_name = d["pool"].mode().iloc[0] if "pool" in d.columns else os.path.basename(path)
        res = cross_validate(make_models()["XGBoost"], Xp, yp, cv=GroupKFold(N_SPLITS),
                             groups=gp, scoring=SCORERS, n_jobs=-1)
        rows.append({"pool": pool_name, "n_rows": len(d), "n_players": int(gp.nunique()),
                     "elo_mean": round(yp.mean()), "elo_std": round(yp.std()),
                     "MAE": -res["test_MAE"].mean(), "RMSE": -res["test_RMSE"].mean(),
                     "R2": res["test_R2"].mean()})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(outdir, "pool_summary.csv"), index=False)
    print("== Cross-pool summary (stratified 400-2800 datasets) ==")
    print(tab.round(3).to_string(index=False), "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.bar(tab["pool"], tab["MAE"], color="#C44E52")
    ax1.set_ylabel("MAE (Elo)"); ax1.set_title("Prediction error by pool")
    ax2.bar(tab["pool"], tab["R2"], color="#4C72B0")
    ax2.set_ylabel("R2"); ax2.set_title("Prediction R2 by pool")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_pool_summary.png"))
    plt.close(fig)
    print(f"Saved -> {outdir}/pool_summary.csv, fig_pool_summary.png")
    return tab


# ----------------------------------------------------------------- main -------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="?", default=None)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--pools", default=None,
                    help="comma-separated single-pool parquet paths "
                         "(e.g. data/dataset_bullet_2k.parquet,data/dataset_blitz_2k.parquet,...) "
                         "-> lightweight cross-pool summary only, skips the full analysis below")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.pools:
        pool_summary([p.strip() for p in args.pools.split(",")], args.outdir)
        return
    if not args.parquet:
        ap.error("either a parquet path or --pools is required")

    df = pd.read_parquet(args.parquet)
    df = df[df["target_elo"].between(400, 3000)]
    X, y, groups = df[FEATURE_NAMES], df["target_elo"], df["player"]
    print(f"Loaded {len(df):,} rows | {groups.nunique()} players | "
          f"Elo mean={y.mean():.0f} std={y.std():.0f}\n")

    # dataset rating distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(y, bins=40, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Target Elo"); ax.set_ylabel("Count")
    ax.set_title(f"Dataset rating distribution (n={len(y):,} rows, {groups.nunique()} players)")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "fig_dataset_distribution.png"))
    plt.close(fig)

    # 0. Univariate signal (Spearman, robust to skew)
    sp = pd.DataFrame({"feature": FEATURE_NAMES,
                       "spearman_vs_elo": [spearmanr(X[f], y, nan_policy="omit").statistic
                                           for f in FEATURE_NAMES]}).sort_values(
        "spearman_vs_elo", key=lambda s: s.abs(), ascending=False)
    sp.to_csv(os.path.join(args.outdir, "feature_signal.csv"), index=False)
    print("== Univariate Spearman correlation with Elo ==")
    print(sp.to_string(index=False), "\n")

    # 1. Leakage demonstration (same model, two CV schemes)
    xgbm = make_models()["XGBoost"]
    lk = []
    for label, cv, grp in [("Random KFold (LEAKY)", KFold(N_SPLITS, shuffle=True, random_state=RNG), None),
                           ("Player GroupKFold", GroupKFold(N_SPLITS), groups)]:
        t = cv_table({"XGBoost": xgbm}, X, y, cv, grp).iloc[0]
        lk.append({"cv_scheme": label, "MAE": t["MAE"], "RMSE": t["RMSE"], "R2": t["R2"]})
    lk = pd.DataFrame(lk); lk.to_csv(os.path.join(args.outdir, "leakage_comparison.csv"), index=False)
    print("== Leakage demonstration (XGBoost) ==")
    print(lk.to_string(index=False), "\n")

    # 2. Model comparison (honest, grouped)
    comp = cv_table(make_models(), X, y, GroupKFold(N_SPLITS), groups)
    comp.to_csv(os.path.join(args.outdir, "model_comparison.csv"), index=False)
    print("== Model comparison (player-grouped 5-fold CV) ==")
    print(comp.round(3).to_string(index=False), "\n")
    # figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(comp["model"], comp["MAE"], yerr=comp["MAE_std"], color="#55A868", capsize=4)
    ax.set_ylabel("MAE (Elo points)"); ax.set_title("Model comparison (GroupKFold)")
    plt.xticks(rotation=20, ha="right"); fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_model_comparison.png")); plt.close(fig)

    # 3. VIF + correlation heatmap
    vif = compute_vif(X); vif.to_csv(os.path.join(args.outdir, "vif.csv"), index=False)
    print("== Multicollinearity (VIF; >5 is concerning) ==")
    print(vif.round(2).to_string(index=False), "\n")
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_NAMES))); ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7); ax.set_yticklabels(FEATURE_NAMES, fontsize=7)
    fig.colorbar(im, fraction=0.046); ax.set_title("Feature correlation matrix")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "fig_corr.png")); plt.close(fig)

    # 4. Ablation
    abl = ablation(X, y, groups); abl.to_csv(os.path.join(args.outdir, "ablation.csv"), index=False)
    print("== Feature-group ablation (XGBoost GroupKFold) ==")
    print(abl.round(2).to_string(index=False), "\n")

    # 4b. Additive ablation: base21 -> +winprob -> +agreement
    add_abl = additive_ablation(X, y, groups, args.outdir)
    add_abl.to_csv(os.path.join(args.outdir, "ablation_additive.csv"), index=False)
    print("== Additive feature-group ablation ==")
    print(add_abl.round(3).to_string(index=False), "\n")

    # 5. Noise-ceiling curve
    ns = [n for n in [1, 2, 3, 5, 8, 10] if (df.groupby("player").size() >= n).sum() >= 30]
    curve = aggregation_curve(df, ns); curve.to_csv(os.path.join(args.outdir, "aggregation_curve.csv"), index=False)
    print("== Noise ceiling: N games aggregated per player ==")
    print(curve.round(3).to_string(index=False), "\n")
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(curve["games_per_player"], curve["MAE"], "o-", color="#C44E52", label="MAE")
    ax1.set_xlabel("games aggregated per player"); ax1.set_ylabel("MAE (Elo)", color="#C44E52")
    ax2 = ax1.twinx(); ax2.plot(curve["games_per_player"], curve["R2"], "s--", color="#4C72B0", label="R2")
    ax2.set_ylabel("R2", color="#4C72B0"); ax1.set_title("Prediction improves as games are aggregated")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "fig_aggregation.png")); plt.close(fig)

    # 5b. Per-pool comparison (time-control study)
    if "pool" in df.columns and df["pool"].nunique() > 1:
        per_pool_analysis(df, args.outdir)

    # 6. Importance
    imp = importances(X, y, args.outdir); imp.to_csv(os.path.join(args.outdir, "feature_importance.csv"), index=False)
    print("== Global feature importance ==")
    print(imp.round(4).to_string(index=False), "\n")

    summary = {
        "n_rows": int(len(df)), "n_players": int(groups.nunique()),
        "elo_mean": float(y.mean()), "elo_std": float(y.std()),
        "leakage": lk.to_dict("records"),
        "best_model": comp.sort_values("MAE").iloc[0]["model"],
        "best_model_MAE": float(comp.sort_values("MAE").iloc[0]["MAE"]),
        "aggregation_best_R2": float(curve["R2"].max()),
        "additive_ablation": add_abl.to_dict("records"),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"All tables + figures written to '{args.outdir}/'.")


if __name__ == "__main__":
    main()
