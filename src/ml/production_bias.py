"""
Out-of-fold prediction bias by rating band for the current, full 29-feature
production model. Narrower than calibration.py, whose legacy-vs-stratified
comparison is archival and no longer re-runnable.

Usage:
    python -m src.ml.production_bias --outdir results
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold, cross_val_predict
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES
from src.ml.train import PARAMS
from src.ml.mixed_training import mixed_training_set

BANDS = [0, 800, 1100, 1400, 1700, 2000, 2300, 3200]
BAND_LABELS = ["<800", "800-1100", "1100-1400", "1400-1700",
              "1700-2000", "2000-2300", "2300+"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset_blitz_2k.parquet")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.dataset)
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    X, y, groups = df[FEATURE_NAMES], df["target_elo"], df["player"]
    pool = df["pool"].mode().iloc[0] if "pool" in df.columns else os.path.basename(args.dataset)

    # the deployed configuration: mixed training set (single games + N-game series)
    frame = X.copy(); frame["target_elo"] = y.values; frame["player"] = groups.values
    Xm, ym, gm, nm = mixed_training_set(frame)
    cv = GroupKFold(n_splits=5)
    preds = cross_val_predict(xgb.XGBRegressor(**PARAMS), Xm, ym, cv=cv, groups=gm, n_jobs=-1)

    result = {"model": f"XGBoost, {len(FEATURE_NAMES)} features, mixed training set, "
                       f"GroupKFold(5) out-of-fold predictions, pool={pool}",
              "n_games": len(df), "n_rows": int(len(ym)), "n_players": int(groups.nunique())}
    tabs = []
    for N in (1, 8):
        m = (nm == N).to_numpy()
        yy, pp = ym[m].reset_index(drop=True), pd.Series(preds[m])
        r_pearson, _ = stats.pearsonr(yy, pp)
        rho_spearman, _ = stats.spearmanr(yy, pp)
        slope, intercept = np.polyfit(yy, pp, 1)
        bands = pd.cut(yy, BANDS, labels=BAND_LABELS)
        tab = pd.DataFrame({"actual": yy, "predicted": pp}).groupby(bands, observed=True).agg(
            n=("actual", "size"), actual_mean=("actual", "mean"), predicted_mean=("predicted", "mean"),
            mae=("actual", lambda s: np.abs(s - pp[s.index]).mean())
        ).reset_index(names="band")
        tab["bias"] = tab["predicted_mean"] - tab["actual_mean"]
        tab.insert(0, "n_games", N)
        tabs.append(tab)
        print(f"== Out-of-fold bias by band ({pool}, N={N}) ==")
        print(tab.round(1).to_string(index=False))
        print(f"overall MAE={np.abs(yy - pp).mean():.1f}  pearson_r={r_pearson:.3f}  "
              f"spearman_rho={rho_spearman:.3f}  slope={slope:.3f}\n")
        result[f"N{N}"] = {"overall_MAE": float(np.abs(yy - pp).mean()), "n_rows": int(m.sum()),
                           "pearson_r": float(r_pearson), "spearman_rho": float(rho_spearman),
                           "slope": float(slope), "intercept": float(intercept)}
    tab = pd.concat(tabs, ignore_index=True)
    with open(os.path.join(args.outdir, "oof_bias_by_band.json"), "w") as f:
        json.dump(result, f, indent=2)
    tab.to_csv(os.path.join(args.outdir, "oof_bias_by_band.csv"), index=False)
    print(f"Saved -> {args.outdir}/oof_bias_by_band.json, oof_bias_by_band.csv")


if __name__ == "__main__":
    main()
