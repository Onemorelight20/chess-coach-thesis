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

    cv = GroupKFold(n_splits=5)
    preds = cross_val_predict(xgb.XGBRegressor(**PARAMS), X, y, cv=cv, groups=groups, n_jobs=-1)

    r_pearson, _ = stats.pearsonr(y, preds)
    rho_spearman, _ = stats.spearmanr(y, preds)
    slope, intercept = np.polyfit(y, preds, 1)
    mae_overall = float(np.abs(y - preds).mean())

    bands = pd.cut(y, BANDS, labels=BAND_LABELS)
    tab = pd.DataFrame({"actual": y, "predicted": preds}).groupby(bands, observed=True).agg(
        n=("actual", "size"), actual_mean=("actual", "mean"), predicted_mean=("predicted", "mean"),
        mae=("actual", lambda s: np.abs(s - preds[s.index]).mean())
    ).reset_index(names="band")
    tab["bias"] = tab["predicted_mean"] - tab["actual_mean"]

    print(f"== Out-of-fold bias by band ({pool}, {len(FEATURE_NAMES)} features) ==")
    print(tab.round(1).to_string(index=False))
    print(f"\noverall MAE={mae_overall:.1f}  pearson_r={r_pearson:.3f}  "
          f"spearman_rho={rho_spearman:.3f}  slope={slope:.3f}")

    result = {
        "model": f"XGBoost, {len(FEATURE_NAMES)} features, GroupKFold(5) out-of-fold predictions, "
                 f"pool={pool}",
        "n_rows": len(df), "n_players": int(groups.nunique()),
        "overall_MAE": mae_overall,
        "pearson_r": float(r_pearson), "spearman_rho": float(rho_spearman),
        "slope": float(slope), "intercept": float(intercept),
    }
    with open(os.path.join(args.outdir, "oof_bias_by_band.json"), "w") as f:
        json.dump(result, f, indent=2)
    tab.to_csv(os.path.join(args.outdir, "oof_bias_by_band.csv"), index=False)
    print(f"Saved -> {args.outdir}/oof_bias_by_band.json, oof_bias_by_band.csv")


if __name__ == "__main__":
    main()
