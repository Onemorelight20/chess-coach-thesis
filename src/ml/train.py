"""
Train and persist the deployable XGBoost Elo regressor on the MIXED training
set (src/ml/mixed_training.py): every single game plus mean vectors of random
N-game series of the same players, so the model sees at training time the same
kind of input app.py feeds it (the mean vector of 1..8 games). Player-grouped CV
for an honest metric, reported per series length, then a final fit on all rows.
XGBoost handles missing values natively, so think-time features can stay NaN
for clock-less games.

Usage:
    python -m src.ml.train data/dataset_blitz_2k.parquet --out src/ml/models/xgboost_elo_blitz.json
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from src.features.extractor import FEATURE_NAMES
from src.ml.mixed_training import mixed_training_set, describe

PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, random_state=42)


def train_model(parquet_path: str, model_out: str = "src/ml/models/xgboost_elo_model.json"):
    df = pd.read_parquet(parquet_path)
    df = df[df["target_elo"].between(400, 3000)].reset_index(drop=True)
    X, y, groups, n_games = mixed_training_set(df)
    print(f"Training rows: {describe(n_games)} from {groups.nunique()} players "
          f"({len(df):,} games in the pool).")

    # Player-grouped 5-fold CV on the mixed rows; metrics per series length, so the
    # single-game figure stays comparable with the model-comparison table.
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = xgb.XGBRegressor(**PARAMS).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    by_n = {}
    for N in sorted(n_games.unique()):
        mask = (n_games == N).to_numpy()
        by_n[int(N)] = {"mae": float(mean_absolute_error(y[mask], oof[mask])),
                        "rmse": float(np.sqrt(np.mean((y[mask] - oof[mask]) ** 2))),
                        "r2": float(r2_score(y[mask], oof[mask])), "n_rows": int(mask.sum())}
    print(f"\n--- Player-grouped 5-fold CV (out-of-fold, by series length) ---")
    for N, v in by_n.items():
        print(f"N={N}: MAE = {v['mae']:.1f}  RMSE = {v['rmse']:.1f}  R^2 = {v['r2']:.3f}  ({v['n_rows']:,} rows)")

    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save_model(model_out)
    joblib.dump(list(FEATURE_NAMES), model_out.replace(".json", "_features.pkl"))
    single = by_n[1]
    with open(model_out.replace(".json", "_metrics.json"), "w") as f:
        json.dump({"mae": single["mae"], "rmse": single["rmse"], "r2": single["r2"],
                   "by_n_games": by_n, "training": "mixed (single games + N-game series)",
                   "n_rows": int(len(X)), "n_games": int(len(df)),
                   "n_players": int(groups.nunique())}, f, indent=2)
    print(f"\nSaved model -> {model_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--out", default="src/ml/models/xgboost_elo_model.json")
    args = ap.parse_args()
    train_model(args.parquet, args.out)
