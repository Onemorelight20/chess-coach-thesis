"""
Train and persist the deployable XGBoost Elo regressor: player-grouped CV for an
honest metric, then a final fit on all data. XGBoost handles missing values
natively, so think-time features can stay NaN for clock-less games.

Usage:
    python -m src.ml.train data/dataset_blitz_1600p.parquet
"""

import argparse
import json
import os

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, cross_validate

from src.features.extractor import FEATURE_NAMES

PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, random_state=42)


def train_model(parquet_path: str, model_out: str = "src/ml/models/xgboost_elo_model.json"):
    df = pd.read_parquet(parquet_path)
    df = df[df["target_elo"].between(400, 3000)]
    X, y, groups = df[FEATURE_NAMES], df["target_elo"], df["player"]
    print(f"Training on {len(df):,} rows from {groups.nunique()} players.")

    res = cross_validate(
        xgb.XGBRegressor(**PARAMS), X, y, cv=GroupKFold(5), groups=groups,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
        n_jobs=-1,
    )
    mae = -res["test_neg_mean_absolute_error"].mean()
    rmse = -res["test_neg_root_mean_squared_error"].mean()
    r2 = res["test_r2"].mean()
    print(f"\n--- Player-grouped 5-fold CV ---")
    print(f"MAE  = {mae:.1f} Elo")
    print(f"RMSE = {rmse:.1f} Elo")
    print(f"R^2  = {r2:.3f}")

    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X, y)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save_model(model_out)
    joblib.dump(list(FEATURE_NAMES), model_out.replace(".json", "_features.pkl"))
    with open(model_out.replace(".json", "_metrics.json"), "w") as f:
        json.dump({"mae": mae, "rmse": rmse, "r2": r2,
                   "n_rows": int(len(df)), "n_players": int(groups.nunique())}, f, indent=2)
    print(f"\nSaved model -> {model_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--out", default="src/ml/models/xgboost_elo_model.json")
    args = ap.parse_args()
    train_model(args.parquet, args.out)
