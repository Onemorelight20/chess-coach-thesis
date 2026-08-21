"""
Calibration study: does predicted Elo track actual Elo across the whole rating
range, or regress every player toward the training sample's mean? Compares the
legacy activity-skewed dataset against the stratified one on their 21 shared
features, plus a post-hoc isotonic recalibration.

ARCHIVAL: the legacy input (data/legacy/dataset_blitz_1600p.parquet) no longer
exists and cannot be rebuilt with the current 29-feature extractor, so this is
not re-runnable; see production_bias.py for the current model.

Usage:
    python -m src.ml.calibration --outdir results
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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ml.train import PARAMS
from src.ml import plotstyle
plotstyle.apply()

LEGACY_PATH = "data/legacy/dataset_blitz_1600p.parquet"
NEW_PATH = "data/dataset_blitz_2k.parquet"
NON_FEATURE_COLS = {"player", "game_id", "target_color", "target_elo",
                    "time_control", "pool", "eco", "result"}
BANDS = [0, 800, 1100, 1400, 1700, 2000, 2300, 3200]
BAND_LABELS = ["<800", "800-1100", "1100-1400", "1400-1700",
              "1700-2000", "2000-2300", "2300+"]


def cv_predict_with_isotonic(X, y, groups, n_splits=5):
    """
    Leakage-free CV predictions, raw and isotonic-recalibrated. The isotonic map
    is fit ONLY on each fold's train-set predictions.
    """
    cv = GroupKFold(n_splits=n_splits)
    raw = np.full(len(y), np.nan)
    calibrated = np.full(len(y), np.nan)
    for train_idx, test_idx in cv.split(X, y, groups):
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_pred = model.predict(X.iloc[train_idx])
        test_pred = model.predict(X.iloc[test_idx])
        raw[test_idx] = test_pred

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train_pred, y.iloc[train_idx].to_numpy())
        calibrated[test_idx] = iso.predict(test_pred)
    return raw, calibrated


def per_band_table(actual, predicted, label):
    bands = pd.cut(actual, BANDS, labels=BAND_LABELS)
    g = pd.DataFrame({"actual": actual, "predicted": predicted}).groupby(bands, observed=True)
    tab = g.agg(n=("actual", "size"), actual_mean=("actual", "mean"),
               predicted_mean=("predicted", "mean")).reset_index(names="band")
    tab["bias"] = tab["predicted_mean"] - tab["actual_mean"]
    tab.insert(0, "dataset", label)
    return tab


def mae(actual, predicted):
    return float(np.abs(np.asarray(actual) - np.asarray(predicted)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", default=LEGACY_PATH,
                    help="old, activity-skewed single-pool dataset")
    ap.add_argument("--new", default=NEW_PATH,
                    help="new, rating-stratified single-pool dataset")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    old = pd.read_parquet(args.legacy).reset_index(drop=True)
    new = pd.read_parquet(args.new).reset_index(drop=True)
    shared = [c for c in old.columns if c in new.columns and c not in NON_FEATURE_COLS]

    old = old[old.target_elo.between(400, 3000)].dropna(subset=["acpl"]).reset_index(drop=True)
    new = new[new.target_elo.between(400, 3000)].dropna(subset=["acpl"]).reset_index(drop=True)
    print(f"Shared features (legacy ∩ stratified): {len(shared)}")
    print(f"legacy      : {len(old):,} rows, {old.player.nunique()} players, "
          f"elo {int(old.target_elo.min())}-{int(old.target_elo.max())} "
          f"(<900: {(old.target_elo < 900).mean()*100:.1f}%)")
    print(f"stratified  : {len(new):,} rows, {new.player.nunique()} players, "
          f"elo {int(new.target_elo.min())}-{int(new.target_elo.max())} "
          f"(<900: {(new.target_elo < 900).mean()*100:.1f}%)\n")

    old_raw, old_cal = cv_predict_with_isotonic(old[shared], old.target_elo, old.player)
    new_raw, new_cal = cv_predict_with_isotonic(new[shared], new.target_elo, new.player)

    tabs = [
        per_band_table(old.target_elo, old_raw, "legacy (skewed sample)"),
        per_band_table(new.target_elo, new_raw, "stratified (400-2800)"),
        per_band_table(new.target_elo, new_cal, "stratified + isotonic"),
    ]
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(os.path.join(args.outdir, "calibration_by_band.csv"), index=False)
    print("== Calibration by rating band (predicted vs actual mean) ==")
    print(tab.round(1).to_string(index=False), "\n")

    low_old = old.target_elo < 1100
    low_new = new.target_elo < 1100
    summary = {
        "legacy_overall_MAE": mae(old.target_elo, old_raw),
        "legacy_low_end_MAE": mae(old.target_elo[low_old], old_raw[low_old.to_numpy()]),
        "stratified_overall_MAE": mae(new.target_elo, new_raw),
        "stratified_low_end_MAE": mae(new.target_elo[low_new], new_raw[low_new.to_numpy()]),
        "stratified_isotonic_overall_MAE": mae(new.target_elo, new_cal),
        "stratified_isotonic_low_end_MAE": mae(new.target_elo[low_new], new_cal[low_new.to_numpy()]),
    }
    print("== Summary MAE ==")
    for k, v in summary.items():
        print(f"  {k:34s}: {v:.1f}")
    low_lift = (1 - summary["stratified_low_end_MAE"] / summary["legacy_low_end_MAE"]) * 100
    print(f"\n  -> stratified sampling cuts low-end (<1100) MAE by {low_lift:.0f}% vs the skewed sample\n")
    summary["low_end_MAE_reduction_pct"] = float(low_lift)

    with open(os.path.join(args.outdir, "calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure: two-panel calibration curves ----
    lims = [min(old.target_elo.min(), new.target_elo.min()) - 50,
           max(old.target_elo.max(), new.target_elo.max()) + 50]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for t, label, color, marker in [(tabs[0], "legacy (skewed sample)", "#C44E52", "o"),
                                    (tabs[1], "stratified (400-2800)", "#4C72B0", "s")]:
        ax1.plot(t["actual_mean"], t["predicted_mean"], marker=marker, color=color, label=label)
    ax1.plot(lims, lims, "k--", lw=1, alpha=0.5, label="perfect calibration")
    ax1.set_xlabel("Actual Elo (band mean)"); ax1.set_ylabel("Predicted Elo (band mean)")
    ax1.set_title("Sampling fix: skewed vs stratified training data")
    ax1.legend(fontsize=8); ax1.set_xlim(lims); ax1.set_ylim(lims)

    for t, label, color, marker in [(tabs[1], "stratified, raw", "#4C72B0", "s"),
                                    (tabs[2], "stratified + isotonic", "#55A868", "^")]:
        ax2.plot(t["actual_mean"], t["predicted_mean"], marker=marker, color=color, label=label)
    ax2.plot(lims, lims, "k--", lw=1, alpha=0.5, label="perfect calibration")
    ax2.set_xlabel("Actual Elo (band mean)"); ax2.set_ylabel("Predicted Elo (band mean)")
    ax2.set_title("Post-hoc isotonic recalibration")
    ax2.legend(fontsize=8); ax2.set_xlim(lims); ax2.set_ylim(lims)

    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_calibration.png"))
    plt.close(fig)
    print(f"Saved -> {args.outdir}/fig_calibration.png, calibration_by_band.csv, calibration_summary.json")


if __name__ == "__main__":
    main()
