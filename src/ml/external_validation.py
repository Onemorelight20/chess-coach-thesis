"""
External validation of the deployed, Lichess-trained model.

Part A -- bias comparison. Held-out Lichess players and the Chess.com sample from
collect_crossplatform.py both go through the frozen model and app.py's pipeline,
separating the model's own residual bias from the platform-scale gap.
Part B -- post-hoc recalibration. Fits isotonic vs linear under K-fold CV and
saves the winning Lichess-scale -> Chess.com-scale map for app.py.

Usage:
    python -m src.ml.external_validation --outdir results
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
import chess.pgn
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_layer.collector import scan_and_bucket, select_players_stratified, parse_increment
from src.engine.analyzer import EngineAnalyzer
from src.features.extractor import FeatureExtractor, FEATURE_NAMES
from src.features.aggregate import aggregate_features
from src.ml.model import ELOModel
from src.ml import plotstyle
plotstyle.apply()

LICHESS_PGN = "lichess_db_standard_rated_2026-01.pgn"
TRAIN_PARQUET_TMPL = "data/dataset_{pool}_2k.parquet"
CHESSCOM_PARQUET_TMPL = "data/crossplatform/dataset_chesscom_{pool}.parquet"
RECAL_PATH_TMPL = "src/ml/models/chesscom_recalibration_{pool}.pkl"
POOL = "blitz"
N_GAMES = 10
BIN_WIDTH = 200
RATING_MIN, RATING_MAX = 400, 2800


def held_out_lichess_bias(pool, n_per_bin=1, max_scan=1_500_000):
    """Predicted vs actual Elo on Lichess players NEVER used to train `pool`'s model."""
    known = set(pd.read_parquet(TRAIN_PARQUET_TMPL.format(pool=pool))["player"].unique())
    print(f"Excluding {len(known)} training players; scanning for unseen {pool} players...")
    buckets, scanned, kept = scan_and_bucket(
        LICHESS_PGN, pool, players=10**9, games_per_player=N_GAMES, max_scan=max_scan)
    unseen = {p: v for p, v in buckets.items() if p not in known}
    chosen = select_players_stratified(
        unseen, N_GAMES, RATING_MIN, RATING_MAX, BIN_WIDTH, n_per_bin)
    print(f"  scanned {scanned:,} games -> {len(chosen)} held-out players across rating bins.")

    analyzer = EngineAnalyzer(time_limit=0.05)  # exact live-app settings
    model = ELOModel(pool=pool)
    fh = open(LICHESS_PGN, encoding="utf-8")
    rows = []
    for player, entries in chosen:
        feats_list, true_elos = [], []
        for offset, color, elo, tc in entries[:N_GAMES]:
            fh.seek(offset)
            game = chess.pgn.read_game(fh)
            if game is None:
                continue
            results = analyzer.analyze_game(game)
            feats_list.append(FeatureExtractor(target_player_color=color).extract(
                results, increment=parse_increment(tc)))
            true_elos.append(elo)
        if not feats_list:
            continue
        agg = aggregate_features(feats_list, method="mean")
        pred = model.predict_and_explain(agg)
        actual = float(np.median(true_elos))
        rows.append({"player": player, "platform": "lichess", "n_games": len(feats_list),
                    "actual_elo": actual, "predicted_elo": pred["predicted_elo"],
                    "bias": pred["predicted_elo"] - actual})
    analyzer.close()
    fh.close()
    return pd.DataFrame(rows)


def chesscom_bias(parquet_path, pool):
    """Predicted vs actual Chess.com rating, aggregating each cached player's games."""
    df = pd.read_parquet(parquet_path)
    model = ELOModel(pool=pool)
    rows = []
    for player, g in df.groupby("player"):
        feats_list = g[FEATURE_NAMES].to_dict("records")
        agg = aggregate_features(feats_list, method="mean")
        pred = model.predict_and_explain(agg)
        actual = float(g["target_elo"].median())
        rows.append({"player": player, "platform": "chess.com", "n_games": len(g),
                    "actual_elo": actual, "predicted_elo": pred["predicted_elo"],
                    "bias": pred["predicted_elo"] - actual})
    return pd.DataFrame(rows)


def recalibrate(df, n_splits=5, seed=42):
    """
    K-fold, leakage-free comparison of isotonic vs linear post-hoc recalibration
    (raw Lichess-scale prediction -> actual Chess.com Elo).
    """
    n_splits = max(2, min(n_splits, df["player"].nunique()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    raw_pred = df["predicted_elo"].to_numpy(dtype=float)
    actual = df["actual_elo"].to_numpy(dtype=float)

    iso_pred = np.full(len(df), np.nan)
    lin_pred = np.full(len(df), np.nan)
    for train_idx, test_idx in kf.split(raw_pred):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_pred[train_idx], actual[train_idx])
        iso_pred[test_idx] = iso.predict(raw_pred[test_idx])

        lin = LinearRegression()
        lin.fit(raw_pred[train_idx].reshape(-1, 1), actual[train_idx])
        lin_pred[test_idx] = lin.predict(raw_pred[test_idx].reshape(-1, 1))

    summary = {
        "n_players": len(df), "n_splits": n_splits,
        "mae_raw": float(np.abs(raw_pred - actual).mean()),
        "mae_isotonic": float(np.abs(iso_pred - actual).mean()),
        "mae_linear": float(np.abs(lin_pred - actual).mean()),
        "bias_raw": float((raw_pred - actual).mean()),
        "bias_isotonic": float((iso_pred - actual).mean()),
        "bias_linear": float((lin_pred - actual).mean()),
    }
    return summary, iso_pred, lin_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL, choices=["bullet", "blitz", "rapid"])
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--chesscom-parquet", default=None)
    ap.add_argument("--per-bin", type=int, default=1,
                    help="held-out Lichess players per 200-Elo bin")
    ap.add_argument("--max-scan", type=int, default=1_500_000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cc_path = args.chesscom_parquet or CHESSCOM_PARQUET_TMPL.format(pool=args.pool)

    print("=== Part 1: held-out Lichess players (never seen in training) ===")
    lichess_df = held_out_lichess_bias(args.pool, args.per_bin, args.max_scan)
    print(lichess_df[["player", "actual_elo", "predicted_elo", "bias"]].to_string(index=False))

    print(f"\n=== Part 2: Chess.com sample (cross-platform, {cc_path}) ===")
    cc_df = chesscom_bias(cc_path, args.pool)
    print(cc_df[["player", "actual_elo", "predicted_elo", "bias"]].to_string(index=False))

    combined = pd.concat([lichess_df, cc_df], ignore_index=True)
    combined.to_csv(os.path.join(args.outdir, "external_validation_bias.csv"), index=False)

    part_a = {
        "pool": args.pool,
        "lichess_n": len(lichess_df), "lichess_mean_bias": float(lichess_df.bias.mean()),
        "lichess_mae": float(lichess_df.bias.abs().mean()),
        "chesscom_n": len(cc_df), "chesscom_mean_bias": float(cc_df.bias.mean()),
        "chesscom_mae": float(cc_df.bias.abs().mean()),
    }
    print("\n=== Summary (Part A) ===")
    print(f"Lichess held-out ({part_a['lichess_n']} players):  "
          f"mean bias {part_a['lichess_mean_bias']:+.0f}, MAE {part_a['lichess_mae']:.0f}")
    print(f"Chess.com sample ({part_a['chesscom_n']} players): "
          f"mean bias {part_a['chesscom_mean_bias']:+.0f}, MAE {part_a['chesscom_mae']:.0f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(lichess_df.actual_elo, lichess_df.bias,
              label=f"Lichess held-out (n={len(lichess_df)})", color=plotstyle.PRIMARY, s=45)
    ax.scatter(cc_df.actual_elo, cc_df.bias,
              label=f"Chess.com (n={len(cc_df)})", color=plotstyle.SECONDARY, s=45)
    ax.axhline(0, color="grey", ls="--", lw=1, label="no bias")
    ax.set_xlabel("Actual Elo"); ax.set_ylabel("Bias (predicted - actual)")
    ax.set_title(f"Prediction bias: unseen Lichess players vs Chess.com ({args.pool})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_external_validation_bias.png"))
    plt.close(fig)

    print("\n=== Part B: post-hoc cross-platform recalibration (Chess.com) ===")
    recal_summary, iso_pred, lin_pred = recalibrate(cc_df)
    print(json.dumps(recal_summary, indent=2))
    winner = "isotonic" if recal_summary["mae_isotonic"] <= recal_summary["mae_linear"] else "linear"
    print(f"-> winning recalibration: {winner}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    lims = [cc_df.actual_elo.min() - 100, cc_df.predicted_elo.max() + 100]
    ax1.scatter(cc_df.actual_elo, cc_df.predicted_elo, color=plotstyle.SECONDARY, s=40, label="raw")
    ax1.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("Actual Chess.com Elo"); ax1.set_ylabel("Predicted (raw, Lichess-scale)")
    ax1.set_title("Before recalibration"); ax1.set_xlim(lims); ax1.set_ylim(lims)
    recal_col = iso_pred if winner == "isotonic" else lin_pred
    ax2.scatter(cc_df.actual_elo, recal_col, color=plotstyle.TERTIARY, s=40, label=winner)
    ax2.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax2.set_xlabel("Actual Chess.com Elo"); ax2.set_ylabel(f"Predicted ({winner}-recalibrated)")
    ax2.set_title("After recalibration (held-out CV)"); ax2.set_xlim(lims); ax2.set_ylim(lims)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_external_validation_recalibration.png"))
    plt.close(fig)

    raw_all = cc_df["predicted_elo"].to_numpy(dtype=float)
    actual_all = cc_df["actual_elo"].to_numpy(dtype=float)
    if winner == "isotonic":
        final_map = IsotonicRegression(out_of_bounds="clip").fit(raw_all, actual_all)
    else:
        final_map = LinearRegression().fit(raw_all.reshape(-1, 1), actual_all)
    recal_path = RECAL_PATH_TMPL.format(pool=args.pool)
    os.makedirs(os.path.dirname(recal_path), exist_ok=True)
    joblib.dump({"method": winner, "model": final_map, "pool": args.pool,
                "n_players_fit": len(cc_df)}, recal_path)
    print(f"Saved deployable recalibration map -> {recal_path}")

    with open(os.path.join(args.outdir, "external_validation_summary.json"), "w") as f:
        json.dump({"part_a": part_a, "part_b": recal_summary, "winner": winner}, f, indent=2)
    print(f"\nSaved -> {args.outdir}/external_validation_bias.csv, "
          f"fig_external_validation_bias.png, fig_external_validation_recalibration.png, "
          f"external_validation_summary.json")


if __name__ == "__main__":
    main()
