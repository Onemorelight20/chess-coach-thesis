"""
Elo-prediction accuracy vs. number of games aggregated per player.

Two configurations, which answer different questions:

  default    -- for each N, mean-aggregate N random games per player into one
                vector and TRAIN on those aggregated rows (player-level KFold).
                This is the ceiling a model built for N-game vectors reaches.
  --deployed -- the PREVIOUS configuration of app.py: a model trained on
                SINGLE-game rows (GroupKFold(5) by player) fed with the mean
                vector of N test-player games. Also reports the alternative of
                averaging the N per-game predictions of the same model.
  --mixed    -- the configuration app.py runs now: the model is trained on the
                MIXED set (single games + N-game series of the training players,
                mixed_training.py) and fed with the mean vector of N test-player
                games; the same vectors are also scored by the single-game model
                for a like-for-like comparison. Writes aggregation_mixed.json and
                redraws fig_aggregation_curve.png with all three configurations.

Usage:
    python -m src.ml.aggregation_curve data/dataset_blitz_2k.parquet --outdir results
    python -m src.ml.aggregation_curve data/dataset_blitz_2k.parquet --deployed --outdir results
    python -m src.ml.aggregation_curve --pools data/dataset_bullet_2k.parquet,data/dataset_blitz_2k.parquet,data/dataset_rapid_2k.parquet,data/dataset_classical_2k.parquet --outdir results
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.extractor import FEATURE_NAMES
from src.features.aggregate import aggregate_features
from src.ml.train import PARAMS
from src.ml.mixed_training import mixed_training_set
from src.ml import plotstyle
plotstyle.apply()


def eval_at_N(df, N, repeats=3, base_seed=0):
    r2s, maes, npl = [], [], []
    for r in range(repeats):
        rng = np.random.default_rng(base_seed + r)
        rows, ys = [], []
        for _player, g in df.groupby("player"):
            if len(g) < N:
                continue
            idx = rng.choice(len(g), N, replace=False)
            sub = g.iloc[idx]
            dicts = sub[FEATURE_NAMES].to_dict("records")
            rows.append(aggregate_features(dicts, "mean"))
            ys.append(float(g["target_elo"].median()))
        X, y = pd.DataFrame(rows), np.array(ys)
        res = cross_validate(XGBRegressor(**PARAMS, n_jobs=1), X, y,
                             cv=KFold(5, shuffle=True, random_state=base_seed + r),
                             scoring=["r2", "neg_mean_absolute_error"], n_jobs=-1)
        r2s.append(res["test_r2"].mean())
        maes.append(-res["test_neg_mean_absolute_error"].mean())
        npl.append(len(y))
    return (float(np.mean(r2s)), float(np.std(r2s)),
            float(np.mean(maes)), int(np.mean(npl)))


def eval_deployed(df, levels, repeats=3, base_seed=0):
    """
    Accuracy of the deployed configuration: a model trained on single-game rows
    of the training players, applied to (a) the mean feature vector of N games
    of each test player and (b) the mean of its N per-game predictions.
    Player-grouped 5-fold CV; y = the player's median rating over all games.
    """
    X, y, groups = df[FEATURE_NAMES], df["target_elo"], df["player"]
    per_player = df.groupby("player")["target_elo"].median()
    acc = {N: {"feat": [], "pred": []} for N in levels}
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        model = XGBRegressor(**PARAMS, n_jobs=-1).fit(X.iloc[tr], y.iloc[tr])
        test = df.iloc[te]
        game_pred = pd.Series(model.predict(test[FEATURE_NAMES]), index=test.index)
        for r in range(repeats):
            rng = np.random.default_rng(base_seed + r)
            # per N: sample once per player per repeat
            for N in levels:
                fv, pv, yy = [], [], []
                for player, g in test.groupby("player"):
                    if len(g) < N:
                        continue
                    idx = rng.choice(len(g), N, replace=False)
                    sub = g.iloc[idx]
                    fv.append(sub[FEATURE_NAMES].mean(numeric_only=True).values)
                    pv.append(game_pred.loc[sub.index].mean())
                    yy.append(per_player.loc[player])
                fv = pd.DataFrame(fv, columns=FEATURE_NAMES)
                acc[N]["feat"].append((np.array(yy), model.predict(fv)))
                acc[N]["pred"].append((np.array(yy), np.array(pv)))
    out = []
    for N in levels:
        def agg_metric(pairs, fn):
            return float(fn(np.concatenate([a for a, _ in pairs]),
                            np.concatenate([b for _, b in pairs])))
        # metrics over the pooled out-of-fold predictions, one value per repeat
        per_rep_feat = [[p for i, p in enumerate(acc[N]["feat"]) if i % repeats == r]
                        for r in range(repeats)]
        per_rep_pred = [[p for i, p in enumerate(acc[N]["pred"]) if i % repeats == r]
                        for r in range(repeats)]
        r2f = [agg_metric(p, r2_score) for p in per_rep_feat]
        maef = [agg_metric(p, mean_absolute_error) for p in per_rep_feat]
        r2p = [agg_metric(p, r2_score) for p in per_rep_pred]
        maep = [agg_metric(p, mean_absolute_error) for p in per_rep_pred]
        n_players = int(sum(len(a) for a, _ in per_rep_feat[0]))
        out.append({"n_games": N, "n_players": n_players,
                    "r2_deployed": float(np.mean(r2f)), "r2_deployed_std": float(np.std(r2f)),
                    "mae_deployed": float(np.mean(maef)),
                    "r2_pred_avg": float(np.mean(r2p)), "r2_pred_avg_std": float(np.std(r2p)),
                    "mae_pred_avg": float(np.mean(maep))})
    return out


def eval_mixed(df, levels, repeats=3, base_seed=0):
    """
    Deployed configuration after the switch to mixed training: per fold the model
    is trained on single games + N-game series of the TRAINING players only and
    applied to the mean vector of N games of each test player. The same test
    vectors are scored by a model trained on single games only (the previous
    configuration). Player-grouped 5-fold CV; y = the player's median rating.
    """
    X, y, groups = df[FEATURE_NAMES], df["target_elo"], df["player"]
    per_player = df.groupby("player")["target_elo"].median()
    acc = {N: {"mixed": [], "single": []} for N in levels}
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        Xm, ym, _, _ = mixed_training_set(df.iloc[tr])
        mixed = XGBRegressor(**PARAMS, n_jobs=-1).fit(Xm, ym)
        single = XGBRegressor(**PARAMS, n_jobs=-1).fit(X.iloc[tr], y.iloc[tr])
        test = df.iloc[te]
        for r in range(repeats):
            rng = np.random.default_rng(base_seed + r)
            for N in levels:
                fv, yy = [], []
                for player, g in test.groupby("player"):
                    if len(g) < N:
                        continue
                    idx = rng.choice(len(g), N, replace=False)
                    fv.append(g.iloc[idx][FEATURE_NAMES].mean(numeric_only=True).values)
                    yy.append(per_player.loc[player])
                fv = pd.DataFrame(fv, columns=FEATURE_NAMES)
                acc[N]["mixed"].append((np.array(yy), mixed.predict(fv)))
                acc[N]["single"].append((np.array(yy), single.predict(fv)))
    out = []
    for N in levels:
        def per_rep(key, r):
            return [p for i, p in enumerate(acc[N][key]) if i % repeats == r]
        def metric(pairs, fn):
            return float(fn(np.concatenate([a for a, _ in pairs]), np.concatenate([b for _, b in pairs])))
        row = {"n_games": N, "n_players": int(sum(len(a) for a, _ in per_rep("mixed", 0)))}
        for key in ("mixed", "single"):
            r2 = [metric(per_rep(key, r), r2_score) for r in range(repeats)]
            mae = [metric(per_rep(key, r), mean_absolute_error) for r in range(repeats)]
            row.update({f"r2_{key}": float(np.mean(r2)), f"r2_{key}_std": float(np.std(r2)),
                        f"mae_{key}": float(np.mean(mae)), f"mae_{key}_std": float(np.std(mae))})
        out.append(row)
    return out


def draw_three_configurations(outdir, mixed_curve):
    """R² and MAE against N for the fixed-N ceiling (aggregation_curve.json, if
    present), the previous single-game configuration and the mixed model."""
    ns = [c["n_games"] for c in mixed_curve]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ceiling_path = os.path.join(outdir, "aggregation_curve.json")
    if os.path.exists(ceiling_path):
        ceil = {c["n_games"]: c for c in json.load(open(ceiling_path))}
        cn = [n for n in ns if n in ceil]
        ax1.plot(cn, [ceil[n]["r2"] for n in cn], "^:", color="grey", label="fixed-N training (one model per N)")
        ax2.plot(cn, [ceil[n]["mae"] for n in cn], "^:", color="grey", label="fixed-N training (one model per N)")
    ax1.errorbar(ns, [c["r2_single"] for c in mixed_curve], yerr=[c["r2_single_std"] for c in mixed_curve],
                 marker="s", capsize=3, color="#d62728", ls="--", label="single-game training, mean vector at input (previous)")
    ax1.errorbar(ns, [c["r2_mixed"] for c in mixed_curve], yerr=[c["r2_mixed_std"] for c in mixed_curve],
                 marker="o", capsize=3, color="#1f77b4", label="mixed training (deployed)")
    ax2.plot(ns, [c["mae_single"] for c in mixed_curve], "s--", color="#d62728", label="single-game training (previous)")
    ax2.plot(ns, [c["mae_mixed"] for c in mixed_curve], "o-", color="#1f77b4", label="mixed training (deployed)")
    ax1.set_xlabel("games aggregated per player (N)"); ax1.set_ylabel("CV R²"); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("games aggregated per player (N)"); ax2.set_ylabel("CV MAE (Elo)"); ax2.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_aggregation_curve.png"))
    plt.close(fig)


def by_pool_mixed(paths, outdir, levels_str):
    """The mixed-model curve per pool (the deployed configuration), one JSON + figure."""
    requested = [int(x) for x in levels_str.split(",")]
    curves = {}
    for path in paths:
        df = pd.read_parquet(path)
        df = df[df["target_elo"].between(400, 3000)]
        pool_name = df["pool"].mode().iloc[0] if "pool" in df.columns else os.path.basename(path)
        per_player = df.groupby("player").size()
        levels = [n for n in requested if (per_player >= n).sum() >= 30]
        curve = eval_mixed(df, levels)
        curves[pool_name] = curve
        print(f"{pool_name:>10}: N={levels[0]} R2={curve[0]['r2_mixed']:.3f} -> "
              f"N={levels[-1]} R2={curve[-1]['r2_mixed']:.3f} (MAE {curve[-1]['mae_mixed']:.1f})")
    with open(os.path.join(outdir, "aggregation_mixed_by_pool.json"), "w") as f:
        json.dump(curves, f, indent=2)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for pool_name, curve in curves.items():
        ax.plot([c["n_games"] for c in curve], [c["r2_mixed"] for c in curve], "o-", label=pool_name)
    ax.set_xlabel("games aggregated per player (N)"); ax.set_ylabel("CV R²")
    ax.set_title("Aggregation curve by time-control pool (mixed training)"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_aggregation_by_pool.png"))
    plt.close(fig)
    print(f"Saved -> {outdir}/aggregation_mixed_by_pool.json, fig_aggregation_by_pool.png")


def by_pool(paths, outdir, levels_str):
    """
    The same mean-aggregation curve, repeated per pool. Mirrors evaluate.py's
    --pools convention but consolidates output into a single JSON.
    """
    requested = [int(x) for x in levels_str.split(",")]
    curves = {}
    for path in paths:
        df = pd.read_parquet(path)
        df = df[df["target_elo"].between(400, 3000)]
        pool_name = df["pool"].mode().iloc[0] if "pool" in df.columns else os.path.basename(path)
        per_player = df.groupby("player").size()
        levels = [n for n in requested if (per_player >= n).sum() >= 30]
        curve = []
        for N in levels:
            r2, r2sd, mae, npl = eval_at_N(df, N)
            curve.append({"n_games": N, "n_players": npl, "r2": r2, "r2_std": r2sd, "mae": mae})
        curves[pool_name] = curve
        print(f"{pool_name:>10}: N={levels[0]} R2={curve[0]['r2']:.3f} -> "
              f"N={levels[-1]} R2={curve[-1]['r2']:.3f}")

    with open(os.path.join(outdir, "aggregation_curve_by_pool.json"), "w") as f:
        json.dump(curves, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for pool_name, curve in curves.items():
        ax.plot([c["n_games"] for c in curve], [c["r2"] for c in curve], "o-", label=pool_name)
    ax.set_xlabel("games aggregated per player (N)"); ax.set_ylabel("CV R²")
    ax.set_title("Aggregation curve by time-control pool"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_aggregation_by_pool.png"))
    plt.close(fig)
    print(f"Saved -> {outdir}/aggregation_curve_by_pool.json, fig_aggregation_by_pool.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="?", default=None)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--levels", default="1,2,3,5,8,12")
    ap.add_argument("--deployed", action="store_true",
                    help="evaluate the deployed configuration (single-game model, mean "
                         "vector at inference) instead of training on aggregated rows")
    ap.add_argument("--mixed", action="store_true",
                    help="evaluate the deployed mixed-training configuration (alone, or per pool with --pools)")
    ap.add_argument("--pools", default=None,
                    help="comma-separated single-pool parquet paths "
                         "(e.g. data/dataset_bullet_2k.parquet,data/dataset_blitz_2k.parquet,...) "
                         "-> per-pool aggregation-curve comparison only, skips the single-dataset run below")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.pools:
        if args.mixed:
            by_pool_mixed([p.strip() for p in args.pools.split(",")], args.outdir, args.levels)
        else:
            by_pool([p.strip() for p in args.pools.split(",")], args.outdir, args.levels)
        return
    if not args.parquet:
        ap.error("either a parquet path or --pools is required")

    df = pd.read_parquet(args.parquet)
    df = df[df["target_elo"].between(400, 3000)]
    print(f"Loaded {len(df):,} rows | {df['player'].nunique()} players (mean aggregation)\n")

    if args.mixed:
        per_player = df.groupby("player").size()
        levels = [int(x) for x in args.levels.split(",") if (per_player >= int(x)).sum() >= 30]
        curve = eval_mixed(df, levels)
        print(f"{'N':>4}{'players':>9}{'R2 mixed':>11}{'MAE':>8}{'R2 single':>12}{'MAE':>8}")
        for c in curve:
            print(f"{c['n_games']:>4}{c['n_players']:>9}{c['r2_mixed']:>11.3f}{c['mae_mixed']:>8.1f}"
                  f"{c['r2_single']:>12.3f}{c['mae_single']:>8.1f}")
        with open(os.path.join(args.outdir, "aggregation_mixed.json"), "w") as f:
            json.dump(curve, f, indent=2)
        draw_three_configurations(args.outdir, curve)
        print(f"\nSaved -> {args.outdir}/aggregation_mixed.json, fig_aggregation_curve.png")
        return

    if args.deployed:
        per_player = df.groupby("player").size()
        levels = [int(x) for x in args.levels.split(",") if (per_player >= int(x)).sum() >= 30]
        curve = eval_deployed(df, levels)
        print(f"{'N':>4}{'players':>9}{'R2 feat-mean':>14}{'MAE':>8}{'R2 pred-mean':>14}{'MAE':>8}")
        for c in curve:
            print(f"{c['n_games']:>4}{c['n_players']:>9}{c['r2_deployed']:>14.3f}"
                  f"{c['mae_deployed']:>8.1f}{c['r2_pred_avg']:>14.3f}{c['mae_pred_avg']:>8.1f}")
        with open(os.path.join(args.outdir, "aggregation_deployed.json"), "w") as f:
            json.dump(curve, f, indent=2)
        print(f"\nSaved -> {args.outdir}/aggregation_deployed.json")
        return

    per_player = df.groupby("player").size()
    requested = [int(x) for x in args.levels.split(",")]
    levels = [n for n in requested if (per_player >= n).sum() >= 30]
    skipped = [n for n in requested if n not in levels]
    if skipped:
        print(f"Skipping N={skipped}: fewer than 30 players have that many games "
              f"(this dataset was collected with --games-per-player={int(per_player.max())}).\n")
    curve = []
    print(f"{'N games':>8}{'players':>9}{'R2':>9}{'MAE':>8}")
    for N in levels:
        r2, r2sd, mae, npl = eval_at_N(df, N)
        curve.append({"n_games": N, "n_players": npl, "r2": r2, "r2_std": r2sd, "mae": mae})
        print(f"{N:>8}{npl:>9}{r2:>9.3f}{mae:>8.0f}")

    with open(os.path.join(args.outdir, "aggregation_curve.json"), "w") as f:
        json.dump(curve, f, indent=2)

    ns = [c["n_games"] for c in curve]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.errorbar(ns, [c["r2"] for c in curve], yerr=[c["r2_std"] for c in curve],
                 marker="o", capsize=3, color="#1f77b4")
    ax1.set_xlabel("games aggregated per player (N)"); ax1.set_ylabel("CV R²")
    ax1.grid(True, alpha=0.3)
    ax2.plot(ns, [c["mae"] for c in curve], marker="s", color="#d62728")
    ax2.set_xlabel("games aggregated per player (N)"); ax2.set_ylabel("CV MAE (Elo)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_aggregation_curve.png"))
    plt.close(fig)
    print(f"\nN=1 R²={curve[0]['r2']:.3f} -> N={ns[-1]} R²={curve[-1]['r2']:.3f}. "
          f"Saved -> {args.outdir}/fig_aggregation_curve.png")


if __name__ == "__main__":
    main()
