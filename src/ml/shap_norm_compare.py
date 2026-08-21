"""
Raw vs population-normalised SHAP attribution for the weakness mechanism.

argmin over raw SHAP values is dominated by a few high-amplitude features, so
low-amplitude categories can never win it. This re-runs synthetic_validation.py's
injection protocol and scores two detectors on identical games:

    raw   : argmin_j  phi_j                 (current method)
    norm  : argmin_j (phi_j - mu_j) / s_j   (population-standardised)

Usage:
    python -m src.ml.shap_norm_compare --pgn sample_test.pgn --games 60 \
        --min-elo 2000 --event blitz --depth 12 --outdir results
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.engine.analyzer import EngineAnalyzer
from src.features.extractor import FeatureExtractor
from src.ml.model import ELOModel
from src.ml.synthetic_validation import CATS, build_treatment, stream_games, bootstrap_ci


def shap_baseline(pool: str, outdir: str) -> pd.DataFrame:
    """Population mean/std of each feature's SHAP contribution (cached)."""
    path = os.path.join(outdir, f"shap_baseline_{pool}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)

    import joblib
    import shap
    import xgboost as xgb

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    feats = joblib.load(os.path.join(base, f"xgboost_elo_{pool}_features.pkl"))
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(base, f"xgboost_elo_{pool}.json"))

    df = pd.read_parquet(os.path.join("data", f"dataset_{pool}_2k.parquet"))
    values = shap.TreeExplainer(model).shap_values(df[feats])
    frame = pd.DataFrame(values, columns=feats)
    stats = pd.DataFrame({"mu": frame.mean(), "sigma": frame.std()})
    stats.to_csv(path)
    return stats


def detect_raw(shap_dict) -> str:
    cands = [c for c in CATS if c in shap_dict]
    return min(cands, key=lambda c: shap_dict[c])


def detect_norm(shap_dict, stats, alpha: float = 1.0) -> str:
    """alpha=0 -> raw argmin; alpha=1 -> full z-score; in between -> partial."""
    cands = [c for c in CATS if c in shap_dict]
    def z(c):
        sigma = float(stats.loc[c, "sigma"]) or 1.0
        return (shap_dict[c] - alpha * float(stats.loc[c, "mu"])) / (sigma ** alpha)
    return min(cands, key=z)


def summarise(hits, label):
    out = {"detector": label, "per_category": {}}
    for c in CATS:
        out["per_category"][c] = {
            "sensitivity_top1": float(np.mean(hits[c])) if hits[c] else 0.0,
            "ci95": bootstrap_ci([[h] for h in hits[c]]),
        }
    flat = [h for c in CATS for h in hits[c]]
    out["overall_top1"] = float(np.mean(flat)) if flat else 0.0
    out["overall_ci95"] = bootstrap_ci(list(hits.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="sample_test.pgn")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--min-elo", type=int, default=2000)
    ap.add_argument("--event", default="blitz")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--rate", type=float, default=0.7)
    ap.add_argument("--magnitude", default="winprob")
    ap.add_argument("--loss-lo", type=int, default=120)
    ap.add_argument("--loss-hi", type=int, default=400)
    ap.add_argument("--wp-lo", type=float, default=0.10)
    ap.add_argument("--wp-hi", type=float, default=0.30)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)

    stats = shap_baseline(args.event, args.outdir)
    print(f"SHAP baseline: {len(stats)} features "
          f"(sigma {stats['sigma'].min():.1f}..{stats['sigma'].max():.1f})")

    games = stream_games(args.pgn, args.games, args.min_elo, args.event)
    print(f"games: {len(games)}")

    analyzer = EngineAnalyzer(depth=args.depth)
    model = ELOModel(pool=args.event)

    hits = {"raw": {c: [] for c in CATS}, "norm": {c: [] for c in CATS}}
    confusion = {k: {gt: {p: 0 for p in CATS} for gt in CATS} for k in hits}

    for gi, (game, color, _elo, _band) in enumerate(games, 1):
        clean = analyzer.analyze_game(game)
        initial_eval = analyzer._capped_white_cp(
            analyzer.engine.analyse(game.board(), analyzer._limit()))

        for cat in CATS:
            treat, _ = build_treatment(analyzer, game, color, clean, initial_eval,
                                       cat, args.rate, rng, args.magnitude,
                                       args.loss_lo, args.loss_hi, args.wp_lo, args.wp_hi)
            feats = FeatureExtractor(color).extract(treat)
            sv = model.predict_and_explain(feats)["shap_values"]

            for key, pred in (("raw", detect_raw(sv)), ("norm", detect_norm(sv, stats, args.alpha))):
                hits[key][cat].append(int(pred == cat))
                confusion[key][cat][pred] += 1
        print(f"  [{gi}/{len(games)}] {color} done", flush=True)

    analyzer.close()

    result = {
        "n_games": len(games),
        "chance_baseline": 1.0 / len(CATS),
        "alpha": args.alpha,
        "raw": summarise(hits["raw"], "argmin raw SHAP"),
        "norm": summarise(hits["norm"], f"argmin standardised SHAP (alpha={args.alpha})"),
        "confusion": confusion,
    }
    suffix = "" if args.alpha == 1.0 else f"_a{args.alpha:g}"
    out = os.path.join(args.outdir, f"shap_norm_comparison{suffix}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'category':<20}{'raw':>8}{'norm':>8}{'delta':>8}")
    for c in CATS:
        r = result["raw"]["per_category"][c]["sensitivity_top1"]
        n = result["norm"]["per_category"][c]["sensitivity_top1"]
        print(f"{c:<20}{r:>8.2f}{n:>8.2f}{n - r:>+8.2f}")
    print(f"{'OVERALL':<20}{result['raw']['overall_top1']:>8.2f}"
          f"{result['norm']['overall_top1']:>8.2f}"
          f"{result['norm']['overall_top1'] - result['raw']['overall_top1']:>+8.2f}")
    print(f"chance = {result['chance_baseline']:.2f}   ->  {out}")


if __name__ == "__main__":
    main()
