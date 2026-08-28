"""
Specificity of the synthetic injection: does degrading one category leave the
other six alone?

synthetic_validation.py already reports a manipulation check -- the target
feature does move. It does not check the converse, which the thesis nonetheless
claims: that the remaining categories stay put, so any change in the feature
vector can be attributed to the injected one. Substitutions could plausibly leak:
a badly degraded opening move may cross the blunder threshold, and the positional
signals of the substituted ply are recomputed, which can shift hanging_rate.

For every injected category this script records the delta of ALL injectable
features, expressed in standard deviations of that feature in the training pool
so the columns are comparable. The output is a 7x7 matrix: rows = injected,
columns = moved.

Usage:
    python -m src.ml.synthetic_specificity lichess_db_standard_rated_2026-01.pgn \
        --games 20 --elo-bands "1000-1400,1400-1800,1800-2200,2200-3000" --depth 12
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
from src.features.extractor import FeatureExtractor, DISPLAY_NAMES
from src.ml.synthetic_validation import CATS, build_treatment, stream_games, parse_bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--min-elo", type=int, default=2000)
    ap.add_argument("--elo-bands", default=None)
    ap.add_argument("--event", default="blitz")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--rate", type=float, default=0.7)
    ap.add_argument("--magnitude", choices=["winprob", "cp"], default="winprob")
    ap.add_argument("--wp-lo", type=float, default=0.10)
    ap.add_argument("--wp-hi", type=float, default=0.30)
    ap.add_argument("--loss-lo", type=int, default=120)
    ap.add_argument("--loss-hi", type=int, default=400)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)
    bands = parse_bands(args.elo_bands) if args.elo_bands else None

    # population spread of each injectable feature: the yardstick for "moved a lot"
    pool = pd.read_parquet(os.path.join("data", f"dataset_{args.event}_2k.parquet"))
    sd = pool[CATS].std(ddof=0).replace(0, 1.0)

    games = stream_games(args.pgn, args.games, args.min_elo, args.event, bands)
    print(f"games: {len(games)}\n")
    analyzer = EngineAnalyzer(depth=args.depth)

    # deltas[injected][moved] -> list of per-game deltas in SD units
    deltas = {inj: {mv: [] for mv in CATS} for inj in CATS}

    for gi, (game, color, elo, _band) in enumerate(games, 1):
        clean = analyzer.analyze_game(game)
        initial = analyzer._capped_white_cp(
            analyzer.engine.analyse(game.board(), analyzer._limit()))
        base = FeatureExtractor(color).extract(clean)

        for inj in CATS:
            treat, _ = build_treatment(analyzer, game, color, clean, initial, inj,
                                       args.rate, rng, args.magnitude,
                                       args.loss_lo, args.loss_hi, args.wp_lo, args.wp_hi)
            feats = FeatureExtractor(color).extract(treat)
            for mv in CATS:
                a, b = feats.get(mv, np.nan), base.get(mv, np.nan)
                if not (np.isnan(a) or np.isnan(b)):
                    deltas[inj][mv].append((a - b) / sd[mv])
        print(f"  [{gi}/{len(games)}] Elo {elo} done", flush=True)

    analyzer.close()

    matrix = {inj: {mv: float(np.mean(v)) if v else 0.0 for mv, v in row.items()}
              for inj, row in deltas.items()}
    summary = {"n_games": len(games), "unit": "standard deviations of the pool",
               "matrix": matrix, "leakage": {}}
    for inj in CATS:
        own = abs(matrix[inj][inj])
        others = {mv: abs(matrix[inj][mv]) for mv in CATS if mv != inj}
        worst = max(others, key=others.get)
        summary["leakage"][inj] = {
            "own_shift_sd": matrix[inj][inj],
            "largest_other": worst,
            "largest_other_shift_sd": matrix[inj][worst],
            "ratio_own_to_other": float(own / others[worst]) if others[worst] > 1e-9 else float("inf"),
        }

    out = os.path.join(args.outdir, "synthetic_specificity.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    head = "".join(f"{DISPLAY_NAMES.get(c, c)[:10]:>12}" for c in CATS)
    print(f"\nЗсув ознак у стандартних відхиленнях (рядок = погіршено)\n{'':22}{head}")
    for inj in CATS:
        row = "".join(f"{matrix[inj][mv]:>12.2f}" for mv in CATS)
        print(f"{DISPLAY_NAMES.get(inj, inj)[:21]:22}{row}")
    print(f"\n{'погіршено':22}{'власний':>10}{'найбільший чужий':>20}{'відношення':>13}")
    for inj, blk in summary["leakage"].items():
        print(f"{DISPLAY_NAMES.get(inj, inj)[:21]:22}{blk['own_shift_sd']:>10.2f}"
              f"{blk['largest_other_shift_sd']:>13.2f} ({blk['largest_other'][:12]:<12})"
              f"{blk['ratio_own_to_other']:>8.1f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
