"""
Justifies the +-1000cp evaluation clip: how often a raw Stockfish evaluation
exceeds +-2000cp on ordinary games, and how large a forced mate scores. eval_cap
is set far above the production value so the measured distribution is not already
clipped by the setting this exists to justify.

Usage:
    python -m src.ml.eval_clip_stats lichess_db_standard_rated_2026-01.pgn \
        --games 120 --event blitz --outdir results
"""

import argparse
import json
import os
import sys

import chess.pgn
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.engine.analyzer import EngineAnalyzer

MEASURE_CAP = 100_000   # effectively "no clip": isolates the raw distribution
CLIP_THRESHOLD = 2000   # the magnitude the thesis reports "% of positions beyond"


def stream_games(path, n, event):
    games = []
    with open(path, encoding="utf-8") as f:
        while len(games) < n:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            if event and event.lower() not in g.headers.get("Event", "").lower():
                continue
            games.append(g)
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--event", default="blitz")
    ap.add_argument("--time-limit", type=float, default=0.05,
                    help="matches the production collector's per-move time budget")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Streaming {args.games} sequential '{args.event}' games from '{args.pgn}'...")
    games = stream_games(args.pgn, args.games, args.event)
    print(f"  collected {len(games)} games.")

    analyzer = EngineAnalyzer(time_limit=args.time_limit, multipv=1, eval_cap=MEASURE_CAP)
    evals = []
    for gi, game in enumerate(games, 1):
        for ply in analyzer.analyze_game(game):
            evals.append(ply["eval_cp"])
        if gi % 20 == 0:
            print(f"  analyzed {gi}/{len(games)} games...")
    analyzer.close()

    evals = np.abs(np.array(evals, dtype=float))
    result = {
        "pool": args.event,
        "games_analyzed": len(games),
        "n_positions": int(len(evals)),
        "pct_over_2000cp": float((evals > CLIP_THRESHOLD).mean() * 100),
        "max_abs_cp": int(evals.max()),
        "analyzer_settings": {
            "time_limit": args.time_limit,
            "multipv": 1,
            "eval_cap_used_for_measurement": MEASURE_CAP,
        },
    }
    with open(os.path.join(args.outdir, "eval_clip_justification.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{result['pct_over_2000cp']:.1f}% of {result['n_positions']} positions exceed "
          f"{CLIP_THRESHOLD}cp; max |eval| = {result['max_abs_cp']} (forced-mate sentinel).")
    print(f"Saved -> {args.outdir}/eval_clip_justification.json")


if __name__ == "__main__":
    main()
