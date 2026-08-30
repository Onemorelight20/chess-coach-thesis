"""
Positive control re-run in the deployed order: does aggregating N games before
attribution recover an injected weakness that per-game detection misses?

The same category is injected into all N games of a pseudo-player, then two
detectors are scored on the identical treated games -- per-game (the published
protocol) vs aggregated (app.py's order).

Usage:
    python -m src.ml.synthetic_aggregate lichess_db_standard_rated_2026-01.pgn \
        --players 15 --per-player 4 --min-elo 2000 --depth 12 --outdir results
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.engine.analyzer import EngineAnalyzer
from src.features.aggregate import aggregate_features
from src.features.extractor import FeatureExtractor, DISPLAY_NAMES
from src.ml.model import ELOModel
from src.ml.synthetic_validation import (
    CATS, build_treatment, detect, stream_games, parse_bands, bootstrap_ci,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--players", type=int, default=15)
    ap.add_argument("--per-player", type=int, default=4)
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
    ap.add_argument("--rule", default="raw", choices=["raw", "standardised"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import src.ml.synthetic_validation as SV
    SV.RULE = getattr(args, "rule", "raw")
    print(f"Ranking rule: {SV.RULE}")
    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)
    bands = parse_bands(args.elo_bands) if args.elo_bands else None

    need = args.players * args.per_player
    print(f"Streaming {need} '{args.event}' games "
          f"({'bands ' + args.elo_bands if bands else 'Elo >= ' + str(args.min_elo)})...")
    games = stream_games(args.pgn, need, args.min_elo, args.event, bands)
    # Group by adjacent rating: a pseudo-player must not mix a 1000 and a 2800 game,
    # or the averaged vector describes nobody.
    games.sort(key=lambda g: g[2])
    groups = [games[i:i + args.per_player]
              for i in range(0, len(games) - len(games) % args.per_player, args.per_player)]
    print(f"  {len(games)} games -> {len(groups)} pseudo-players of {args.per_player}")
    for i, grp in enumerate(groups, 1):
        print(f"    player {i}: Elo {grp[0][2]}-{grp[-1][2]}")
    print()
    if not groups:
        print("Not enough games.")
        return

    analyzer = EngineAnalyzer(depth=args.depth)
    model = ELOModel(pool=args.event)

    per_game = {c: [] for c in CATS}     # one entry per treated game
    aggreg = {c: [] for c in CATS}       # one entry per pseudo-player
    # the same two detectors on UNTOUCHED games: without this prior, a hit rate
    # cannot be told apart from the detector's habit of naming a given feature.
    per_game_base = {c: [] for c in CATS}
    aggreg_base = {c: [] for c in CATS}
    conf = {k: {gt: {p: 0 for p in CATS} for gt in CATS} for k in ("per_game", "aggregated")}

    for gi, group in enumerate(groups, 1):
        prepared = []
        for game, color, elo, _band in group:
            clean = analyzer.analyze_game(game)
            initial = analyzer._capped_white_cp(
                analyzer.engine.analyse(game.board(), analyzer._limit()))
            prepared.append((game, color, clean, initial))

        clean_feats = [FeatureExtractor(color).extract(clean)
                       for _game, color, clean, _initial in prepared]
        base_pg = [detect(model, f) for f in clean_feats]
        base_agg = detect(model, aggregate_features(clean_feats, "mean"))
        for cat in CATS:
            per_game_base[cat].extend(int(b == cat) for b in base_pg)
            aggreg_base[cat].append(int(base_agg == cat))
            feats = []
            for game, color, clean, initial in prepared:
                treat, _ = build_treatment(analyzer, game, color, clean, initial, cat,
                                           args.rate, rng, args.magnitude,
                                           args.loss_lo, args.loss_hi, args.wp_lo, args.wp_hi)
                f = FeatureExtractor(color).extract(treat)
                feats.append(f)
                pred = detect(model, f)
                per_game[cat].append(int(pred == cat))
                conf["per_game"][cat][pred] += 1

            pred_agg = detect(model, aggregate_features(feats, "mean"))
            aggreg[cat].append(int(pred_agg == cat))
            conf["aggregated"][cat][pred_agg] += 1
        print(f"  [{gi}/{len(groups)}] player done", flush=True)

    analyzer.close()

    result = {
        "n_players": len(groups), "per_player": args.per_player,
        "n_games": len(groups) * args.per_player,
        "chance_baseline": 1.0 / len(CATS),
        "injection": {"magnitude": args.magnitude, "rate": args.rate,
                      "depth": args.depth, "min_elo": args.min_elo,
                      "elo_bands": args.elo_bands},
        "per_category": {}, "confusion": conf,
    }
    for c in CATS:
        pg = float(np.mean(per_game[c])) if per_game[c] else 0.0
        ag = float(np.mean(aggreg[c])) if aggreg[c] else 0.0
        pg_b = float(np.mean(per_game_base[c])) if per_game_base[c] else 0.0
        ag_b = float(np.mean(aggreg_base[c])) if aggreg_base[c] else 0.0
        result["per_category"][c] = {
            "per_game": pg, "per_game_clean": pg_b, "per_game_lift": pg - pg_b,
            "aggregated": ag, "aggregated_clean": ag_b, "aggregated_lift": ag - ag_b,
            "aggregated_ci95": bootstrap_ci([[h] for h in aggreg[c]]),
        }
    result["overall_per_game"] = float(np.mean([h for c in CATS for h in per_game[c]]))
    result["overall_aggregated"] = float(np.mean([h for c in CATS for h in aggreg[c]]))
    result["overall_aggregated_ci95"] = bootstrap_ci(list(aggreg.values()))

    out = os.path.join(args.outdir, f"synthetic_aggregate{args.tag}.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n{'category':24}{'per-game':>9}{'(lift)':>9}{'aggreg':>9}{'(lift)':>9}")
    for c in CATS:
        b = result["per_category"][c]
        print(f"{DISPLAY_NAMES.get(c, c)[:23]:24}{b['per_game']:>9.2f}{b['per_game_lift']:>+9.2f}"
              f"{b['aggregated']:>9.2f}{b['aggregated_lift']:>+9.2f}")
    print(f"{'OVERALL':24}{result['overall_per_game']:>10.2f}"
          f"{result['overall_aggregated']:>12.2f}"
          f"{result['overall_aggregated'] - result['overall_per_game']:>+9.2f}")
    lo, hi = result["overall_aggregated_ci95"]
    print(f"\naggregated 95% CI {lo:.2f}-{hi:.2f} vs chance "
          f"{result['chance_baseline']:.2f}  ->  {out}")


if __name__ == "__main__":
    main()
