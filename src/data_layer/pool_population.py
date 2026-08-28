"""
Rating distribution of the players who actually play a given time control.

Answers a question the dataset itself cannot: is the narrow rating spread of the
`classical` sample (sigma = 411 vs 619-672 elsewhere, table 3.6) a property of how
we sampled, or of who plays that pool at all? The collector applies two filters
(>= 8 games per player, <= 200 players per rating bin), so its output cannot
separate the two. This scans the raw dump with NO filtering and reports the
distribution twice: over every player seen, and over those with >= 8 games. If the
two shapes match, the filters are not what makes the sample narrow.

Usage:
    python -m src.data_layer.pool_population --pool classical --max-scan 400000
"""

import argparse
import collections
import statistics

import chess.pgn

BIN_WIDTH = 200
RATING_MIN, RATING_MAX = 400, 2800


def scan(pgn_path, pool, max_scan):
    per_player = collections.defaultdict(list)
    scanned = kept = 0
    with open(pgn_path, encoding="utf-8") as f:
        while scanned < max_scan:
            h = chess.pgn.read_headers(f)
            if h is None:
                break
            scanned += 1
            if pool.lower() not in h.get("Event", "").lower():
                continue
            we, be = h.get("WhiteElo", "?"), h.get("BlackElo", "?")
            if not (we.isdigit() and be.isdigit()):
                continue
            kept += 1
            per_player[h.get("White", "?")].append(int(we))
            per_player[h.get("Black", "?")].append(int(be))
    return per_player, scanned, kept


def distribution(medians):
    n_bins = (RATING_MAX - RATING_MIN) // BIN_WIDTH
    c = collections.Counter(
        min(int((r - RATING_MIN) // BIN_WIDTH), n_bins - 1)
        for r in medians if r >= RATING_MIN)
    total = sum(c.values())
    return {b: 100 * c.get(b, 0) / total for b in range(n_bins)}, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="lichess_db_standard_rated_2026-01.pgn")
    ap.add_argument("--pool", default="classical")
    ap.add_argument("--max-scan", type=int, default=400_000)
    ap.add_argument("--min-games", type=int, default=8,
                    help="the collector's per-player threshold, for the second column")
    args = ap.parse_args()

    per_player, scanned, kept = scan(args.pgn, args.pool, args.max_scan)
    all_med = [statistics.median(v) for v in per_player.values()]
    q_med = [statistics.median(v) for v in per_player.values() if len(v) >= args.min_games]
    d_all, n_all = distribution(all_med)
    d_q, n_q = distribution(q_med)

    print(f"scanned {scanned:,} games; {args.pool} games: {kept:,} "
          f"({100 * kept / scanned:.2f}% of the dump)")
    print(f"players seen: {n_all:,}; with >= {args.min_games} games: {n_q:,} "
          f"({100 * n_q / n_all:.1f}%)\n")
    print(f"{'rating bin':>12}{'all players':>14}{f'>= {args.min_games} games':>16}")
    for b in sorted(d_all):
        lo = RATING_MIN + b * BIN_WIDTH
        print(f"{lo:>6}-{lo + BIN_WIDTH:<5}{d_all[b]:>13.1f}%{d_q[b]:>15.1f}%")

    def share(dist, lo, hi):
        return sum(v for b, v in dist.items()
                   if lo <= RATING_MIN + b * BIN_WIDTH < hi)
    print(f"\nall players: below 1000 {share(d_all, 0, 1000):.1f}%, "
          f"1200-2000 {share(d_all, 1200, 2000):.1f}%, "
          f"above 2200 {share(d_all, 2200, 3000):.1f}%")


if __name__ == "__main__":
    main()
