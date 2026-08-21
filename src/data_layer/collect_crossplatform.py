"""
Cross-platform (Chess.com) sample collector for external validation.

Chess.com publishes no bulk dump, so this snowball-crawls the public API: a BFS
over opponents that keeps a player once they land in an under-filled 200-Elo bin.
Two resumable passes -- `discover` (network -> JSON cache) and `analyze` (engine
-> parquet with the dataset_<pool>_2k schema).

Usage:
    python -m src.data_layer.collect_crossplatform discover \\
        --seeds bohdanboretskyi,DaisyDrew --pool blitz --per-bin 6 \\
        --out data/crossplatform/chesscom_blitz_discovery.json

    python -m src.data_layer.collect_crossplatform analyze \\
        data/crossplatform/chesscom_blitz_discovery.json \\
        --out data/crossplatform/dataset_chesscom_blitz.parquet
"""

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict, deque

import chess.pgn
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data_layer.fetch_games import fetch_user_games, FetchError
from src.data_layer.ingestion import is_valid_game
from src.engine.analyzer import EngineAnalyzer
from src.features.extractor import FeatureExtractor, FEATURE_NAMES

MIN_VALID_GAMES = 5     # a player needs at least this many analyzable games to be kept
REQUEST_DELAY = 0.35    # politeness delay between Chess.com API calls
RATE_LIMIT_BACKOFF = 65  # seconds to wait after a 429 before retrying once
FLUSH_EVERY = 50        # engine-pass checkpoint interval (games), like collector.py


def _fetch_with_retry(username, pool, max_games):
    """fetch_user_games, retrying once after a Chess.com rate limit (429)."""
    try:
        return fetch_user_games("chess.com", username, pool, max_games)
    except FetchError as e:
        if "rate limit" in str(e).lower():
            print(f"    rate limited on {username!r}; waiting {RATE_LIMIT_BACKOFF}s...")
            time.sleep(RATE_LIMIT_BACKOFF)
            return fetch_user_games("chess.com", username, pool, max_games)
        raise


def _bin_index(rating, rating_min, bin_width):
    return int((rating - rating_min) // bin_width)


def discover(seeds, pool, per_bin, rating_min, rating_max, bin_width,
            games_per_player, max_players, out_path):
    n_bins = (rating_max - rating_min) // bin_width
    cached = {"players": {}, "visited": [], "queue": []}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            cached = json.load(f)
        cached.setdefault("visited", [])
        cached.setdefault("queue", [])
        print(f"Resuming: {len(cached['players'])} players cached, "
              f"{len(cached['queue'])} still queued, {len(cached['visited'])} ever visited.")

    bins = defaultdict(list)
    for uname, rec in cached["players"].items():
        b = _bin_index(rec["rating"], rating_min, bin_width)
        if 0 <= b < n_bins:
            bins[b].append(uname)

    def bins_full():
        return all(len(bins[b]) >= per_bin for b in range(n_bins))

    # Persisting the frontier is what lets a second `discover` run continue rather
    # than find its seeds already visited and have nowhere to go.
    visited = set(cached["visited"])
    queue = deque(cached["queue"])
    # New seeds go to the FRONT: chosen deliberately, so they must not sit behind
    # an old backlog past --max-players.
    for s in reversed(seeds):
        if s.lower() not in visited:
            visited.add(s.lower())
            queue.appendleft(s)

    def save():
        cached["visited"] = list(visited)
        cached["queue"] = list(queue)
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cached, f)
        os.replace(tmp, out_path)

    n_visited_this_run = 0
    while queue and not bins_full() and n_visited_this_run < max_players:
        uname = queue.popleft()
        n_visited_this_run += 1
        try:
            time.sleep(REQUEST_DELAY)
            page = _fetch_with_retry(uname, pool, games_per_player)
        except FetchError as e:
            print(f"  [{n_visited_this_run}] {uname:20} -> skip ({e})")
            save()
            continue

        games = [g for g in page.games if not g.get("note")]
        for g in games:                                # keep exploring regardless
            opp = g.get("opponent")
            if opp and opp.lower() not in visited:
                visited.add(opp.lower())
                queue.append(opp)

        if len(games) < MIN_VALID_GAMES:
            print(f"  [{n_visited_this_run}] {uname:20} -> only {len(games)} valid games, skip")
            save()
            continue
        ratings = [g["player_elo"] for g in games if g.get("player_elo")]
        if not ratings:
            save()
            continue
        rating = float(np.median(ratings))
        b = _bin_index(rating, rating_min, bin_width)
        if not (0 <= b < n_bins) or len(bins[b]) >= per_bin:
            print(f"  [{n_visited_this_run}] {uname:20} rating~{rating:.0f} -> bin full/out of range, skip")
            save()
            continue

        bins[b].append(uname)
        cached["players"][uname] = {"rating": rating, "pool": pool, "games": games}
        print(f"  [{n_visited_this_run}] {uname:20} rating~{rating:.0f} -> KEPT "
              f"(bin {rating_min + b * bin_width}-{rating_min + (b + 1) * bin_width}, "
              f"{len(bins[b])}/{per_bin})")
        save()

    print(f"\nVisited {n_visited_this_run} candidates this run. "
          f"Cached players: {len(cached['players'])}.")
    print("Players per bin:")
    for b in range(n_bins):
        lo = rating_min + b * bin_width
        print(f"  {lo:>4}-{lo + bin_width:<4} {len(bins[b])}/{per_bin}")


def analyze(discovery_path, out_path, time_limit=0.05):
    with open(discovery_path, encoding="utf-8") as f:
        cached = json.load(f)
    players = cached["players"]
    print(f"Loaded {len(players)} discovered players from {discovery_path}.")

    partial_path = out_path + ".partial"
    rows = []
    done_keys = set()
    if os.path.exists(partial_path):
        prev = pd.read_parquet(partial_path)
        rows = prev.to_dict("records")
        done_keys = set(prev["player"] + "::" + prev["game_url"])
        print(f"Resuming: {len(rows)} rows already checkpointed.")

    analyzer = EngineAnalyzer(time_limit=time_limit)  # exact live-app settings
    n_since_flush = 0

    def flush():
        tmp = partial_path + ".tmp"
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        os.replace(tmp, partial_path)

    for player, rec in players.items():
        for g in rec["games"]:
            key = f"{player}::{g.get('url', '')}"
            if key in done_keys:
                continue
            parsed = chess.pgn.read_game(io.StringIO(g["pgn"]))
            if parsed is None or not is_valid_game(parsed):
                continue
            results = analyzer.analyze_game(parsed)
            inc = 0.0
            tc = g.get("time_control", "")
            if "+" in tc:
                try:
                    inc = float(tc.split("+")[1])
                except ValueError:
                    inc = 0.0
            feats = FeatureExtractor(target_player_color=g["color"]).extract(results, increment=inc)
            row = dict(feats)
            row.update(player=player, game_url=g.get("url", ""), target_color=g["color"],
                      target_elo=g.get("player_elo"), time_control=tc,
                      pool=rec["pool"], platform="chess.com", result=g.get("result", "?"))
            rows.append(row)
            n_since_flush += 1
            if n_since_flush % FLUSH_EVERY == 0:
                flush()
                print(f"  ...{len(rows)} rows analyzed, checkpointed.")

    analyzer.close()
    flush()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=[f for f in FEATURE_NAMES if "think_time" not in f])
    df.to_parquet(out_path, index=False)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"\nWrote {len(df):,} rows ({df['player'].nunique()} players) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover")
    d.add_argument("--seeds", required=True, help="comma-separated seed usernames")
    d.add_argument("--pool", default="blitz", choices=["bullet", "blitz", "rapid"])
    d.add_argument("--per-bin", type=int, default=6)
    d.add_argument("--rating-min", type=int, default=400)
    d.add_argument("--rating-max", type=int, default=2800)
    d.add_argument("--bin-width", type=int, default=200)
    d.add_argument("--games-per-player", type=int, default=12)
    d.add_argument("--max-players", type=int, default=400, help="safety cap on crawl size")
    d.add_argument("--out", default="data/crossplatform/chesscom_blitz_discovery.json")

    a = sub.add_parser("analyze")
    a.add_argument("discovery_json")
    a.add_argument("--out", default="data/crossplatform/dataset_chesscom_blitz.parquet")
    a.add_argument("--time-limit", type=float, default=0.05)

    args = ap.parse_args()
    if args.cmd == "discover":
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        discover([s.strip() for s in args.seeds.split(",")], args.pool, args.per_bin,
                 args.rating_min, args.rating_max, args.bin_width,
                 args.games_per_player, args.max_players, args.out)
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        analyze(args.discovery_json, args.out, args.time_limit)


if __name__ == "__main__":
    main()
