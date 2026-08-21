"""
Player-grouped dataset collector, restricted to one time-control class since Elo
and centipawn loss are not comparable across pools. Each row is one (game, target
player) pair carrying the player identity and a game id, which is what makes
leakage-free GroupKFold possible.

Pass 1 (headers only): scan the PGN, bucket byte offsets by player.
Pass 2 (engine, parallel): run Stockfish once per selected game and extract a
feature vector for every target player in it.
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import chess.pgn
import pandas as pd
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.engine.analyzer import EngineAnalyzer
from src.features.extractor import FeatureExtractor, FEATURE_NAMES

FLUSH_EVERY = 200  # resumable checkpoint interval, in completed games


def parse_increment(time_control: str) -> float:
    """'180+2' -> 2.0 ; '300' -> 0.0 ; '-' -> 0.0"""
    if not time_control or "+" not in time_control:
        return 0.0
    try:
        return float(time_control.split("+")[1])
    except (ValueError, IndexError):
        return 0.0


def pool_of(time_control: str) -> str:
    """Lichess speed class from estimated duration = base + 40*increment."""
    if not time_control or time_control == "-":
        return "unknown"
    try:
        base = int(time_control.split("+")[0])
    except (ValueError, IndexError):
        return "unknown"
    est = base + 40 * parse_increment(time_control)
    if est < 179:
        return "bullet"
    if est < 479:
        return "blitz"
    if est < 1499:
        return "rapid"
    return "classical"


# ---- Pass 2 worker (top-level, so multiprocessing can pickle it) -------------
_WORKER_CFG = {}


def _init_worker(pgn_path, time_limit, depth, engine_path):
    """Each worker keeps one open file handle and one engine for its lifetime."""
    _WORKER_CFG["fh"] = open(pgn_path, "r", encoding="utf-8")
    _WORKER_CFG["analyzer"] = EngineAnalyzer(
        engine_path=engine_path, time_limit=time_limit, depth=depth
    )


def _process_game(task):
    """
    task = (offset, time_control, [(color, elo, player), ...])
    Returns a list of feature-row dicts (one per target player in this game).
    """
    offset, time_control, targets = task
    fh = _WORKER_CFG["fh"]
    analyzer = _WORKER_CFG["analyzer"]

    try:
        fh.seek(offset)
        game = chess.pgn.read_game(fh)
        if game is None:
            return []
        analysis = analyzer.analyze_game(game)
    except Exception as e:  # noqa: BLE001 - a bad game must not kill the pool
        print(f"[worker] game at offset {offset} failed: {e}", file=sys.stderr)
        return []

    increment = parse_increment(time_control)
    result = game.headers.get("Result", "*")
    eco = game.headers.get("ECO", "?")
    pool = pool_of(time_control)
    rows = []
    for color, elo, player in targets:
        feats = FeatureExtractor(target_player_color=color).extract(
            analysis, increment=increment
        )
        if all(feats[f] != feats[f] for f in FEATURE_NAMES):  # all NaN -> no moves
            continue
        row = dict(feats)
        row.update(
            player=player,
            game_id=offset,
            target_color=color,
            target_elo=elo,
            time_control=time_control,
            pool=pool,
            eco=eco,
            result=result,
        )
        rows.append(row)
    return rows


# ---- Pass 1: header scan + player bucketing ---------------------------------
def scan_and_bucket(pgn_path, event_keyword, players, games_per_player, max_scan,
                    stop_when=None):
    buckets = defaultdict(list)  # player -> [(offset, color, elo, time_control)]
    scanned = 0
    kept = 0
    with open(pgn_path, "r", encoding="utf-8") as f:
        while scanned < max_scan:
            offset = f.tell()
            headers = chess.pgn.read_headers(f)
            if headers is None:
                break
            scanned += 1

            if event_keyword and event_keyword.lower() not in headers.get("Event", "").lower():
                continue
            we, be = headers.get("WhiteElo", "?"), headers.get("BlackElo", "?")
            if not (we.isdigit() and be.isdigit()):
                continue

            tc = headers.get("TimeControl", "-")
            buckets[headers.get("White", "?")].append((offset, "white", int(we), tc))
            buckets[headers.get("Black", "?")].append((offset, "black", int(be), tc))
            kept += 1

            if scanned % 20000 == 0:
                qualifying = sum(1 for v in buckets.values() if len(v) >= games_per_player)
                print(f"  scanned {scanned:,} games | kept {kept:,} | "
                      f"players with >={games_per_player} games: {qualifying:,}")
                done = stop_when(buckets) if stop_when else qualifying >= players
                if done:
                    break
    return buckets, scanned, kept


def representative_rating(entries):
    """Mean Elo of a player's kept games (entries are (offset, color, elo, tc))."""
    return sum(e[2] for e in entries) / len(entries)


def _bin_index(rating, rating_min, bin_width):
    return int((rating - rating_min) // bin_width)


def _qualifying_bin_counts(buckets, games_per_player, rating_min, rating_max, bin_width):
    """How many qualifying players fall in each rating bin (for the stop check)."""
    counts = defaultdict(int)
    for entries in buckets.values():
        if len(entries) < games_per_player:
            continue
        r = representative_rating(entries)
        if rating_min <= r < rating_max:
            counts[_bin_index(r, rating_min, bin_width)] += 1
    return counts


def select_players_stratified(buckets, games_per_player, rating_min, rating_max,
                              bin_width, per_bin):
    """
    Pick players spread evenly across rating bins instead of by activity, so the
    model stays calibrated at both ends. Each bin contributes up to `per_bin`
    players; sparse tail bins contribute what they have.
    """
    bins = defaultdict(list)
    for player, entries in buckets.items():
        if len(entries) < games_per_player:
            continue
        r = representative_rating(entries)
        if rating_min <= r < rating_max:
            bins[_bin_index(r, rating_min, bin_width)].append((player, entries))

    chosen = []
    for b in sorted(bins):
        chosen.extend(sorted(bins[b], key=lambda kv: len(kv[1]), reverse=True)[:per_bin])
    return chosen


def _tasks_from_chosen(chosen, games_per_player):
    """Merge the chosen players' games so each unique game is analyzed only once."""
    # game offset -> {time_control, targets: {color: (elo, player)}}
    games = {}
    for player, entries in chosen:
        for offset, color, elo, tc in entries[:games_per_player]:
            g = games.setdefault(offset, {"tc": tc, "targets": {}})
            g["targets"][color] = (elo, player)

    tasks = [
        (offset, g["tc"], [(c, elo, pl) for c, (elo, pl) in g["targets"].items()])
        for offset, g in games.items()
    ]
    return tasks, len(chosen)


def build_tasks(buckets, players, games_per_player):
    """Default selection: the most active qualifying players (skews high-rated)."""
    qualifying = sorted(
        ((p, v) for p, v in buckets.items() if len(v) >= games_per_player),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )[:players]
    return _tasks_from_chosen(qualifying, games_per_player)


def _print_rating_histogram(chosen, rating_min, rating_max, bin_width):
    counts = defaultdict(int)
    for _player, entries in chosen:
        counts[_bin_index(representative_rating(entries), rating_min, bin_width)] += 1
    print("  selected players by rating bin:")
    for i in range((rating_max - rating_min) // bin_width):
        lo = rating_min + i * bin_width
        n = counts.get(i, 0)
        print(f"    {lo:>4}-{lo + bin_width:<4} {n:>4}  {'#' * min(n, 60)}")


def run(pgn_path, output_path, event_keyword="blitz", players=150,
        games_per_player=8, max_scan=2_000_000, time_limit=0.05, depth=None,
        engine_path=None, stratified=False, rating_min=600, rating_max=2800,
        bin_width=200, per_bin=150):
    if engine_path is None:
        engine_path = os.getenv("STOCKFISH_PATH", "stockfish")

    t0 = time.time()
    if stratified:
        n_bins = (rating_max - rating_min) // bin_width
        print(f"Pass 1 (stratified): scanning '{pgn_path}' for '{event_keyword}' games — "
              f"up to {per_bin} players x {n_bins} bins of {bin_width} Elo "
              f"across {rating_min}-{rating_max}...")

        def stop(b):  # stop scanning once every rating bin has enough players
            counts = _qualifying_bin_counts(b, games_per_player, rating_min,
                                            rating_max, bin_width)
            return all(counts.get(i, 0) >= per_bin for i in range(n_bins))

        buckets, scanned, kept = scan_and_bucket(
            pgn_path, event_keyword, players, games_per_player, max_scan, stop_when=stop)
        chosen = select_players_stratified(
            buckets, games_per_player, rating_min, rating_max, bin_width, per_bin)
        tasks, chosen_players = _tasks_from_chosen(chosen, games_per_player)
        _print_rating_histogram(chosen, rating_min, rating_max, bin_width)
    else:
        print(f"Pass 1: scanning '{pgn_path}' for '{event_keyword}' games "
              f"(target {players} players x {games_per_player} games)...")
        buckets, scanned, kept = scan_and_bucket(
            pgn_path, event_keyword, players, games_per_player, max_scan)
        tasks, chosen_players = build_tasks(buckets, players, games_per_player)
    print(f"  scanned {scanned:,} games in {time.time()-t0:.0f}s; "
          f"selected {chosen_players} players -> {len(tasks):,} unique games to analyze.")

    if not tasks:
        print("No games matched the filters. Try a different --event or larger --max-scan.")
        return

    # --- resume from a checkpoint if one exists -----------------------------
    partial_path = output_path + ".partial"
    rows = []
    if os.path.exists(partial_path):
        prev = pd.read_parquet(partial_path)
        rows = prev.to_dict("records")
        done_ids = set(prev["game_id"].unique())
        before = len(tasks)
        tasks = [t for t in tasks if t[0] not in done_ids]
        print(f"Resuming from checkpoint: {len(done_ids):,} games already analyzed "
              f"({len(rows):,} rows); {len(tasks):,} of {before:,} games remaining.")

    pool_size = max(1, cpu_count() - 1)

    def flush_checkpoint():
        # Write-then-rename, so a kill mid-write cannot corrupt the checkpoint.
        tmp = partial_path + ".tmp"
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        os.replace(tmp, partial_path)

    if tasks:
        print(f"Pass 2: engine analysis on {pool_size} workers "
              f"({'depth '+str(depth) if depth else str(time_limit)+'s'}/move); "
              f"checkpoint every {FLUSH_EVERY} games -> {partial_path}")
        t0 = time.time()
        with Pool(pool_size, initializer=_init_worker,
                  initargs=(pgn_path, time_limit, depth, engine_path)) as pool:
            for i, game_rows in enumerate(tqdm.tqdm(
                    pool.imap_unordered(_process_game, tasks),
                    total=len(tasks), desc="Analyzing"), 1):
                rows.extend(game_rows)
                if i % FLUSH_EVERY == 0:
                    flush_checkpoint()
        flush_checkpoint()
        print(f"\nPass 2 done in {(time.time() - t0) / 60:.1f} min.")
    else:
        print("All selected games are already in the checkpoint; nothing to analyze.")

    df = pd.DataFrame(rows)
    # Clock-derived features stay optional, so a clockless game is still usable.
    df = df.dropna(subset=[f for f in FEATURE_NAMES if "time" not in f])
    df.to_parquet(output_path, index=False)
    if os.path.exists(partial_path):
        os.remove(partial_path)  # consolidated into the final file
    print(f"Wrote {df.shape[0]:,} rows "
          f"({df['player'].nunique()} players) x {df.shape[1]} cols -> {output_path}")
    print("Rows per player: "
          f"min={df.groupby('player').size().min()}, "
          f"median={int(df.groupby('player').size().median())}, "
          f"max={df.groupby('player').size().max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Collect a player-grouped chess feature dataset.")
    ap.add_argument("input_pgn")
    ap.add_argument("output_parquet")
    ap.add_argument("--event", default="blitz",
                    help="substring required in the PGN Event tag (blitz/rapid/classical)")
    ap.add_argument("--players", type=int, default=150)
    ap.add_argument("--games-per-player", type=int, default=8)
    ap.add_argument("--max-scan", type=int, default=2_000_000)
    ap.add_argument("--time-limit", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=None,
                    help="fixed engine depth (overrides --time-limit; reproducible)")
    ap.add_argument("--stratified", action="store_true",
                    help="sample players evenly across rating bins (full-range calibration) "
                         "instead of taking the most active players")
    ap.add_argument("--rating-min", type=int, default=600)
    ap.add_argument("--rating-max", type=int, default=2800)
    ap.add_argument("--bin-width", type=int, default=200)
    ap.add_argument("--per-bin", type=int, default=150,
                    help="max players per rating bin in --stratified mode")
    args = ap.parse_args()
    run(args.input_pgn, args.output_parquet, event_keyword=args.event,
        players=args.players, games_per_player=args.games_per_player,
        max_scan=args.max_scan, time_limit=args.time_limit, depth=args.depth,
        stratified=args.stratified, rating_min=args.rating_min, rating_max=args.rating_max,
        bin_width=args.bin_width, per_bin=args.per_bin)
