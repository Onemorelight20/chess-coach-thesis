"""
Rating-targeted seed finder for the cross-platform crawl.

The snowball in `collect_crossplatform.py` follows opponents, and Chess.com
matchmaking pairs similar ratings -- so a crawl started from low-rated seeds
stays in the low-rated basin. In `bullet` that basin is the whole population
(median encountered rating ~440), and 3 830 requests filled every bin below
2000 while finding exactly one player above 2200.

This picks seeds that already sit in the empty bins, using two public endpoints
that the snowball never touches:
    /pub/titled/{TITLE}   -- CM/NM/FM are typically 2000-2600 bullet, IM/GM higher
    /pub/player/{u}/stats -- current rating, so a candidate is checked for ONE
                             cheap request instead of a full 12-game fetch

Usage:
    python -m src.data_layer.seed_by_rating --pool bullet \\
        --bins 2000-2200,2200-2400,2400-2600,2600-2800 --per-bin 8
"""

import argparse
import json
import random
import sys
import time
import urllib.request

TITLED_URL = "https://api.chess.com/pub/titled/{title}"
COUNTRY_URL = "https://api.chess.com/pub/country/{iso}/players"
STATS_URL = "https://api.chess.com/pub/player/{user}/stats"
UA = {"User-Agent": "thesis-research/1.0 (masters thesis; contact via chess.com)"}
DELAY = 0.3
POOL_KEY = {"bullet": "chess_bullet", "blitz": "chess_blitz", "rapid": "chess_rapid"}

# Titled players cover 2000+ but nothing below; country lists (capped at 10 000
# each by the API) are an untargeted draw and so cover the mass of the scale.
# Spread over continents so the pool is not one federation's playing culture.
DEFAULT_COUNTRIES = "US,IN,UA,PL,DE,BR,ES,PH,GB,CA,AR,FR,RU,VN,ID,NG,TR,IT,MX,AU"


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def candidate_pool(titles, countries, seed=42):
    """Usernames from the requested sources, shuffled deterministically."""
    names = []
    for t in titles:
        try:
            names += _get(TITLED_URL.format(title=t)).get("players", [])
            print(f"  titled {t}: cumulative {len(names)}")
        except Exception as e:  # noqa: BLE001
            print(f"  titled {t}: skipped ({e})")
        time.sleep(DELAY)
    for c in countries:
        try:
            names += _get(COUNTRY_URL.format(iso=c), timeout=120).get("players", [])
            print(f"  country {c}: cumulative {len(names)}")
        except Exception as e:  # noqa: BLE001
            print(f"  country {c}: skipped ({e})")
        time.sleep(DELAY)
    seen, uniq = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            uniq.append(n)
    random.Random(seed).shuffle(uniq)
    return uniq


def ratings_of(user, pools):
    """
    Current rating per pool, {} on failure. One /stats call carries all three
    pools, so seeding several pools at once costs the same as seeding one.
    """
    try:
        s = _get(STATS_URL.format(user=user))
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for p in pools:
        r = (s.get(POOL_KEY[p], {}).get("last") or {}).get("rating")
        if r is not None:
            out[p] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="bullet",
                    choices=["bullet", "blitz", "rapid", "all"],
                    help="'all' fills every pool's bins from the same /stats calls")
    ap.add_argument("--bins", required=True, help="e.g. 2000-2200,2200-2400")
    ap.add_argument("--per-bin", type=int, default=8)
    ap.add_argument("--titles", default="CM,NM,FM,IM,GM,WGM,WIM,WFM",
                    help="comma-separated, or empty string to skip titled players")
    ap.add_argument("--countries", default="",
                    help=f"comma-separated ISO codes, or 'default' for {DEFAULT_COUNTRIES}")
    ap.add_argument("--max-checks", type=int, default=1500)
    ap.add_argument("--out", default=None, help="write usernames here, one per line")
    args = ap.parse_args()

    bins = []
    for b in args.bins.split(","):
        lo, hi = b.split("-")
        bins.append((int(lo), int(hi)))

    countries = args.countries
    if countries == "default":
        countries = DEFAULT_COUNTRIES
    titles = [t for t in args.titles.split(",") if t]
    countries = [c for c in countries.split(",") if c]
    print(f"Collecting candidates (titles={titles or '-'}, countries={countries or '-'})...")
    names = candidate_pool(titles, countries)
    print(f"{len(names)} candidates; checking {args.pool} ratings "
          f"(need {args.per_bin} per bin)\n")

    pools = ["bullet", "blitz", "rapid"] if args.pool == "all" else [args.pool]
    found = {p: {b: [] for b in bins} for p in pools}

    def done():
        return all(len(v) >= args.per_bin for p in pools for v in found[p].values())

    checked = 0
    for user in names:
        if checked >= args.max_checks or done():
            break
        checked += 1
        time.sleep(DELAY)
        for pool, r in ratings_of(user, pools).items():
            for b in bins:
                if b[0] <= r < b[1] and len(found[pool][b]) < args.per_bin:
                    found[pool][b].append(user)
                    print(f"  [{checked}] {user:<24} {pool:<6} {r} -> "
                          f"bin {b[0]}-{b[1]} ({len(found[pool][b])}/{args.per_bin})")
                    break

    print(f"\nChecked {checked} players.")
    wrote = False
    for pool in pools:
        picked = []
        print(f"\n{pool}:")
        for b in bins:
            print(f"  {b[0]}-{b[1]}: {len(found[pool][b])}/{args.per_bin}")
            picked += found[pool][b]
        if not picked:
            continue
        if args.out:
            path = args.out if len(pools) == 1 else args.out.replace(".txt", f"_{pool}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(picked) + "\n")
            print(f"  -> {path} ({len(picked)} seeds)")
            wrote = True
        else:
            print(f"  --seeds {','.join(picked)}")
    if args.out and not wrote:
        print("No seeds found -- widen --titles/--countries or raise --max-checks.")
        sys.exit(1)


if __name__ == "__main__":
    main()
