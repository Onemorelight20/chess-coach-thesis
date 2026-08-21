"""
Fetch a player's recent games from Lichess or Chess.com by username.

Each game is normalised to a dict carrying the games table's columns plus the PGN
and the known target color, so analysis never has to guess which side the user
played. Fetched one pool at a time, because the Elo models are pool-specific.
Kept free of Streamlit so it can be unit-tested on its own.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import chess.pgn
import requests

from src.data_layer.ingestion import is_valid_game

LICHESS_POOLS = ["bullet", "blitz", "rapid", "classical"]
CHESSCOM_POOLS = ["bullet", "blitz", "rapid"]  # chess.com has no 'classical'

# A descriptive User-Agent is required by Chess.com and polite for Lichess.
_USER_AGENT = "AI-Chess-Coach/1.0 (thesis project; contact: bogdanbor.etc@gmail.com)"
_TIMEOUT = 20
_RESULT_RE = re.compile(r'\[Result\s+"([^"]+)"\]')


class FetchError(Exception):
    """User-facing fetch failure (bad username, rate limit, network error)."""


@dataclass
class FetchPage:
    games: list                 # list[dict] of normalized games (see _normalize_*)
    cursor: Optional[object]    # opaque token to pass back for the next page
    has_more: bool              # whether another page is (likely) available


# --------------------------------------------------------------------- helpers
def _result_from_pov(result_header: str, color: str) -> str:
    """Map a PGN Result tag + the player's color to Win/Loss/Draw."""
    if result_header == "1/2-1/2":
        return "Draw"
    if result_header == "1-0":
        return "Win" if color == "white" else "Loss"
    if result_header == "0-1":
        return "Win" if color == "black" else "Loss"
    return "?"


def _result_header(pgn: str) -> str:
    m = _RESULT_RE.search(pgn or "")
    return m.group(1) if m else "*"


def _analysis_note(pgn: str) -> str:
    """Empty string if the game can be analysed; otherwise a short reason why not."""
    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
    except Exception:  # noqa: BLE001
        game = None
    if game is None:
        return "Unreadable PGN"
    if not is_valid_game(game):
        return "Too short (<10 moves)"
    return ""


def _format_time(base_seconds, increment) -> str:
    """180,0 -> '3+0' ; 60,1 -> '1+1' ; 30,0 -> '30s+0'."""
    base, inc = int(base_seconds or 0), int(increment or 0)
    if base and base % 60 == 0:
        return f"{base // 60}+{inc}"
    return f"{base}s+{inc}"


def _ms_to_date(ms) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _http_get(url, *, params=None, accept=None, stream=False, platform="server"):
    headers = {"User-Agent": _USER_AGENT}
    if accept:
        headers["Accept"] = accept
    try:
        resp = requests.get(url, params=params, headers=headers,
                            timeout=_TIMEOUT, stream=stream)
    except requests.RequestException as e:
        raise FetchError(f"Network error contacting {platform}: {e}") from e
    if resp.status_code == 404:
        raise FetchError("Player not found — check the username and platform.")
    if resp.status_code == 429:
        raise FetchError(f"{platform} rate limit hit — wait a minute and try again.")
    if resp.status_code != 200:
        raise FetchError(f"{platform} returned HTTP {resp.status_code}.")
    return resp


# --------------------------------------------------------------------- Lichess
def _fetch_lichess(username, pool, max_games, cursor) -> FetchPage:
    params = {
        "max": max_games,
        "perfType": pool,
        "pgnInJson": "true",
        "clocks": "true",
        "sort": "dateDesc",
    }
    if cursor:
        params["until"] = cursor
    resp = _http_get(f"https://lichess.org/api/games/user/{username}",
                     params=params, accept="application/x-ndjson",
                     stream=True, platform="Lichess")

    games, oldest = [], None
    uname = username.lower()
    for line in resp.iter_lines():
        if not line:
            continue
        g = json.loads(line)
        created = g.get("createdAt")
        oldest = created if oldest is None else min(oldest, created)

        players = g.get("players", {})
        white, black = players.get("white", {}), players.get("black", {})
        wname = ((white.get("user") or {}).get("name") or "Anonymous")
        bname = ((black.get("user") or {}).get("name") or "Anonymous")
        if wname.lower() == uname:
            color, opp, opp_elo, my_elo = "white", bname, black.get("rating"), white.get("rating")
        else:
            color, opp, opp_elo, my_elo = "black", wname, white.get("rating"), black.get("rating")

        clock = g.get("clock") or {}
        pgn = g.get("pgn", "")
        games.append({
            "pgn": pgn,
            "time_control": _format_time(clock.get("initial"), clock.get("increment")),
            "pool": g.get("speed", pool),
            "opponent": opp,
            "opponent_elo": opp_elo,
            "player_elo": my_elo,
            "result": _result_from_pov(_result_header(pgn), color),
            "color": color,
            "played_at": _ms_to_date(created),
            "url": f"https://lichess.org/{g.get('id', '')}",
            "note": _analysis_note(pgn),
        })

    has_more = len(games) == max_games
    next_cursor = (oldest - 1) if (oldest and has_more) else None
    return FetchPage(games, next_cursor, has_more)


# -------------------------------------------------------------------- Chess.com
def _chesscom_archives(username) -> list:
    data = _http_get(
        f"https://api.chess.com/pub/player/{username}/games/archives",
        platform="Chess.com").json()
    return data.get("archives", [])


def _chesscom_filtered_month(url, pool, uname) -> list:
    data = _http_get(url, platform="Chess.com").json()
    games = [g for g in data.get("games", [])
             if g.get("time_class") == pool and g.get("rules", "chess") == "chess"]
    games.sort(key=lambda g: g.get("end_time", 0), reverse=True)
    return games


def _parse_chesscom_tc(tc: str):
    """'180' -> (180, 0) ; '180+2' -> (180, 2) ; '1/86400' (daily) -> (0, 0)."""
    if not tc or "/" in tc:
        return 0, 0
    if "+" in tc:
        base, inc = tc.split("+", 1)
        return (int(base) if base.isdigit() else 0,
                int(inc) if inc.isdigit() else 0)
    return (int(tc) if tc.isdigit() else 0), 0


def _normalize_chesscom(g, uname) -> dict:
    white, black = g.get("white", {}), g.get("black", {})
    if (white.get("username", "")).lower() == uname:
        color, opp_side, my_side = "white", black, white
    else:
        color, opp_side, my_side = "black", white, black
    pgn = g.get("pgn", "")
    base, inc = _parse_chesscom_tc(g.get("time_control", ""))
    return {
        "pgn": pgn,
        "time_control": _format_time(base, inc),
        "pool": g.get("time_class", ""),
        "opponent": opp_side.get("username", "?"),
        "opponent_elo": opp_side.get("rating"),
        "player_elo": my_side.get("rating"),
        "result": _result_from_pov(_result_header(pgn), color),
        "color": color,
        "played_at": _ms_to_date((g.get("end_time") or 0) * 1000),
        "url": g.get("url", ""),
        "note": _analysis_note(pgn),
    }


def _fetch_chesscom(username, pool, max_games, cursor) -> FetchPage:
    uname = username.lower()
    archives = list(reversed(_chesscom_archives(uname)))  # newest month first
    if not archives:
        return FetchPage([], None, False)

    arch_idx, offset = cursor if cursor else (0, 0)
    out = []
    while arch_idx < len(archives):
        month = _chesscom_filtered_month(archives[arch_idx], pool, uname)
        i = offset
        while i < len(month) and len(out) < max_games:
            out.append(_normalize_chesscom(month[i], uname))
            i += 1
        if len(out) >= max_games:
            if i < len(month):                       # more games left in this month
                return FetchPage(out, (arch_idx, i), True)
            return FetchPage(out, (arch_idx + 1, 0), arch_idx + 1 < len(archives))
        arch_idx += 1                                # month exhausted, go older
        offset = 0
    return FetchPage(out, None, False)


# ----------------------------------------------------------------------- public
def fetch_user_games(platform: str, username: str, pool: str,
                     max_games: int = 10, cursor=None) -> FetchPage:
    """
    Fetch up to `max_games` of `username`'s games in the given `pool`
    (bullet/blitz/rapid/classical) from `platform` ('lichess' or 'chess.com'),
    most recent first. Pass the returned `cursor` back to page to older games.
    Raises FetchError with a user-friendly message on any failure.
    """
    username = (username or "").strip()
    if not username:
        raise FetchError("Please enter a username.")
    key = platform.lower().replace(" ", "").replace(".", "")
    if key == "lichess":
        return _fetch_lichess(username, pool, max_games, cursor)
    if key in ("chesscom", "chess"):
        return _fetch_chesscom(username, pool, max_games, cursor)
    raise FetchError(f"Unknown platform: {platform}")
