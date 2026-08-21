import os
import random

import requests

try:
    import pandas as pd
except ImportError:  # the API fallback below works without the local index
    pd = None

from src.features.extractor import DISPLAY_NAMES

LICHESS_PUZZLE_NEXT = "https://lichess.org/api/puzzle/next"
PUZZLE_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "puzzles.parquet",
)

# Module-level cache so the ~5M-row puzzle index is loaded once per process.
_PUZZLES = None
_PUZZLES_TRIED = False


def _load_puzzles():
    global _PUZZLES, _PUZZLES_TRIED
    if not _PUZZLES_TRIED:
        _PUZZLES_TRIED = True
        if pd is not None and os.path.exists(PUZZLE_DB_PATH):
            _PUZZLES = pd.read_parquet(PUZZLE_DB_PATH)
    return _PUZZLES


class RecommenderEngine:
    """
    Turns a SHAP-identified weakness into a training puzzle whose theme matches
    the weakness and whose rating matches the predicted Elo.

    Rating matching is ABSOLUTE when the local puzzle index (puzzles.parquet) is
    present; the live API fallback only offers a difficulty relative to 1500.
    """

    def __init__(self, timeout: float = 6.0, window: int = 80, seed=None):
        self.timeout = timeout
        self.window = window                 # +/- rating band around predicted Elo
        self.rng = random.Random(seed)
        self.feature_to_tag = {
            # ---- accuracy ----
            "acpl": "mix",
            "inaccuracy_rate": "quietMove",
            "mistake_rate": "advantage",
            "blunder_rate": "hangingPiece",
            # ---- accuracy by phase ----
            "opening_acpl": "opening",
            "middlegame_acpl": "middlegame",
            "endgame_acpl": "endgame",
            # ---- situational accuracy ----
            "conversion_acpl": "crushing",
            "defense_acpl": "defensiveMove",
            # ---- style / activity ----
            "engine_match_rate": "advantage",
            "capture_rate": "capturingDefender",
            "check_given_rate": "attackingF2F7",
            # ---- tactical / positional ----
            "hanging_rate": "trappedPiece",
            "king_danger": "exposedKing",
            "mobility_avg": "advancedPawn",
            "center_control_avg": "opening",
            "material_volatility": "sacrifice",
            "pin_exposure": "pin",
            # ---- time management (mapped to puzzle tempo) ----
            "think_time_consistency": "intermezzo",
            "avg_think_time": "long",
            "time_pressure_rate": "short",
            # ---- outcome-aligned accuracy (win-probability) ----
            "wp_loss_mean": "advantage",
            "wp_blunder_rate": "hangingPiece",
            "cpl_std": "quietMove",
            # ---- engine move-agreement & criticality (MultiPV) ----
            "top1_agree": "mix",
            "top3_agree": "mix",
            "avg_move_rank": "mix",
            "acpl_critical": "crushing",
            "critical_agree": "oneMove",
        }

    # ---- absolute-rating selection from the local puzzle index ----------------
    def _select_from_db(self, theme: str, predicted_elo: int):
        """Pick a puzzle of `theme` rated near predicted_elo. Returns (dict, window)."""
        puzzles = _load_puzzles()
        if puzzles is None:
            return None, None
        for w in (self.window, 150, 250, 400, 700):
            band = puzzles[(puzzles.Rating >= predicted_elo - w)
                           & (puzzles.Rating <= predicted_elo + w)]
            if band.empty:
                continue
            themed = band[(" " + band.Themes + " ").str.contains(f" {theme} ", regex=False)]
            if themed.empty:
                continue
            # Prefer well-tested puzzles when enough exist.
            good = themed[themed.Popularity >= 80]
            pool = good if len(good) >= 5 else themed
            row = pool.iloc[self.rng.randrange(len(pool))]
            return ({"id": row.PuzzleId, "rating": int(row.Rating),
                     "themes": str(row.Themes).split()}, w)
        return None, None

    # ---- live API fallback (relative difficulty only) -------------------------
    @staticmethod
    def _difficulty_band(predicted_elo: int) -> str:
        if predicted_elo < 1200:
            return "easiest"
        if predicted_elo < 1400:
            return "easier"
        if predicted_elo <= 1600:
            return "normal"
        if predicted_elo <= 1850:
            return "harder"
        return "hardest"

    def _fetch_from_api(self, theme: str, predicted_elo: int, color):
        params = {"angle": theme, "difficulty": self._difficulty_band(predicted_elo)}
        if color in ("white", "black"):
            params["color"] = color
        try:
            resp = requests.get(LICHESS_PUZZLE_NEXT, params=params, timeout=self.timeout)
            resp.raise_for_status()
            p = resp.json().get("puzzle", {})
            if not p.get("id"):
                return None
            return {"id": p["id"], "rating": p.get("rating"), "themes": p.get("themes", [])}
        except (requests.RequestException, ValueError):
            return None

    def get_recommendation(self, predicted_elo: int, weakest_feature: str,
                           color: str = None) -> dict:
        tag = self.feature_to_tag.get(weakest_feature, "mix")
        display = DISPLAY_NAMES.get(weakest_feature, weakest_feature)

        result = {"weakness_identified": weakest_feature, "puzzle_theme": tag,
                  "target_elo": predicted_elo}

        puzzle, used_window = self._select_from_db(tag, predicted_elo)
        source = "database"
        if puzzle is None:
            puzzle = self._fetch_from_api(tag, predicted_elo, color)
            source = "api"

        if puzzle is not None:
            rating = puzzle["rating"]
            if source == "database":
                match = (f"rated **{rating}** (within ±{used_window} of your "
                         f"predicted {predicted_elo} Elo)")
            else:
                match = (f"rated {rating} ({self._difficulty_band(predicted_elo)} "
                         f"relative to Lichess's 1500 baseline)")
            result.update(
                puzzle_id=puzzle["id"],
                puzzle_rating=rating,
                puzzle_themes=puzzle["themes"],
                source=source,
                link=f"https://lichess.org/training/{puzzle['id']}",
                message=(f"SHAP attributes the largest drop in your predicted rating to "
                         f"**{display}**. Here is a '{tag}' puzzle {match} that targets "
                         f"exactly this gap."),
            )
        else:
            result.update(
                puzzle_id=None, puzzle_rating=None, puzzle_themes=[tag], source="theme_page",
                link=f"https://lichess.org/training/{tag}",
                message=(f"SHAP attributes the largest drop in your predicted rating to "
                         f"**{display}**. Drilling '{tag}' puzzles around {predicted_elo} "
                         f"Elo targets this gap. (No local puzzle index and the Lichess API "
                         f"was unreachable; showing the theme trainer.)"),
            )
        return result
