import numpy as np

# Single source of truth for the feature vector and its ordering.
FEATURE_NAMES = [
    # accuracy
    "acpl", "inaccuracy_rate", "mistake_rate", "blunder_rate",
    # accuracy by phase (phases defined by piece count)
    "opening_acpl", "middlegame_acpl", "endgame_acpl",
    # situational accuracy
    "conversion_acpl", "defense_acpl",
    # style / activity
    "engine_match_rate", "capture_rate", "check_given_rate",
    # tactical / positional
    "hanging_rate", "king_danger", "mobility_avg", "center_control_avg",
    "material_volatility", "pin_exposure",
    # time management
    "think_time_consistency", "avg_think_time", "time_pressure_rate",
    # outcome-aligned accuracy (win probability derived from Stockfish's eval)
    "wp_loss_mean", "wp_blunder_rate", "cpl_std",
    # engine move-agreement & criticality (MultiPV)
    "top1_agree", "top3_agree", "avg_move_rank", "acpl_critical", "critical_agree",
]

# Every feature maps to a Lichess puzzle theme (RecommenderEngine.feature_to_tag).
WEAKNESS_CANDIDATES = list(FEATURE_NAMES)

DISPLAY_NAMES = {
    "acpl": "Average Centipawn Loss",
    "inaccuracy_rate": "Inaccuracy Rate",
    "mistake_rate": "Mistake Rate",
    "blunder_rate": "Blunder Rate",
    "opening_acpl": "Opening Accuracy (ACPL)",
    "middlegame_acpl": "Middlegame Accuracy (ACPL)",
    "endgame_acpl": "Endgame Accuracy (ACPL)",
    "conversion_acpl": "Advantage Conversion (ACPL when winning)",
    "defense_acpl": "Defensive Accuracy (ACPL when losing)",
    "engine_match_rate": "Engine-Match Rate (best-move %)",
    "capture_rate": "Capture Rate",
    "check_given_rate": "Check-Giving Rate",
    "hanging_rate": "Hanging-Piece Rate",
    "king_danger": "King Danger (attackers near king)",
    "mobility_avg": "Average Mobility (legal moves)",
    "center_control_avg": "Center Control",
    "material_volatility": "Material Lost in Swings (avg pawns/round)",
    "pin_exposure": "Pin Exposure",
    "think_time_consistency": "Think-Time Consistency (std s/move)",
    "avg_think_time": "Average Think Time (s/move)",
    "time_pressure_rate": "Time-Pressure Rate (<15s)",
    "wp_loss_mean": "Win-Probability Lost per Move",
    "wp_blunder_rate": "Blunder Rate (win-prob drop ≥ 10%)",
    "cpl_std": "Accuracy Consistency (std of centipawn loss)",
    "top1_agree": "Top-1 Engine-Match Rate",
    "top3_agree": "Top-3 Engine-Match Rate",
    "avg_move_rank": "Average Move Rank (1 = engine's best)",
    "acpl_critical": "Accuracy in Critical Positions (ACPL)",
    "critical_agree": "Best-Move Rate in Critical Positions",
}

# +1 = higher value means weaker play, -1 = the reverse. Needed wherever a
# weakness is inferred from raw feature levels rather than from the model.
_HIGHER_IS_BETTER = {
    "engine_match_rate", "mobility_avg", "center_control_avg", "avg_think_time",
    "top1_agree", "top3_agree", "critical_agree",
}
FEATURE_ORIENTATION = {f: (-1 if f in _HIGHER_IS_BETTER else 1) for f in FEATURE_NAMES}

# Logistic centipawns -> win probability (Lichess / Maia, KDD 2020).
WP_K = 0.00368208

OPENING_MIN_PIECES = 28
ENDGAME_MAX_PIECES = 12
WINNING_CP = 200
LOSING_CP = -200
TIME_PRESSURE_S = 15.0


class FeatureExtractor:
    def __init__(self, target_player_color: str = "white"):
        self.target_color = target_player_color.lower()

    def extract(self, analysis_results: list, increment: float = 0.0) -> dict:
        moves = [m for m in analysis_results if m["turn"] == self.target_color]
        n = len(moves)
        if n == 0:
            return {f: np.nan for f in FEATURE_NAMES}

        white = self.target_color == "white"
        cpl = np.array([m["cpl"] for m in moves], dtype=float)
        peval = np.array([m["eval_cp"] if white else -m["eval_cp"] for m in moves], dtype=float)
        pieces = np.array([m["piece_count"] for m in moves], dtype=float)
        material = np.array([m["material"] for m in moves], dtype=float)

        acpl = float(cpl.mean())

        # Material net-lost per move-round; downside-only, so it is not zero-sum.
        signed_material = material if white else -material
        if n > 1:
            swings = np.diff(signed_material)
            losses = -swings[swings < 0]
            material_loss = float(losses.mean()) if losses.size else 0.0
        else:
            material_loss = 0.0

        def subset_acpl(mask):
            return float(cpl[mask].mean()) if mask.any() else acpl

        # cpl = eval_best - eval_played (mover POV), so eval_best = peval + cpl.
        wp_after = 1.0 / (1.0 + np.exp(-WP_K * np.clip(peval, -2000, 2000)))
        wp_before = 1.0 / (1.0 + np.exp(-WP_K * np.clip(peval + cpl, -2000, 2000)))
        wp_loss = np.maximum(0.0, wp_before - wp_after)

        ranks = np.array([m.get("move_rank", np.nan) for m in moves], dtype=float)
        critical = np.array([bool(m.get("is_critical", False)) for m in moves])

        f = {
            "acpl": acpl,
            "inaccuracy_rate": float(((cpl >= 50) & (cpl < 100)).mean()),
            "mistake_rate": float(((cpl >= 100) & (cpl < 300)).mean()),
            "blunder_rate": float((cpl >= 300).mean()),
            "opening_acpl": subset_acpl(pieces >= OPENING_MIN_PIECES),
            "middlegame_acpl": subset_acpl((pieces > ENDGAME_MAX_PIECES) & (pieces < OPENING_MIN_PIECES)),
            "endgame_acpl": subset_acpl(pieces <= ENDGAME_MAX_PIECES),
            "conversion_acpl": subset_acpl(peval > WINNING_CP),
            "defense_acpl": subset_acpl(peval < LOSING_CP),
            "engine_match_rate": float((cpl == 0).mean()),
            "capture_rate": float(np.mean([m["is_capture"] for m in moves])),
            "check_given_rate": float(np.mean([m["is_check"] for m in moves])),
            "hanging_rate": float(np.mean([m["hanging"] for m in moves])),
            "king_danger": float(np.mean([m["king_danger"] for m in moves])),
            "mobility_avg": float(np.mean([m["mobility"] for m in moves])),
            "center_control_avg": float(np.mean([m["center_control"] for m in moves])),
            "material_volatility": material_loss,
            "pin_exposure": float(np.mean([m["pins"] for m in moves])),
            "wp_loss_mean": float(wp_loss.mean()),
            "wp_blunder_rate": float((wp_loss >= 0.10).mean()),
            "cpl_std": float(cpl.std()),
            "top1_agree": float(np.mean(ranks == 1)),
            "top3_agree": float(np.mean(ranks <= 3)),
            "avg_move_rank": float(np.nanmean(ranks)),
            "acpl_critical": float(cpl[critical].mean()) if critical.any() else acpl,
            "critical_agree": (float(np.mean(ranks[critical] == 1))
                               if critical.any() else float(np.mean(ranks == 1))),
        }

        # Think time per move from consecutive clock readings of the same player.
        clocks = [m["clock"] for m in moves]
        spent = []
        for i in range(1, n):
            if clocks[i] is not None and clocks[i - 1] is not None:
                dt = clocks[i - 1] - clocks[i] + increment
                if dt >= 0:
                    spent.append(dt)
        if len(spent) >= 2:
            f["think_time_consistency"] = float(np.std(spent))
            f["avg_think_time"] = float(np.mean(spent))
        else:
            f["think_time_consistency"] = np.nan
            f["avg_think_time"] = np.nan
        valid_clocks = [c for c in clocks if c is not None]
        f["time_pressure_rate"] = (float(np.mean([c < TIME_PRESSURE_S for c in valid_clocks]))
                                   if valid_clocks else np.nan)

        return {k: f[k] for k in FEATURE_NAMES}
