"""
Fast unit tests (no Stockfish, no network) for the feature pipeline. They catch a
feature added to FEATURE_NAMES but silently missing from a companion mapping
(DISPLAY_NAMES, FEATURE_ORIENTATION, RecommenderEngine.feature_to_tag).
"""

import numpy as np
import pytest

from src.features.extractor import (
    FeatureExtractor, FEATURE_NAMES, DISPLAY_NAMES, WEAKNESS_CANDIDATES,
    FEATURE_ORIENTATION, WP_K,
)
from src.features.aggregate import aggregate_features
from src.ml.recommender import RecommenderEngine


def _move(turn, cpl, piece_count, eval_cp, rank, critical, material,
         is_capture=False, is_check=False, hanging=0, king_danger=0,
         mobility=20, center_control=0, pins=0, clock=None):
    return dict(turn=turn, cpl=float(cpl), piece_count=piece_count, eval_cp=float(eval_cp),
               move_rank=rank, is_critical=critical, material=float(material),
               is_capture=is_capture, is_check=is_check, hanging=hanging,
               king_danger=king_danger, mobility=mobility, center_control=center_control,
               pins=pins, clock=clock)


@pytest.fixture
def synthetic_analysis():
    """
    4 target (white) moves with hand-computable stats, interleaved with 2 black
    moves whose cpl=999 would corrupt every white-only statistic if the color
    filter in extract() were broken.
    """
    white_moves = [
        _move("white", cpl=0, piece_count=32, eval_cp=20, rank=1, critical=True,
             material=0, mobility=30, pins=0, king_danger=0, center_control=2, clock=100),
        _move("white", cpl=60, piece_count=30, eval_cp=-40, rank=2, critical=False,
             material=-1, is_capture=True, mobility=28, pins=0, king_danger=1,
             center_control=1, clock=90),
        _move("white", cpl=150, piece_count=10, eval_cp=-190, rank=3, critical=True,
             material=-1, is_check=True, mobility=15, pins=1, king_danger=2,
             center_control=0, clock=70),
        _move("white", cpl=400, piece_count=8, eval_cp=-590, rank=5, critical=False,
             material=-3, hanging=2, mobility=10, pins=1, king_danger=3,
             center_control=0, clock=40),
    ]
    black_moves = [
        _move("black", cpl=999, piece_count=31, eval_cp=999, rank=1, critical=False, material=999),
        _move("black", cpl=999, piece_count=9, eval_cp=999, rank=1, critical=False, material=999),
    ]
    return white_moves[:1] + black_moves[:1] + white_moves[1:] + black_moves[1:]


def test_extractor_matches_hand_computed_stats(synthetic_analysis):
    feats = FeatureExtractor(target_player_color="white").extract(synthetic_analysis, increment=0.0)

    # accuracy (cpl = [0, 60, 150, 400])
    assert feats["acpl"] == pytest.approx(152.5)
    assert feats["inaccuracy_rate"] == pytest.approx(0.25)   # only the cpl=60 move
    assert feats["mistake_rate"] == pytest.approx(0.25)      # only the cpl=150 move
    assert feats["blunder_rate"] == pytest.approx(0.25)      # only the cpl=400 move
    assert feats["engine_match_rate"] == pytest.approx(0.25)  # only the cpl=0 move

    # phase (pieces = [32, 30, 10, 8]; opening>=28, endgame<=12)
    assert feats["opening_acpl"] == pytest.approx(30.0)       # mean(cpl[0], cpl[1]) = mean(0, 60)
    assert feats["middlegame_acpl"] == pytest.approx(feats["acpl"])  # no ply in that band -> falls back
    assert feats["endgame_acpl"] == pytest.approx(275.0)      # mean(cpl[2], cpl[3]) = mean(150, 400)

    # situational (peval = eval_cp, since color=white; conversion>200, defense<-200)
    assert feats["conversion_acpl"] == pytest.approx(feats["acpl"])  # no ply above +200 -> fallback
    assert feats["defense_acpl"] == pytest.approx(400.0)      # only the -590 ply is < -200

    # style / tactical (raw means, not booleans)
    assert feats["capture_rate"] == pytest.approx(0.25)
    assert feats["check_given_rate"] == pytest.approx(0.25)
    assert feats["hanging_rate"] == pytest.approx(0.5)        # mean([0, 0, 0, 2])
    assert feats["king_danger"] == pytest.approx(1.5)         # mean([0, 1, 2, 3])
    assert feats["mobility_avg"] == pytest.approx(20.75)      # mean([30, 28, 15, 10])
    assert feats["center_control_avg"] == pytest.approx(0.75)  # mean([2, 1, 0, 0])
    assert feats["pin_exposure"] == pytest.approx(0.5)        # mean([0, 0, 1, 1])
    assert feats["material_volatility"] == pytest.approx(1.5)  # net-loss swings: 1, 2 -> mean 1.5

    # MultiPV agreement / criticality (ranks = [1, 2, 3, 5]; critical = [T, F, T, F])
    assert feats["top1_agree"] == pytest.approx(0.25)
    assert feats["top3_agree"] == pytest.approx(0.75)
    assert feats["avg_move_rank"] == pytest.approx(2.75)
    assert feats["acpl_critical"] == pytest.approx(75.0)      # mean(cpl[0], cpl[2]) = mean(0, 150)
    assert feats["critical_agree"] == pytest.approx(0.5)      # ranks at critical plies = [1, 3]

    # think-time (clocks = [100, 90, 70, 40], increment=0 -> spent = [10, 20, 30])
    assert feats["avg_think_time"] == pytest.approx(20.0)
    assert feats["think_time_consistency"] == pytest.approx(np.std([10, 20, 30]))
    assert feats["time_pressure_rate"] == pytest.approx(0.0)  # nothing below 15s

    # Re-derived independently, since hand-computing the sigmoid isn't practical.
    cpl = np.array([0, 60, 150, 400], dtype=float)
    peval = np.array([20, -40, -190, -590], dtype=float)
    wp_after = 1.0 / (1.0 + np.exp(-WP_K * peval))
    wp_before = 1.0 / (1.0 + np.exp(-WP_K * (peval + cpl)))
    expected_wp_loss = np.maximum(0.0, wp_before - wp_after)
    assert feats["wp_loss_mean"] == pytest.approx(expected_wp_loss.mean())
    assert feats["wp_blunder_rate"] == pytest.approx(float((expected_wp_loss >= 0.10).mean()))
    assert feats["cpl_std"] == pytest.approx(np.std(cpl))

    # every declared feature must be present, and nothing extra
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_extractor_no_target_moves_returns_all_nan():
    feats = FeatureExtractor(target_player_color="white").extract(
        [_move("black", cpl=10, piece_count=32, eval_cp=0, rank=1, critical=False, material=0)])
    assert set(feats.keys()) == set(FEATURE_NAMES)
    assert all(v != v for v in feats.values())  # NaN != NaN


def test_aggregate_features_mean_and_median():
    a = {f: 10.0 for f in FEATURE_NAMES}
    b = {f: 20.0 for f in FEATURE_NAMES}
    c = {f: 30.0 for f in FEATURE_NAMES}

    mean_agg = aggregate_features([a, b, c], method="mean")
    assert mean_agg["acpl"] == pytest.approx(20.0)

    median_agg = aggregate_features([a, b, c], method="median")
    assert median_agg["acpl"] == pytest.approx(20.0)

    # a feature missing from every game dict stays NaN, not a crash
    empty_agg = aggregate_features([], method="mean")
    assert all(v != v for v in empty_agg.values())

    with pytest.raises(ValueError):
        aggregate_features([a, b], method="not_a_real_method")


def test_feature_orientation_covers_every_feature_and_known_directions():
    assert set(FEATURE_ORIENTATION) == set(FEATURE_NAMES)
    assert set(FEATURE_ORIENTATION.values()) <= {1, -1}
    # loss-like: higher = weaker play
    assert FEATURE_ORIENTATION["acpl"] == 1
    assert FEATURE_ORIENTATION["blunder_rate"] == 1
    assert FEATURE_ORIENTATION["avg_move_rank"] == 1
    # agreement/activity-like: higher = stronger play
    assert FEATURE_ORIENTATION["engine_match_rate"] == -1
    assert FEATURE_ORIENTATION["top1_agree"] == -1
    assert FEATURE_ORIENTATION["top3_agree"] == -1
    assert FEATURE_ORIENTATION["critical_agree"] == -1


def test_display_names_and_weakness_candidates_cover_every_feature():
    assert set(DISPLAY_NAMES) == set(FEATURE_NAMES)
    assert set(WEAKNESS_CANDIDATES) == set(FEATURE_NAMES)


def test_recommender_has_a_puzzle_tag_for_every_feature():
    r = RecommenderEngine()
    missing = [f for f in FEATURE_NAMES if f not in r.feature_to_tag]
    assert not missing, f"no puzzle tag mapped for: {missing}"
