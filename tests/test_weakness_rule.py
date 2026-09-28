"""
Tests for the weakness rule (2.7): candidate pool, sign test, band reference
interpolation, degenerate statistics, fallback to the raw rule, and the margin.
No Stockfish, no network; the stored per-pool references are read if present.
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.features.extractor import (
    FEATURE_NAMES, THEMED_CANDIDATES, SIGN_EXCLUDED, TRAINABLE_CANDIDATES,
    FEATURE_ORIENTATION,
)
from src.ml import build_shap_reference as bsr
from src.ml.candidate_signs import sign_test
from src.ml.recommender import RecommenderEngine
from src.ml.model import ELOModel, margin_quantile

POOLS = ("bullet", "blitz", "rapid", "classical")


def test_candidate_pools_are_consistent():
    assert len(THEMED_CANDIDATES) == 17
    assert set(THEMED_CANDIDATES) <= set(FEATURE_NAMES)
    assert set(SIGN_EXCLUDED) <= set(THEMED_CANDIDATES)
    assert set(TRAINABLE_CANDIDATES) == set(THEMED_CANDIDATES) - set(SIGN_EXCLUDED)
    assert not set(TRAINABLE_CANDIDATES) & set(SIGN_EXCLUDED)
    # every themed feature has a puzzle theme other than the catch-all
    tags = RecommenderEngine().feature_to_tag
    assert all(tags[c] != "mix" for c in THEMED_CANDIDATES)


def test_sign_test_uses_orientation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    agg = pd.DataFrame({"blunder_rate": x, "mobility_avg": x})
    sv = np.column_stack([-x, x])                       # blunder: worse -> lower; mobility: more -> higher
    passed, table = sign_test(agg, sv, ["blunder_rate", "mobility_avg"],
                              ["blunder_rate", "mobility_avg"])
    assert passed == ["blunder_rate", "mobility_avg"]
    assert table["blunder_rate"]["rho_shap"] < 0 < table["mobility_avg"]["rho_shap"]
    # inverted contribution fails regardless of how strong the correlation is
    passed, _ = sign_test(agg, np.column_stack([x, x]), ["blunder_rate", "mobility_avg"],
                          ["blunder_rate"])
    assert passed == []


@pytest.mark.parametrize("pool", POOLS)
def test_stored_reference_matches_pool_sign_test(pool):
    ref = bsr.load(pool)
    if ref is None:
        pytest.skip(f"no reference for {pool}")
    assert set(ref["candidates"]) <= set(THEMED_CANDIDATES)
    assert set(ref["candidates"]) | set(ref["sign_excluded"]) == set(THEMED_CANDIDATES)
    assert not set(ref["candidates"]) & set(ref["sign_excluded"])
    assert list(ref["mean"].columns) == ref["candidates"]
    # stored sign table agrees with the stored split
    for c, row in ref["sign_test"].items():
        assert row["correct_sign"] == (c in ref["candidates"])


def test_stats_at_interpolates_and_clamps():
    cands = ["a", "b"]
    ref = {"candidates": cands, "band_centres": [1000.0, 2000.0],
           "mean": pd.DataFrame({"a": [0.0, 10.0], "b": [5.0, 5.0]}),
           "std": pd.DataFrame({"a": [1.0, 3.0], "b": [np.nan, np.nan]}),
           "band_edges": [0, 1500, 3000], "n_bands": 2}
    mu, sd = bsr.stats_at(ref, 1500)
    assert mu["a"] == pytest.approx(5.0) and sd["a"] == pytest.approx(2.0)
    mu, sd = bsr.stats_at(ref, 500)                     # below the first knot: constant
    assert mu["a"] == pytest.approx(0.0) and sd["a"] == pytest.approx(1.0)
    mu, sd = bsr.stats_at(ref, 5000)                    # above the last knot: constant
    assert mu["a"] == pytest.approx(10.0)
    # zero-variance candidate is dropped, never scored as NaN
    scored = bsr.standardised_scores({"a": 7.0, "b": 1.0}, mu, sd, cands)
    assert set(scored) == {"a"} and np.isfinite(scored["a"])
    m = bsr.standardise_matrix(np.array([[7.0, 1.0]]), np.array([10.0, 5.0]), np.array([3.0, 0.0]))
    assert m[0, 0] == pytest.approx(-1.0) and np.isnan(m[0, 1])


def test_predict_and_explain_falls_back_to_raw_rule_without_reference():
    model = ELOModel(model_path="/nonexistent/xgboost_elo_none.json", pool=None)
    if not model.is_mock:                              # blitz fallback found: skip
        pytest.skip("production model present; mock path not exercised")
    out = model.predict_and_explain({f: 50.0 for f in FEATURE_NAMES}, k=3)
    assert out["weakness_rule"] == "raw-argmin"
    assert len(out["weaknesses"]) == 3
    assert out["margin_quantile"] is None
    assert isinstance(out["predicted_elo"], int)


@pytest.mark.skipif(bsr.load("blitz") is None, reason="no blitz reference")
def test_predict_and_explain_deployed_rule_blitz():
    model = ELOModel(pool="blitz")
    if model.is_mock:
        pytest.skip("no production blitz model")
    df_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_blitz_2k.parquet")
    if not os.path.exists(df_path):
        pytest.skip("no blitz dataset")
    df = pd.read_parquet(df_path)
    row = df[FEATURE_NAMES].head(8).mean().to_dict()
    out = model.predict_and_explain(row, k=3)
    assert out["weakness_rule"] == "band-standardised"
    ref = bsr.load("blitz")
    assert all(w["feature"] in ref["candidates"] for w in out["weaknesses"])
    assert out["weakest_margin"] >= 0
    assert out["margin_quantile"] in (0.10, 0.25, 0.50, 1.0)
    # rounding, not truncation
    raw = float(model.model.predict(pd.DataFrame([row], columns=FEATURE_NAMES))[0])
    assert out["predicted_elo"] == int(round(raw))


def test_margin_quantile_thresholds():
    assert margin_quantile(float("nan")) is None
    assert margin_quantile(None) is None
    assert margin_quantile(1.5) == 0.10
    assert margin_quantile(0.68) == 0.25
    assert margin_quantile(0.39) == 0.50
    assert margin_quantile(0.1) == 1.0


def test_orientation_disputed_features_are_documented():
    # the two features whose declared orientation is not supported by the data
    # must still be in the orientation map (the sensitivity check relies on it)
    assert FEATURE_ORIENTATION["material_volatility"] == 1
    assert FEATURE_ORIENTATION["king_danger"] == 1
