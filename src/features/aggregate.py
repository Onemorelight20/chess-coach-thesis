"""
Aggregate a player's per-game feature vectors into one vector for multi-game Elo
prediction.

Mean is the deployed default: on blitz (aggregation_methods.py, N=8) it beats the
median (R2 0.889 vs 0.879) and ties trimmed20; mean+std scores 0.896 but doubles
the vector and would require training on aggregated rows.
"""

import warnings

import numpy as np

from src.features.extractor import FEATURE_NAMES

AGG_METHODS = ("mean", "median", "trimmed20")


def aggregate_features(feature_dicts, method: str = "mean") -> dict:
    """Collapse N per-game feature dicts into one aggregated feature dict."""
    if not feature_dicts:
        return {f: np.nan for f in FEATURE_NAMES}
    X = np.array([[fd.get(f, np.nan) for f in FEATURE_NAMES] for fd in feature_dicts],
                 dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column -> NaN, handled by XGBoost
        if method == "mean":
            v = np.nanmean(X, axis=0)
        elif method == "median":
            v = np.nanmedian(X, axis=0)
        elif method == "trimmed20":
            from scipy.stats import trim_mean
            v = np.array([trim_mean(c[~np.isnan(c)], 0.2) if np.any(~np.isnan(c)) else np.nan
                          for c in X.T])
        else:
            raise ValueError(f"unknown aggregation method: {method}")
    return {f: float(val) for f, val in zip(FEATURE_NAMES, v)}
