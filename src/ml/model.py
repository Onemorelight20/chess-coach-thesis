import os

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.features.extractor import FEATURE_NAMES, WEAKNESS_CANDIDATES, DISPLAY_NAMES
from src.ml import build_shap_reference


class ELOModel:
    def __init__(self, pool: str = None, model_path: str = None):
        """
        Load the trained XGBoost Elo regressor: pool-specific model first, then
        blitz (largest pool), then a clearly-flagged mock. There is deliberately
        no mixed-pool model, since Elo is not comparable across time controls.
        """
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        candidates = []
        if model_path:
            candidates.append(model_path)
        if pool:
            candidates.append(os.path.join(base, f"xgboost_elo_{pool}.json"))
        candidates.append(os.path.join(base, "xgboost_elo_blitz.json"))

        self.pool = pool
        loaded = next((p for p in candidates
                       if os.path.exists(p) and os.path.exists(p.replace(".json", "_features.pkl"))),
                      None)
        if loaded:
            print(f"Loading trained XGBoost model from {loaded}...")
            self.model = xgb.XGBRegressor()
            self.model.load_model(loaded)
            self.feature_names = joblib.load(loaded.replace(".json", "_features.pkl"))
            self.model_path = loaded
            self.is_mock = False
        else:
            print("WARNING: Trained model not found. Using mock model for demonstration.")
            self._initialize_mock_model()
            self.model_path = None
            self.is_mock = True

        self.explainer = shap.TreeExplainer(self.model)
        self._shap_reference = None if self.is_mock else build_shap_reference.load(pool)

    def _initialize_mock_model(self):
        self.feature_names = list(FEATURE_NAMES)
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.random((300, len(self.feature_names))) * 100,
                         columns=self.feature_names)
        # Plausible direction only; this model is never used for real numbers.
        y = (2200
             - 4.0 * X["acpl"]
             - 3.0 * X["endgame_acpl"]
             - 2.5 * X["middlegame_acpl"]
             - 1.5 * X["mistake_rate"]
             + rng.normal(0, 60, len(X)))
        self.model = xgb.XGBRegressor(n_estimators=60, max_depth=3, random_state=42)
        self.model.fit(X, y)

    def predict_and_explain(self, features: dict, k: int = 3) -> dict:
        """
        Predict Elo and name the training category.

        The category is NOT the raw `argmin` of the SHAP values. Two corrections
        are applied (see build_shap_reference.py and subsection 3.3):
        candidates are limited to features a player can actually train, and each
        contribution is standardised against players of the same predicted-rating
        band, so the answer is "where is this player unusual for their level"
        rather than "which feature has the largest raw contribution". Without the
        reference file the method falls back to the raw rule and says so in
        `weakness_rule`.
        """
        row = {f: features.get(f, np.nan) for f in self.feature_names}
        x_input = pd.DataFrame([row], columns=self.feature_names)

        predicted_elo = int(self.model.predict(x_input)[0])

        shap_values = self.explainer.shap_values(x_input)[0]
        shap_dict = dict(zip(self.feature_names, [float(v) for v in shap_values]))

        ref = self._shap_reference
        if ref is not None:
            cands = [c for c in ref["candidates"] if c in shap_dict]
            mu, sd = build_shap_reference.stats_at(ref, predicted_elo)
            scored = {c: (shap_dict[c] - float(mu[c])) / float(sd[c])
                      for c in cands if pd.notna(sd[c])}
            rule = "band-standardised"
        else:
            scored = {c: shap_dict[c] for c in WEAKNESS_CANDIDATES if c in shap_dict}
            rule = "raw-argmin"

        # Report the k weakest, not one. Naming a single category throws away a
        # reproducible answer: on split halves of the same player the top-1 pick
        # agrees only 11% of the time, while the top-2 sets overlap 33% and the
        # top-3 sets 57% -- the halves rank the same problem areas, they just
        # reorder neighbours. Agreement with the deviation-based reference rises
        # the same way (0.28 -> 0.41 -> 0.49). See subsection 3.3.
        ranked = sorted(scored, key=scored.get)[:max(1, k)]
        weaknesses = [{"feature": f,
                       "display": DISPLAY_NAMES.get(f, f),
                       "impact": shap_dict[f],
                       "score": float(scored[f])} for f in ranked]
        top = weaknesses[0]

        return {
            "predicted_elo": predicted_elo,
            "shap_values": shap_dict,
            "weaknesses": weaknesses,
            "weakest_feature": top["feature"],
            "weakest_display": top["display"],
            "weakest_impact": top["impact"],
            "weakest_score": top["score"],
            "weakness_rule": rule,
            "is_mock": self.is_mock,
        }
