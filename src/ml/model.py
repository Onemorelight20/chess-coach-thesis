import os

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.features.extractor import FEATURE_NAMES, WEAKNESS_CANDIDATES, DISPLAY_NAMES


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

    def predict_and_explain(self, features: dict) -> dict:
        """
        Predict Elo and use SHAP to identify the weakest category: the candidate
        feature whose SHAP value pulls the predicted Elo down the most.
        """
        row = {f: features.get(f, np.nan) for f in self.feature_names}
        x_input = pd.DataFrame([row], columns=self.feature_names)

        predicted_elo = int(self.model.predict(x_input)[0])

        shap_values = self.explainer.shap_values(x_input)[0]
        shap_dict = dict(zip(self.feature_names, [float(v) for v in shap_values]))

        candidates = [c for c in WEAKNESS_CANDIDATES if c in shap_dict]
        weakest_feature = min(candidates, key=lambda c: shap_dict[c])

        return {
            "predicted_elo": predicted_elo,
            "shap_values": shap_dict,
            "weakest_feature": weakest_feature,
            "weakest_display": DISPLAY_NAMES.get(weakest_feature, weakest_feature),
            "weakest_impact": shap_dict[weakest_feature],
            "is_mock": self.is_mock,
        }
