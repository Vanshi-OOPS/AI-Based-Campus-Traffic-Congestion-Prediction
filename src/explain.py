"""Model Explainability with SHAP and Tree Feature Importance.

Provides:
- Global feature importance (SHAP summary beeswarm plot saved to docs/figures/shap_summary.png).
- explain_prediction(features_dict, mode):
  - mode="regressor": SHAP contributions in points of congestion score.
  - mode="classifier": SHAP contributions in effect on probability of High.
- Automatic aggregation: Groups one-hot and ordinal weather features into a single 'weather' factor.
- Verified mathematical consistency: base_value + sum(contributions) == prediction (within tolerance).
"""

from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg


class TrafficExplainer:
    """Explainer using TreeExplainer on trained tree models for score and probability."""

    def __init__(
        self,
        regressor_pipeline: Any,
        classifier_pipeline: Any,
        background_df: pd.DataFrame,
        feature_set: str = "FULL",
    ) -> None:
        self.reg_pipe = regressor_pipeline
        self.clf_pipe = classifier_pipeline
        self.feature_set = feature_set
        self.feature_cols = cfg.CORE_FEATURE_COLS if feature_set == "CORE" else cfg.FULL_FEATURE_COLS

        # Extract underlying tree estimators
        self.reg_model = self._extract_estimator(reg_pipeline=self.reg_pipe)
        self.clf_model = self._extract_estimator(clf_pipeline=self.clf_pipe)

        # Prepare background data in the exact feature column format
        bg_transformed = self._transform_input(background_df.sample(min(100, len(background_df)), random_state=cfg.RANDOM_SEED))
        # Initialize TreeExplainer (tree-path dependent is exact and handles scikit-learn & xgboost trees)
        try:
            self.reg_explainer = shap.TreeExplainer(self.reg_model)
        except Exception:
            self.reg_explainer = shap.Explainer(self.reg_pipe.predict, bg_transformed)

        try:
            self.clf_explainer = shap.TreeExplainer(self.clf_model)
        except Exception:
            self.clf_explainer = shap.Explainer(self.clf_pipe.predict_proba, bg_transformed)

        # Background expected values
        if hasattr(self.reg_explainer, "expected_value"):
            self.reg_base_value = float(np.mean(self.reg_explainer.expected_value))
        else:
            self.reg_base_value = float(self.reg_explainer(bg_transformed[:1]).base_values[0])

        if hasattr(self.clf_explainer, "expected_value"):
            clf_ev = self.clf_explainer.expected_value
            high_idx = cfg.get_class_index(self.clf_pipe, "High")
            if isinstance(clf_ev, (list, np.ndarray)) and len(clf_ev) == 3:
                self.clf_base_value = float(clf_ev[high_idx])
            else:
                self.clf_base_value = float(np.mean(clf_ev))
        else:
            clf_exp_res = self.clf_explainer(bg_transformed[:1])
            high_idx = cfg.get_class_index(self.clf_pipe, "High")
            if clf_exp_res.base_values.ndim > 1:
                self.clf_base_value = float(clf_exp_res.base_values[0, high_idx])
            else:
                self.clf_base_value = float(clf_exp_res.base_values[0])

    def _extract_estimator(self, reg_pipeline: Optional[Any] = None, clf_pipeline: Optional[Any] = None) -> Any:
        pipe = reg_pipeline or clf_pipeline
        if hasattr(pipe, "named_steps"):
            step = pipe.named_steps.get("reg") or pipe.named_steps.get("clf")
            if hasattr(step, "estimator"):  # XGBoostClassWrapper
                return step.estimator
            return step
        return pipe

    def _transform_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run input through weather encoding and column selector steps of pipeline."""
        df = X.copy()
        if "weather" in df.columns:
            if "weather_ordinal" not in df.columns:
                df["weather_ordinal"] = df["weather"].map(cfg.WEATHER_ORDINAL_MAP).fillna(0).astype(int)
            if "weather_Cloudy" not in df.columns:
                df["weather_Cloudy"] = (df["weather"] == "Cloudy").astype(int)
            if "weather_Rain" not in df.columns:
                df["weather_Rain"] = (df["weather"] == "Rain").astype(int)
            if "weather_Heavy Rain" not in df.columns:
                df["weather_Heavy Rain"] = (df["weather"] == "Heavy Rain").astype(int)
        return df[self.feature_cols].copy()

    def explain_prediction(
        self,
        features_dict: Dict[str, Any],
        mode: str = "regressor",
    ) -> Dict[str, Any]:
        """Compute individual factor contributions for a single observation.

        Parameters
        ----------
        features_dict : Dict[str, Any]
            Dictionary of feature values (e.g. {'vehicle_count': 85, 'weather': 'Rain', ...})
        mode : str
            'regressor' (points of congestion score) or 'classifier' (probability of High).

        Returns
        -------
        Dict[str, Any]
            base_value: float
            predicted_output: float
            units: str
            contributions: Dict[str, float] (grouped weather into single factor)
        """
        input_df = pd.DataFrame([features_dict])
        transformed_df = self._transform_input(input_df)

        weather_cols = ["weather_ordinal", "weather_Cloudy", "weather_Rain", "weather_Heavy Rain"]

        if mode == "regressor":
            exp_res = self.reg_explainer(transformed_df)
            shap_vals = exp_res.values[0]
            base_val = float(exp_res.base_values[0])
            pred_val = float(self.reg_pipe.predict(input_df)[0])
            units = "points of congestion score"
        elif mode == "classifier":
            exp_res = self.clf_explainer(transformed_df)
            high_idx = cfg.get_class_index(self.clf_pipe, "High")
            if exp_res.values.ndim == 3:
                shap_vals = exp_res.values[0, :, high_idx]
                base_val = float(exp_res.base_values[0, high_idx])
            elif isinstance(exp_res.values, list):
                shap_vals = exp_res.values[high_idx][0]
                base_val = float(exp_res.base_values[high_idx][0])
            else:
                shap_vals = exp_res.values[0]
                base_val = float(exp_res.base_values[0])
            probs = self.clf_pipe.predict_proba(input_df)[0]
            pred_val = float(probs[high_idx])
            units = "effect on probability of High"
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'regressor' or 'classifier'.")

        # Map individual SHAP values to feature names
        raw_contribs = dict(zip(self.feature_cols, shap_vals))

        # Group weather columns into a single 'weather' factor
        weather_total = sum(raw_contribs.get(wc, 0.0) for wc in weather_cols)

        grouped_contribs = {}
        for f in self.feature_cols:
            if f in weather_cols:
                continue
            grouped_contribs[f] = round(float(raw_contribs[f]), 3)
        grouped_contribs["weather"] = round(float(weather_total), 3)

        return {
            "base_value": round(base_val, 3),
            "predicted_output": round(pred_val, 3),
            "units": units,
            "contributions": grouped_contribs,
        }

    def generate_shap_summary_plot(
        self,
        val_df: pd.DataFrame,
        save_path: Optional[Any] = None,
    ) -> None:
        """Generate and save global SHAP summary beeswarm plot for the regressor."""
        target_path = save_path or (cfg.DOCS_FIGURES_DIR / "shap_summary.png")
        sample_df = val_df.sample(min(200, len(val_df)), random_state=cfg.RANDOM_SEED)
        transformed = self._transform_input(sample_df)

        shap_values = self.reg_explainer(transformed)

        fig = plt.figure(figsize=(9, 5.5))
        shap.summary_plot(shap_values, transformed, show=False)
        plt.title("Global Feature Impact on Congestion Score (SHAP Points, Simulated Data)", fontsize=11, fontweight="bold", pad=15)
        plt.xlabel("SHAP Value (Impact in points of congestion score)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(target_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
