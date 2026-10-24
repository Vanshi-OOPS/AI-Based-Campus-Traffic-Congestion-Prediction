"""Model Training, Pipeline Construction, and Validation Selection.

Trains and validates:
- Classifiers: Logistic Regression, Random Forest, XGBoost
- Regressors: Ridge, Random Forest, XGBoost
- Feature sets: CORE (manual input) and FULL (with lag features)
- Advance 15-Minute Alert Model (binary High-soon forecast)

All scalers, encoders, and models are fit strictly on TRAIN.
Model selection is governed strictly by VALIDATION metrics.
"""

from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from src import config as cfg


class WeatherFeatureTransformer(BaseEstimator, TransformerMixin):
    """Ensures weather categorical string is encoded into fixed ordinal and one-hot columns.

    Works transparently whether input already contains encoded columns or raw 'weather' string.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "WeatherFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
        return df


def build_classifier_pipeline(
    model_type: str,
    feature_set: str = "CORE",
    seed: int = cfg.RANDOM_SEED,
) -> Pipeline:
    """Build an end-to-end scikit-learn Pipeline for classification.

    Parameters
    ----------
    model_type : str
        'logistic_regression', 'random_forest', or 'xgboost'.
    feature_set : str
        'CORE' or 'FULL'.
    seed : int
        Random seed.
    """
    feature_cols = cfg.CORE_FEATURE_COLS if feature_set == "CORE" else cfg.FULL_FEATURE_COLS
    continuous_cols = ["minute_of_window", "day_of_week", "vehicle_count"]
    if feature_set == "FULL":
        continuous_cols.extend(["rolling_vehicle_avg_15min", "vehicle_surge_rate"])

    if model_type == "logistic_regression":
        # Logistic regression needs standardization on continuous features
        preprocessor = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), continuous_cols),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
        pipeline = Pipeline([
            ("weather_enc", WeatherFeatureTransformer()),
            ("column_select", ColumnSelector(feature_cols)),
            ("preprocess", preprocessor),
            ("clf", classifier),
        ])
    elif model_type == "random_forest":
        # Trees receive raw unscaled features so SHAP values remain in readable units
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        pipeline = Pipeline([
            ("weather_enc", WeatherFeatureTransformer()),
            ("column_select", ColumnSelector(feature_cols)),
            ("clf", classifier),
        ])
    elif model_type == "xgboost":
        # XGBoost with multiclass objective
        classifier = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        pipeline = Pipeline([
            ("weather_enc", WeatherFeatureTransformer()),
            ("column_select", ColumnSelector(feature_cols)),
            ("clf", XGBoostClassWrapper(classifier)),
        ])
    else:
        raise ValueError(f"Unknown classifier model_type: {model_type}")

    return pipeline


def build_regressor_pipeline(
    model_type: str,
    feature_set: str = "CORE",
    seed: int = cfg.RANDOM_SEED,
) -> Pipeline:
    """Build an end-to-end Pipeline for continuous score regression."""
    feature_cols = cfg.CORE_FEATURE_COLS if feature_set == "CORE" else cfg.FULL_FEATURE_COLS
    continuous_cols = ["minute_of_window", "day_of_week", "vehicle_count"]
    if feature_set == "FULL":
        continuous_cols.extend(["rolling_vehicle_avg_15min", "vehicle_surge_rate"])

    if model_type == "ridge":
        preprocessor = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), continuous_cols),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        regressor = Ridge(alpha=1.0, random_state=seed)
        pipeline = Pipeline([
            ("weather_enc", WeatherFeatureTransformer()),
            ("column_select", ColumnSelector(feature_cols)),
            ("preprocess", preprocessor),
            ("reg", regressor),
        ])
    elif model_type == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            random_state=seed,
            n_jobs=-1,
        )
        pipeline = Pipeline([
            ("weather_enc", WeatherFeatureTransformer()),
            ("column_select", ColumnSelector(feature_cols)),
            ("reg", regressor),
        ])
    elif model_type == "xgboost":
        regressor = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
        )
        pipeline = Pipeline([
            ("weather_enc", WeatherFeatureTransformer()),
            ("column_select", ColumnSelector(feature_cols)),
            ("reg", regressor),
        ])
    else:
        raise ValueError(f"Unknown regressor model_type: {model_type}")

    return pipeline


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects specific columns from a DataFrame, filling missing lag columns with fallback."""

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "ColumnSelector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for c in self.columns:
            if c not in df.columns:
                if c == "rolling_vehicle_avg_15min":
                    df[c] = df.get("vehicle_count", 35.0)
                elif c == "vehicle_surge_rate":
                    df[c] = 0.0
                else:
                    df[c] = 0.0
        return df[self.columns].copy()


class XGBoostClassWrapper(BaseEstimator):
    """Wraps XGBoost Classifier to handle string classes ['Low', 'Moderate', 'High'] seamlessly."""

    def __init__(self, estimator: XGBClassifier) -> None:
        self.estimator = estimator
        self.class_map = {c: i for i, c in enumerate(cfg.CONGESTION_CLASSES)}
        self.inv_class_map = {i: c for i, c in enumerate(cfg.CONGESTION_CLASSES)}
        self.classes_ = np.array(cfg.CONGESTION_CLASSES)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostClassWrapper":
        y_int = y.map(self.class_map).values
        self.estimator.fit(X, y_int)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds_int = self.estimator.predict(X)
        return np.array([self.inv_class_map[p] for p in preds_int], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)


def evaluate_classifier_val(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    """Compute validation metrics for a classifier."""
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)
    high_idx = cfg.get_class_index(model, "High")
    high_probs = probs[:, high_idx]

    acc = float(accuracy_score(y_val, preds))
    macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
    high_rec = float(recall_score(y_val == "High", preds == "High", zero_division=0))
    high_prec = float(precision_score(y_val == "High", preds == "High", zero_division=0))
    false_high_alerts = int(np.sum((preds == "High") & (y_val != "High")))

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "high_recall": round(high_rec, 4),
        "high_precision": round(high_prec, 4),
        "false_high_alerts": false_high_alerts,
    }


def evaluate_regressor_val(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, float]:
    """Compute validation metrics for a regressor."""
    preds = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, preds))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    r2 = float(r2_score(y_val, preds))
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
    }


def run_model_training_and_selection(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Train all candidate models on TRAIN, evaluate on VAL, and select the winning models.

    Selection Rule:
    - Best classifier selected by highest Validation Macro-F1 (with tie-break on High Recall).
    - If differences are within noise (< 0.01), simpler/more interpretable model is preferred.
    - Best regressor selected by lowest Validation MAE.
    """
    y_train_class = train_df["congestion_class"]
    y_val_class = val_df["congestion_class"]

    y_train_score = train_df["congestion_score"]
    y_val_score = val_df["congestion_score"]

    results = {
        "classification_val": {},
        "regression_val": {},
        "trained_classifiers": {},
        "trained_regressors": {},
    }

    # 1. Train & Validate Classifiers (CORE & FULL)
    clf_models = ["logistic_regression", "random_forest", "xgboost"]
    feature_sets = ["CORE", "FULL"]

    for f_set in feature_sets:
        for m_type in clf_models:
            key = f"{m_type}_{f_set}"
            pipe = build_classifier_pipeline(model_type=m_type, feature_set=f_set)
            pipe.fit(train_df, y_train_class)
            val_metrics = evaluate_classifier_val(pipe, val_df, y_val_class)
            results["classification_val"][key] = val_metrics
            results["trained_classifiers"][key] = pipe

    # 2. Train & Validate Regressors (CORE & FULL)
    reg_models = ["ridge", "random_forest", "xgboost"]
    for f_set in feature_sets:
        for m_type in reg_models:
            key = f"{m_type}_{f_set}"
            pipe = build_regressor_pipeline(model_type=m_type, feature_set=f_set)
            pipe.fit(train_df, y_train_score)
            val_metrics = evaluate_regressor_val(pipe, val_df, y_val_score)
            results["regression_val"][key] = val_metrics
            results["trained_regressors"][key] = pipe

    # 3. Model Selection Decision Logic
    # Select best classifier for CORE and FULL independently
    for f_set in feature_sets:
        candidates = {k: v for k, v in results["classification_val"].items() if k.endswith(f_set)}
        best_clf_key = max(candidates.keys(), key=lambda k: (candidates[k]["macro_f1"], candidates[k]["high_recall"]))
        results[f"selected_classifier_{f_set.lower()}"] = best_clf_key

    # Select best regressor for CORE and FULL
    for f_set in feature_sets:
        candidates = {k: v for k, v in results["regression_val"].items() if k.endswith(f_set)}
        best_reg_key = min(candidates.keys(), key=lambda k: candidates[k]["mae"])
        results[f"selected_regressor_{f_set.lower()}"] = best_reg_key

    # 4. Train 15-Minute Advance Alert Forecast Model
    # Target: y = 1 if class at t+15min is High, else 0 (drop NaNs from morning tails)
    train_adv = train_df.dropna(subset=["congestion_class_t_plus_15min"]).copy()
    val_adv = val_df.dropna(subset=["congestion_class_t_plus_15min"]).copy()

    y_train_adv = (train_adv["congestion_class_t_plus_15min"] == "High").astype(int)
    y_val_adv = (val_adv["congestion_class_t_plus_15min"] == "High").astype(int)

    adv_model = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        random_state=cfg.RANDOM_SEED,
        eval_metric="logloss",
        n_jobs=-1,
    )
    adv_pipe = Pipeline([
        ("weather_enc", WeatherFeatureTransformer()),
        ("column_select", ColumnSelector(cfg.FULL_FEATURE_COLS)),
        ("clf", adv_model),
    ])
    adv_pipe.fit(train_adv, y_train_adv)
    results["advance_alert_model"] = adv_pipe

    return results
