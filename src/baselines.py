"""Baseline Models for Campus Traffic Congestion Prediction.

All baselines are fitted/tuned strictly on TRAIN (and optionally validated on VAL).
They never access TEST set observations during fitting.

Implements:
1. MajorityClassBaseline: Always predicts the most frequent class in TRAIN.
2. VehicleCountRuleBaseline: Threshold cutoffs on vehicle_count tuned via grid search on TRAIN.
3. CountAndWeatherRuleBaseline: Threshold cutoffs conditioned on weather category.
4. HistoricalTimeSlotBaseline: Mean vehicle count per minute_of_window from TRAIN, mapped to class.
"""

from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import config as cfg


class MajorityClassBaseline:
    """Predicts the most frequent class observed in the training set."""

    def __init__(self) -> None:
        self.majority_class: str = "Low"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MajorityClassBaseline":
        self.majority_class = str(y.mode().iloc[0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class, dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Probability 1.0 for majority class, 0.0 for others
        probs = np.zeros((len(X), len(cfg.CONGESTION_CLASSES)))
        idx = cfg.CONGESTION_CLASSES.index(self.majority_class)
        probs[:, idx] = 1.0
        return probs


class VehicleCountRuleBaseline:
    """Classifies congestion based on two thresholds on vehicle_count:

    Low if count < T_low
    Moderate if T_low <= count < T_high
    High if count >= T_high

    Tuned via grid search on TRAIN to maximize macro-F1.
    """

    def __init__(self, t_low: float = 45.0, t_high: float = 75.0) -> None:
        self.t_low = t_low
        self.t_high = t_high

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VehicleCountRuleBaseline":
        counts = X["vehicle_count"].values
        y_arr = y.values

        best_f1 = -1.0
        best_t_low = 45.0
        best_t_high = 75.0

        # Search grid around realistic count ranges
        t_low_candidates = np.arange(30, 65, 2)
        t_high_candidates = np.arange(65, 105, 2)

        for tl in t_low_candidates:
            for th in t_high_candidates:
                if tl >= th:
                    continue
                preds = np.where(counts < tl, "Low", np.where(counts < th, "Moderate", "High"))
                score = f1_score(y_arr, preds, average="macro", zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t_low = float(tl)
                    best_t_high = float(th)

        self.t_low = best_t_low
        self.t_high = best_t_high
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        counts = X["vehicle_count"].values
        return np.where(
            counts < self.t_low,
            "Low",
            np.where(counts < self.t_high, "Moderate", "High")
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        probs = np.zeros((len(X), len(cfg.CONGESTION_CLASSES)))
        for i, c in enumerate(cfg.CONGESTION_CLASSES):
            probs[preds == c, i] = 1.0
        return probs


class CountAndWeatherRuleBaseline:
    """Tuned vehicle count thresholds per weather condition.

    Accounts for reduced capacity during rain by lowering the thresholds for Moderate/High.
    Fitted strictly on TRAIN.
    """

    def __init__(self) -> None:
        self.weather_thresholds: Dict[str, Tuple[float, float]] = {
            w: (45.0, 75.0) for w in cfg.WEATHER_CATEGORIES
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CountAndWeatherRuleBaseline":
        df = X.copy()
        df["target"] = y.values

        for w in cfg.WEATHER_CATEGORIES:
            sub = df[df["weather"] == w]
            if len(sub) < 10 or sub["target"].nunique() < 2:
                # Fallback to general vehicle rule
                gen_rule = VehicleCountRuleBaseline().fit(X, y)
                self.weather_thresholds[w] = (gen_rule.t_low, gen_rule.t_high)
                continue

            rule = VehicleCountRuleBaseline().fit(sub[["vehicle_count"]], sub["target"])
            self.weather_thresholds[w] = (rule.t_low, rule.t_high)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.empty(len(X), dtype=object)
        weathers = X["weather"].values
        counts = X["vehicle_count"].values

        for w in cfg.WEATHER_CATEGORIES:
            mask = (weathers == w)
            if not np.any(mask):
                continue
            tl, th = self.weather_thresholds.get(w, (45.0, 75.0))
            sub_counts = counts[mask]
            preds[mask] = np.where(
                sub_counts < tl,
                "Low",
                np.where(sub_counts < th, "Moderate", "High")
            )
        # Any unmapped rows fallback to general cutoff
        unmapped = pd.isna(preds) | (preds == None)
        if np.any(unmapped):
            preds[unmapped] = np.where(counts[unmapped] < 45.0, "Low", np.where(counts[unmapped] < 75.0, "Moderate", "High"))
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        probs = np.zeros((len(X), len(cfg.CONGESTION_CLASSES)))
        for i, c in enumerate(cfg.CONGESTION_CLASSES):
            probs[preds == c, i] = 1.0
        return probs


class HistoricalTimeSlotBaseline:
    """Per-minute_of_window average count computed from TRAIN mornings only,

    mapped to class using the tuned vehicle count thresholds.
    Never uses val or test observations.
    """

    def __init__(self) -> None:
        self.slot_means: Dict[int, float] = {}
        self.rule: VehicleCountRuleBaseline = VehicleCountRuleBaseline()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalTimeSlotBaseline":
        # Fit rule on train counts
        self.rule.fit(X, y)
        # Compute slot historical means
        slot_grp = X.groupby("minute_of_window")["vehicle_count"].mean()
        self.slot_means = slot_grp.to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        hist_counts = X["minute_of_window"].map(self.slot_means).fillna(35.0).values
        dummy_df = pd.DataFrame({"vehicle_count": hist_counts})
        return self.rule.predict(dummy_df)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        probs = np.zeros((len(X), len(cfg.CONGESTION_CLASSES)))
        for i, c in enumerate(cfg.CONGESTION_CLASSES):
            probs[preds == c, i] = 1.0
        return probs
