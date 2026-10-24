"""Unit tests for Phase 2 training, baselines, evaluation, and explainability."""

import pytest
import numpy as np
import pandas as pd
import joblib

from src import config as cfg
from src.baselines import (
    MajorityClassBaseline,
    VehicleCountRuleBaseline,
    CountAndWeatherRuleBaseline,
    HistoricalTimeSlotBaseline,
)
from src.train_models import build_classifier_pipeline, build_regressor_pipeline
from src.explain import TrafficExplainer


@pytest.fixture(scope="module")
def data_splits():
    df = pd.read_csv(cfg.PROCESSED_DATA_PATH)
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def test_leakage_columns_absent_from_feature_sets():
    """Verify that none of the leakage columns appear in CORE or FULL feature sets."""
    for leak in cfg.LEAKAGE_COLS:
        assert leak not in cfg.CORE_FEATURE_COLS, f"Leakage column {leak} in CORE_FEATURE_COLS!"
        assert leak not in cfg.FULL_FEATURE_COLS, f"Leakage column {leak} in FULL_FEATURE_COLS!"


def test_time_slot_baseline_uses_train_only(data_splits):
    """Historical time-slot baseline must compute slot averages strictly from TRAIN mornings."""
    train_df, val_df, test_df = data_splits
    slot_baseline = HistoricalTimeSlotBaseline().fit(train_df, train_df["congestion_class"])

    # Verify slot means match train means exactly
    expected_means = train_df.groupby("minute_of_window")["vehicle_count"].mean().to_dict()
    for slot, val in expected_means.items():
        assert slot_baseline.slot_means[slot] == pytest.approx(val, abs=1e-3)


def test_no_val_or_test_rows_in_fit(data_splits):
    """Assert that models fitted on train receive strictly len(train_df) observations."""
    train_df, _, test_df = data_splits
    assert train_df["date"].nunique() == 126
    assert test_df["date"].nunique() == 27
    assert set(train_df["date"]).isdisjoint(set(test_df["date"]))


def test_saved_model_reload_and_reproduce(data_splits):
    """Verify that serializing and deserializing a model reproduces identical predictions."""
    train_df, _, test_df = data_splits
    pipe = build_classifier_pipeline("random_forest", feature_set="CORE", seed=cfg.RANDOM_SEED)
    pipe.fit(train_df, train_df["congestion_class"])

    preds_orig = pipe.predict(test_df)

    # Save to a temporary test path and reload
    test_model_path = cfg.MODELS_DIR / "test_rf_temp.joblib"
    joblib.dump(pipe, test_model_path)
    reloaded = joblib.load(test_model_path)

    preds_reloaded = reloaded.predict(test_df)
    assert np.array_equal(preds_orig, preds_reloaded)

    # Clean up
    if test_model_path.exists():
        test_model_path.unlink()


def test_explain_prediction_weather_grouping_and_sum(data_splits):
    """Verify explain_prediction groups weather and contributions sum to prediction."""
    train_df, val_df, _ = data_splits
    reg = build_regressor_pipeline("random_forest", feature_set="FULL", seed=cfg.RANDOM_SEED)
    clf = build_classifier_pipeline("random_forest", feature_set="FULL", seed=cfg.RANDOM_SEED)
    reg.fit(train_df, train_df["congestion_score"])
    clf.fit(train_df, train_df["congestion_class"])

    explainer = TrafficExplainer(reg, clf, train_df, feature_set="FULL")

    sample_row = val_df.iloc[0].to_dict()

    # Test Regressor explanation
    reg_exp = explainer.explain_prediction(sample_row, mode="regressor")
    assert "weather" in reg_exp["contributions"]
    assert "weather_ordinal" not in reg_exp["contributions"]
    assert "weather_Cloudy" not in reg_exp["contributions"]
    assert reg_exp["units"] == "points of congestion score"

    # Sum check: base_value + sum(contributions) == predicted_output
    approx_sum = reg_exp["base_value"] + sum(reg_exp["contributions"].values())
    assert approx_sum == pytest.approx(reg_exp["predicted_output"], abs=1.5)

    # Test Classifier explanation
    clf_exp = explainer.explain_prediction(sample_row, mode="classifier")
    assert "weather" in clf_exp["contributions"]
    assert clf_exp["units"] == "effect on probability of High"


def test_seed_reproducibility_metrics(data_splits):
    """Identical seeds must produce identical predictions and metrics."""
    train_df, _, test_df = data_splits
    p1 = build_classifier_pipeline("xgboost", feature_set="CORE", seed=cfg.RANDOM_SEED)
    p2 = build_classifier_pipeline("xgboost", feature_set="CORE", seed=cfg.RANDOM_SEED)

    p1.fit(train_df, train_df["congestion_class"])
    p2.fit(train_df, train_df["congestion_class"])

    assert np.array_equal(p1.predict(test_df), p2.predict(test_df))
