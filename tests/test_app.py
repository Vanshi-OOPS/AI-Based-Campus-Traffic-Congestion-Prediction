"""Unit and Integration Tests for Phase 3 Streamlit Dashboard and Components.

Tests:
1. Artifact loader integrity and error handling (missing files, schema checks)
2. Central operational alert logic (p(High) >= cutoff rule, badge colors, diversion routes, boundary disagreements)
3. SHAP factor chart explanations (grouping weather features, points of score & prob units)
4. Timeline data generation and lag continuity
5. Headless dashboard script execution without exceptions
"""

import sys
import subprocess
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src import config as cfg
from app.components.artifacts import load_all_artifacts, ArtifactValidationError
from app.components.alert_logic import evaluate_operational_alert
from app.components.factor_chart import render_regressor_factor_chart, render_classifier_probability_chart
from src.features import compute_morning_lag_features


def test_artifacts_load_and_validate():
    """Verify that all Phase 2 serialized artifacts load and conform to the schema."""
    artifacts = load_all_artifacts()

    assert "feature_meta" in artifacts
    assert "eval_metrics" in artifacts
    assert "run_metadata" in artifacts
    assert "models" in artifacts
    assert "alert_threshold" in artifacts

    # Check model dictionary
    models = artifacts["models"]
    for required_key in ["clf_core", "clf_full", "reg_core", "reg_full", "explainer"]:
        assert required_key in models, f"Missing required model key: {required_key}"
        assert models[required_key] is not None

    # Check alert threshold
    threshold = artifacts["alert_threshold"]
    assert isinstance(threshold, float)
    assert 0.0 < threshold < 1.0

    # Check feature schemas
    assert artifacts["feature_meta"]["core_features"] == cfg.CORE_FEATURE_COLS
    assert artifacts["feature_meta"]["full_features"] == cfg.FULL_FEATURE_COLS


def test_artifact_validation_error_on_missing_dir(monkeypatch, tmp_path):
    """Verify ArtifactValidationError is raised when model artifacts are missing."""
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "nonexistent_models")
    # Clear streamlit cache to test cold load
    load_all_artifacts.clear()

    with pytest.raises(ArtifactValidationError):
        load_all_artifacts()

    # Reset cache after test
    monkeypatch.undo()
    load_all_artifacts.clear()


def test_alert_logic_standard_levels():
    """Verify standard alert mapping for Low, Moderate, and High scenarios."""
    cutoff = 0.732

    # Case 1: Low congestion
    res_low = evaluate_operational_alert(
        predicted_class="Low",
        congestion_score=28.5,
        high_probability=0.08,
        alert_threshold=cutoff,
    )
    assert res_low["alert_level"] == "Low"
    assert res_low["banner_type"] == "success"
    assert res_low["diversion_route_note"] is None
    assert res_low["disagreement_note"] is None

    # Case 2: Moderate congestion
    res_mod = evaluate_operational_alert(
        predicted_class="Moderate",
        congestion_score=52.0,
        high_probability=0.25,
        alert_threshold=cutoff,
    )
    assert res_mod["alert_level"] == "Moderate"
    assert res_mod["banner_type"] == "warning"
    assert res_mod["diversion_route_note"] is None
    assert res_mod["disagreement_note"] is None

    # Case 3: High congestion triggered by probability
    res_high = evaluate_operational_alert(
        predicted_class="High",
        congestion_score=82.0,
        high_probability=0.88,
        alert_threshold=cutoff,
    )
    assert res_high["alert_level"] == "High"
    assert res_high["banner_type"] == "error"
    assert cfg.ALERT_MSG_HIGH in res_high["banner_message"]
    assert "Suggested diversion route:" in res_high["diversion_route_note"]
    assert "confirm with the college administration" in res_high["diversion_route_note"]


def test_alert_logic_cutoff_boundaries():
    """Verify behavior at exact threshold cutoff boundary."""
    cutoff = 0.732

    # Exactly at cutoff -> must trigger High alert
    res_at_cutoff = evaluate_operational_alert(
        predicted_class="Moderate",
        congestion_score=68.0,
        high_probability=0.732,
        alert_threshold=cutoff,
    )
    assert res_at_cutoff["alert_level"] == "High"
    assert res_at_cutoff["banner_type"] == "error"

    # Just below cutoff with Moderate class -> does not trigger High
    res_below_cutoff = evaluate_operational_alert(
        predicted_class="Moderate",
        congestion_score=66.0,
        high_probability=0.7319,
        alert_threshold=cutoff,
    )
    assert res_below_cutoff["alert_level"] == "Moderate"
    assert res_below_cutoff["banner_type"] == "warning"


def test_alert_logic_disagreement_detection():
    """Verify notes are surfaced when regressor continuous score and classifier disagree."""
    cutoff = 0.732

    # Continuous score is 72.0 (High threshold), but classifier predicts Moderate
    res_disagree = evaluate_operational_alert(
        predicted_class="Moderate",
        congestion_score=72.0,
        high_probability=0.55,
        alert_threshold=cutoff,
    )
    assert res_disagree["disagreement_note"] is not None
    assert "proximity to the decision boundary" in res_disagree["disagreement_note"]
    assert "72.0" in res_disagree["disagreement_note"]
    assert "Moderate" in res_disagree["disagreement_note"]

    # Continuous score is 38.0 (Low threshold), but classifier predicts Moderate
    res_disagree_low = evaluate_operational_alert(
        predicted_class="Moderate",
        congestion_score=38.0,
        high_probability=0.15,
        alert_threshold=cutoff,
    )
    assert res_disagree_low["disagreement_note"] is not None
    assert "38.0" in res_disagree_low["disagreement_note"]


def test_factor_chart_shap_explanation():
    """Verify TrafficExplainer returns properly formatted factors with weather grouped."""
    artifacts = load_all_artifacts()
    explainer = artifacts["models"]["explainer"]

    sample_features = {
        "minute_of_window": 45,
        "day_of_week": 1,
        "vehicle_count": 95,
        "weather": "Rain",
        "event_flag": 0,
        "rolling_vehicle_avg_15min": 85.0,
        "vehicle_surge_rate": 10.0,
    }

    # Test Regressor explanation
    reg_exp = explainer.explain_prediction(sample_features, mode="regressor")
    assert "weather" in reg_exp["contributions"]
    assert "weather_ordinal" not in reg_exp["contributions"]
    assert "weather_Rain" not in reg_exp["contributions"]
    assert reg_exp["units"] == "points of congestion score"

    # Additivity test: base_value + sum(contributions) == predicted_output
    approx_score = reg_exp["base_value"] + sum(reg_exp["contributions"].values())
    assert approx_score == pytest.approx(reg_exp["predicted_output"], abs=1.5)

    # Test Classifier explanation
    clf_exp = explainer.explain_prediction(sample_features, mode="classifier")
    assert "weather" in clf_exp["contributions"]
    assert clf_exp["units"] == "effect on probability of High"

    # Test Figure rendering
    fig_reg = render_regressor_factor_chart(reg_exp)
    assert fig_reg is not None
    import matplotlib.pyplot as plt
    plt.close(fig_reg)

    fig_clf = render_classifier_probability_chart(clf_exp)
    assert fig_clf is not None
    plt.close(fig_clf)


def test_timeline_lag_continuity():
    """Verify that simulated timeline computes lag features with proper first-row fallback."""
    # Synthesize a 5-interval morning
    df_morning = pd.DataFrame({
        "minute_of_window": [0, 5, 10, 15, 20],
        "vehicle_count": [40, 50, 60, 70, 80],
    })
    lags_df = compute_morning_lag_features(df_morning)

    # At t=0 (first interval): surge is 0, rolling average falls back to vehicle_count
    assert lags_df.loc[0, "vehicle_surge_rate"] == 0.0
    assert lags_df.loc[0, "rolling_vehicle_avg_15min"] == 40.0

    # At t=1: rolling avg is mean(count[0]) = 40.0, surge is 50 - 40 = 10.0
    assert lags_df.loc[1, "vehicle_surge_rate"] == 10.0
    assert lags_df.loc[1, "rolling_vehicle_avg_15min"] == 40.0

    # At t=3: rolling avg uses up to 3 previous intervals [0, 1, 2] -> mean(40, 50, 60) = 50.0
    assert lags_df.loc[3, "rolling_vehicle_avg_15min"] == pytest.approx(50.0)


def test_headless_dashboard_execution():
    """Verify that running the main Streamlit application script exits cleanly with code 0."""
    import os
    env = os.environ.copy()
    env["STREAMLIT_TESTING"] = "1"
    res = subprocess.run(
        [sys.executable, "app/main.py"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, f"App execution failed with stderr:\n{res.stderr}"
