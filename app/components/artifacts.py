"""Artifact loading and validation utilities for Streamlit dashboard.

Loads pre-trained models, feature schemas, evaluation metrics, and run metadata
from models/ using st.cache_resource. Verifies schema integrity on startup.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
import joblib
import streamlit as st

from src import config as cfg


class ArtifactValidationError(Exception):
    """Raised when an artifact fails schema or existence checks."""
    pass


@st.cache_resource
def load_all_artifacts() -> Dict[str, Any]:
    """Load and validate all Phase 2 serialized models, metadata, and evaluation metrics."""
    models_dir = cfg.MODELS_DIR
    if not models_dir.exists():
        raise ArtifactValidationError(f"Models directory not found at {models_dir}. Please run train.py first.")

    # 1. Load Metadata & Schemas
    feat_meta_path = models_dir / "feature_columns.json"
    eval_metrics_path = models_dir / "eval_metrics.json"
    run_meta_path = models_dir / "run_metadata.json"

    for path, name in [
        (feat_meta_path, "feature_columns.json"),
        (eval_metrics_path, "eval_metrics.json"),
        (run_meta_path, "run_metadata.json"),
    ]:
        if not path.exists():
            raise ArtifactValidationError(f"Missing required artifact: {name}. Run train.py first.")

    with open(feat_meta_path, "r") as f:
        feature_meta = json.load(f)

    with open(eval_metrics_path, "r") as f:
        eval_metrics = json.load(f)

    with open(run_meta_path, "r") as f:
        run_metadata = json.load(f)

    # 2. Load Model Pipelines
    required_models = {
        "clf_core": "congestion_classifier.joblib",
        "clf_full": "congestion_classifier_full.joblib",
        "reg_core": "congestion_regressor.joblib",
        "reg_full": "congestion_regressor_full.joblib",
        "explainer": "traffic_explainer.joblib",
    }

    loaded_models = {}
    for key, filename in required_models.items():
        m_path = models_dir / filename
        if not m_path.exists():
            raise ArtifactValidationError(f"Missing model binary: {filename}. Run train.py first.")
        loaded_models[key] = joblib.load(m_path)

    # Optional Advance Alert Model
    adv_path = models_dir / "advance_alert_model.joblib"
    loaded_models["adv_model"] = joblib.load(adv_path) if adv_path.exists() else None

    # 3. Schema & Feature Column Validation
    expected_core = feature_meta.get("core_features", [])
    expected_full = feature_meta.get("full_features", [])

    if expected_core != cfg.CORE_FEATURE_COLS:
        raise ArtifactValidationError(
            f"CORE features mismatch! Config: {cfg.CORE_FEATURE_COLS} vs Artifact: {expected_core}"
        )
    if expected_full != cfg.FULL_FEATURE_COLS:
        raise ArtifactValidationError(
            f"FULL features mismatch! Config: {cfg.FULL_FEATURE_COLS} vs Artifact: {expected_full}"
        )

    # Validation-chosen High-alert threshold
    alert_threshold = feature_meta.get("alert_threshold", 0.50)

    return {
        "feature_meta": feature_meta,
        "eval_metrics": eval_metrics,
        "run_metadata": run_metadata,
        "models": loaded_models,
        "alert_threshold": float(alert_threshold),
    }
