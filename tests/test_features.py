"""Unit tests for feature engineering pipeline (src/features.py)."""

import pytest
import pandas as pd
import numpy as np

from src import config as cfg
from src.simulate_data import simulate_traffic_dataset
from src.features import build_featured_dataset


@pytest.fixture(scope="module")
def processed_data():
    """Build processed dataset for a 14-day sample."""
    raw_df, _ = simulate_traffic_dataset(num_days=14, seed=cfg.RANDOM_SEED)
    proc_df = build_featured_dataset(raw_df)
    return proc_df


def test_first_interval_rule(processed_data):
    """Documented first-interval rule: at t=0, surge=0, rolling avg falls back to current count."""
    first_rows = processed_data[processed_data["minute_of_window"] == 0]
    assert (first_rows["vehicle_surge_rate"] == 0.0).all()
    assert (first_rows["rolling_vehicle_avg_15min"] == first_rows["vehicle_count"]).all()


def test_lag_features_math(processed_data):
    """Verify rolling average excludes current row and surge rate equals diff."""
    # Test on a specific morning
    first_date = processed_data["date"].iloc[0]
    morning = processed_data[processed_data["date"] == first_date].sort_values("minute_of_window")
    counts = morning["vehicle_count"].values

    # At t=1 (minute 5): rolling avg must be counts[0]
    assert morning["rolling_vehicle_avg_15min"].iloc[1] == counts[0]
    assert morning["vehicle_surge_rate"].iloc[1] == counts[1] - counts[0]

    # At t=2 (minute 10): rolling avg must be mean(counts[0:2])
    assert morning["rolling_vehicle_avg_15min"].iloc[2] == pytest.approx(np.mean(counts[0:2]), abs=1e-2)
    assert morning["vehicle_surge_rate"].iloc[2] == counts[2] - counts[1]

    # At t=3 (minute 15): rolling avg must be mean(counts[0:3])
    assert morning["rolling_vehicle_avg_15min"].iloc[3] == pytest.approx(np.mean(counts[0:3]), abs=1e-2)
    assert morning["vehicle_surge_rate"].iloc[3] == counts[3] - counts[2]

    # At t=5 (minute 25): rolling avg must be mean(counts[2:5])
    assert morning["rolling_vehicle_avg_15min"].iloc[5] == pytest.approx(np.mean(counts[2:5]), abs=1e-2)
    assert morning["vehicle_surge_rate"].iloc[5] == counts[5] - counts[4]


def test_no_leakage_columns_in_model_features():
    """Verify that no leakage column is included in the model feature list."""
    for leak in cfg.LEAKAGE_COLS:
        assert leak not in cfg.MODEL_FEATURE_COLS, f"Leakage column '{leak}' in MODEL_FEATURE_COLS!"


def test_forecast_target_shift_and_tail_nan(processed_data):
    """Forecast target must be shifted by 3 steps, with the last 3 rows of each morning being NaN."""
    for _, group in processed_data.groupby("date"):
        # Last 3 rows (indices 22, 23, 24) must be NaN
        tail = group["congestion_class_t_plus_15min"].iloc[-3:]
        assert tail.isna().all()

        # Earlier rows must equal the class 3 steps ahead
        for i in range(len(group) - 3):
            actual_future = group["congestion_class"].iloc[i + 3]
            target_val = group["congestion_class_t_plus_15min"].iloc[i]
            assert target_val == actual_future


def test_weather_encoding_consistency(processed_data):
    """One-hot and ordinal weather encodings must match categorical weather exactly."""
    for _, row in processed_data.iterrows():
        w = row["weather"]
        assert row["weather_ordinal"] == cfg.WEATHER_ORDINAL_MAP[w]
        assert row["weather_Cloudy"] == (1 if w == "Cloudy" else 0)
        assert row["weather_Rain"] == (1 if w == "Rain" else 0)
        assert row["weather_Heavy Rain"] == (1 if w == "Heavy Rain" else 0)


def test_chronological_splits_integrity(processed_data):
    """Splits must be contiguous whole days without overlap."""
    train_dates = set(processed_data[processed_data["split"] == "train"]["date"])
    val_dates = set(processed_data[processed_data["split"] == "val"]["date"])
    test_dates = set(processed_data[processed_data["split"] == "test"]["date"])

    assert train_dates.isdisjoint(val_dates)
    assert val_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(test_dates)
