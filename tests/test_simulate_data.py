"""Unit tests for the simulated data generator (src/simulate_data.py)."""

import pytest
import pandas as pd
import numpy as np

from src import config as cfg
from src.simulate_data import simulate_traffic_dataset


@pytest.fixture(scope="module")
def sample_data():
    """Generate a 14-day sample dataset for fast testing."""
    pub_df, diag_df = simulate_traffic_dataset(num_days=14, seed=cfg.RANDOM_SEED)
    return pub_df, diag_df


def test_row_counts_and_grid(sample_data):
    """Verify exact 25 timestamps per morning and expected total rows."""
    pub_df, diag_df = sample_data
    expected_rows = 14 * cfg.INTERVALS_PER_MORNING
    assert len(pub_df) == expected_rows
    assert len(diag_df) == expected_rows

    # Verify 25 timestamps per morning
    counts = pub_df.groupby("date")["timestamp"].count()
    assert (counts == 25).all()
    assert pub_df["timestamp"].duplicated().sum() == 0


def test_seed_reproducibility():
    """Verify that running with identical seeds generates bitwise identical outputs."""
    df1, _ = simulate_traffic_dataset(num_days=5, seed=cfg.RANDOM_SEED)
    df2, _ = simulate_traffic_dataset(num_days=5, seed=cfg.RANDOM_SEED)
    pd.testing.assert_frame_equal(df1, df2)

    # Different seed must yield different outputs
    df3, _ = simulate_traffic_dataset(num_days=5, seed=999)
    assert not df1["vehicle_count"].equals(df3["vehicle_count"])


def test_per_morning_constants(sample_data):
    """Weather and event_flag must remain constant within each morning."""
    pub_df, _ = sample_data
    for date, group in pub_df.groupby("date"):
        assert group["weather"].nunique() == 1, f"Weather changed within morning {date}"
        assert group["event_flag"].nunique() == 1, f"Event flag changed within morning {date}"


def test_score_bounds_and_classes(sample_data):
    """Scores must be clipped to [0, 100] and classes must follow strict thresholds."""
    pub_df, _ = sample_data
    scores = pub_df["congestion_score"]
    assert (scores >= 0.0).all()
    assert (scores <= 100.0).all()

    for _, row in pub_df.iterrows():
        s = row["congestion_score"]
        c = row["congestion_class"]
        if s < cfg.SCORE_LOW_MAX:
            assert c == "Low", f"Score {s} misclassified as {c}"
        elif s < cfg.SCORE_MODERATE_MAX:
            assert c == "Moderate", f"Score {s} misclassified as {c}"
        else:
            assert c == "High", f"Score {s} misclassified as {c}"


def test_speed_ratio_bounds_and_monotonicity(sample_data):
    """Speed ratio must lie in [0.15, 1.0] and decrease as score increases."""
    pub_df, _ = sample_data
    ratios = pub_df["speed_ratio"]
    assert (ratios >= cfg.SPEED_RATIO_MIN).all()
    assert (ratios <= cfg.SPEED_RATIO_MAX).all()

    # Monotonicity check
    corr = np.corrcoef(pub_df["congestion_score"], pub_df["speed_ratio"])[0, 1]
    assert corr < -0.99, f"Speed ratio is not strongly negatively correlated with score: {corr}"


def test_queue_non_negativity(sample_data):
    """Queue length must always be >= 0."""
    _, diag_df = sample_data
    assert (diag_df["queue_length"] >= 0.0).all()


def test_data_source_label(sample_data):
    """Every row must explicitly declare data_source as 'simulated'."""
    pub_df, _ = sample_data
    assert (pub_df["data_source"] == "simulated").all()
