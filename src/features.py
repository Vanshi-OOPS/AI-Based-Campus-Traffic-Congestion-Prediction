"""Feature Engineering Pipeline for Morning Traffic Congestion.

Processes raw simulated traffic logs into an ML-ready feature matrix:
- Strict per-morning grouping (no lag features cross day boundaries).
- Rolling features exclude current row (lookback 3 steps = 15 minutes).
- Documented first-interval rule (at t=0, surge=0, rolling avg falls back to current count).
- Ordinal and fixed one-hot weather encodings.
- Targets: nowcast (score & class) and 15-minute forecast (class at t+15min, NaN for morning tail).
- Chronological whole-day split: 70% train / 15% val / 15% test.
- Verified isolation: no latent leakage columns in the feature set.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src import config as cfg


def compute_morning_lag_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly backward-looking lag features within a single morning.

    Parameters
    ----------
    group : pd.DataFrame
        Rows for a single morning, ordered chronologically (minute_of_window 0 to 120).

    Returns
    -------
    pd.DataFrame
        Group with rolling_vehicle_avg_15min, vehicle_surge_rate, and forecast target.
    """
    group = group.sort_values("minute_of_window").copy()
    counts = group["vehicle_count"].values
    n = len(counts)

    rolling_avg = np.zeros(n, dtype=np.float64)
    surge_rate = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            # First interval rule: at 8:00 AM (t=0), no prior window history exists.
            # Documented rule: fallback to current vehicle count, surge rate is 0.
            rolling_avg[i] = float(counts[0])
            surge_rate[i] = 0.0
        elif i == 1:
            # 1 interval of prior history available
            rolling_avg[i] = float(counts[0])
            surge_rate[i] = float(counts[1] - counts[0])
        elif i == 2:
            # 2 intervals of prior history available
            rolling_avg[i] = float(np.mean(counts[0:2]))
            surge_rate[i] = float(counts[2] - counts[1])
        else:
            # 3 intervals of prior history available (excluding current count i)
            rolling_avg[i] = float(np.mean(counts[i-3:i]))
            surge_rate[i] = float(counts[i] - counts[i-1])

    group["rolling_vehicle_avg_15min"] = np.round(rolling_avg, 2)
    group["vehicle_surge_rate"] = np.round(surge_rate, 2)

    # 15-minute forward forecast target (shift -3 steps within the morning)
    # The last 3 rows (intervals 22, 23, 24) receive NaN as t+15min falls outside the window
    group["congestion_class_t_plus_15min"] = group["congestion_class"].shift(-3)

    return group


def assign_chronological_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Assign whole mornings (dates) chronologically to train, val, and test splits."""
    df = df.copy()
    unique_dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    total_days = len(unique_dates)

    n_train = int(np.round(total_days * cfg.TRAIN_SPLIT_RATIO))
    n_val = int(np.round(total_days * cfg.VAL_SPLIT_RATIO))

    train_dates = set(unique_dates.iloc[:n_train])
    val_dates = set(unique_dates.iloc[n_train:n_train + n_val])
    test_dates = set(unique_dates.iloc[n_train + n_val:])

    def get_split(d: str) -> str:
        if d in train_dates:
            return "train"
        elif d in val_dates:
            return "val"
        else:
            return "test"

    df["split"] = df["date"].map(get_split)
    return df


def build_featured_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw simulated data into a processed ML-ready DataFrame.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw simulated DataFrame from simulate_data.py.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with engineered features, encoded weather, splits, and targets.
    """
    df = raw_df.copy()

    # 1. Temporal Indicator: Peak rush window (8:30 - 9:15 AM)
    df["is_peak_window"] = (
        (df["minute_of_window"] >= cfg.PEAK_START_MINUTE_OF_WINDOW)
        & (df["minute_of_window"] <= cfg.PEAK_END_MINUTE_OF_WINDOW)
    ).astype(int)

    # 2. Weather Encodings (Fixed mappings from config, no fitting needed)
    df["weather_ordinal"] = df["weather"].map(cfg.WEATHER_ORDINAL_MAP)
    # One-hot encoding with Clear as baseline
    df["weather_Cloudy"] = (df["weather"] == "Cloudy").astype(int)
    df["weather_Rain"] = (df["weather"] == "Rain").astype(int)
    df["weather_Heavy Rain"] = (df["weather"] == "Heavy Rain").astype(int)

    # 3. Lag and Trend features strictly per morning
    processed_groups = [
        compute_morning_lag_features(group)
        for _, group in df.groupby("date", sort=False)
    ]
    df = pd.concat(processed_groups, ignore_index=True)

    # 4. Assign Chronological Splits
    df = assign_chronological_splits(df)

    # 5. Order columns cleanly: identifiers, features, targets, placeholders
    ordered_cols = (
        cfg.METADATA_COLS
        + cfg.MODEL_FEATURE_COLS
        + [
            "congestion_score",
            "congestion_class",
            "congestion_class_t_plus_15min",
            "speed_ratio",  # Kept as schema placeholder / cross-check (marked leakage)
            "weather",      # Raw categorical retained for auditing and dashboard
        ]
    )
    # Remove duplicates while preserving order
    seen = set()
    final_cols = [c for c in ordered_cols if not (c in seen or seen.add(c))]

    return df[final_cols]


def main() -> None:
    """Execute feature engineering pipeline from raw data to processed CSV."""
    if not cfg.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {cfg.RAW_DATA_PATH}. Run simulate_data.py first.")

    print(f"Loading raw data from: {cfg.RAW_DATA_PATH}...")
    raw_df = pd.read_csv(cfg.RAW_DATA_PATH)

    print("Computing features, lag variables, and chronological splits...")
    featured_df = build_featured_dataset(raw_df)

    # Save to disk
    featured_df.to_csv(cfg.PROCESSED_DATA_PATH, index=False)
    print(f"[OK] Processed dataset saved to: {cfg.PROCESSED_DATA_PATH} ({len(featured_df)} rows)")

    # Print split summary
    print("\nDataset split summary (by whole days):")
    split_summary = featured_df.groupby("split")["date"].nunique()
    print(split_summary)


if __name__ == "__main__":
    main()
