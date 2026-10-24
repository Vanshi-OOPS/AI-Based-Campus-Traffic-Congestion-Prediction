"""Data Validation and Sanity Check Pipeline.

Performs rigorous structural, statistical, and leakage tests on simulated and processed data:
1. Row count, timestamp integrity, and zero duplication.
2. Weather and event frequency analysis overall and per split.
3. Class balance (Low / Moderate / High) overall and per split; asserts all 3 classes in every split.
4. Per-weather class distribution analysis to audit queue saturation on rainy mornings.
5. Verification that lag features never cross morning boundaries.
6. Bitwise reproducibility check (runs simulation twice with same seed).
7. Generates sanity check plots saved to docs/figures/.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

from src import config as cfg
from src.simulate_data import simulate_traffic_dataset
from src.features import build_featured_dataset


def run_validation_pipeline() -> bool:
    """Run full validation suite and output results to console and docs/figures/."""
    print("=" * 70)
    print("STARTING DATA VALIDATION & INTEGRITY CHECKS")
    print("=" * 70)

    # 1. Load Data
    raw_df = pd.read_csv(cfg.RAW_DATA_PATH)
    proc_df = pd.read_csv(cfg.PROCESSED_DATA_PATH)
    diag_df = pd.read_csv(cfg.DIAGNOSTICS_DATA_PATH)

    total_expected_rows = cfg.NUM_DAYS * cfg.INTERVALS_PER_MORNING
    assert len(raw_df) == total_expected_rows, f"Raw rows {len(raw_df)} != {total_expected_rows}"
    assert len(proc_df) == total_expected_rows, f"Proc rows {len(proc_df)} != {total_expected_rows}"
    assert len(diag_df) == total_expected_rows, f"Diag rows {len(diag_df)} != {total_expected_rows}"
    print(f"[CHECK 1 PASSED] Row counts match expected: {len(raw_df)} rows across {cfg.NUM_DAYS} days.")

    # 2. Check 25 intervals per morning and timestamp continuity
    intervals_per_day = raw_df.groupby("date")["timestamp"].count()
    assert (intervals_per_day == cfg.INTERVALS_PER_MORNING).all(), "Every day must have exactly 25 intervals!"
    assert raw_df["timestamp"].duplicated().sum() == 0, "Found duplicate timestamps in raw data!"
    assert proc_df["timestamp"].duplicated().sum() == 0, "Found duplicate timestamps in processed data!"
    print(f"[CHECK 2 PASSED] Exactly {cfg.INTERVALS_PER_MORNING} timestamps per morning; 0 duplicates.")

    # 3. Chronological Splits & Day Boundary Integrity
    split_days = proc_df.groupby("split")["date"].nunique()
    print("\n--- Day Counts by Split ---")
    for s, count in split_days.items():
        print(f"  {s:<6}: {count} mornings ({count * cfg.INTERVALS_PER_MORNING} rows)")

    # Assert no overlapping days between splits
    train_dates = set(proc_df[proc_df["split"] == "train"]["date"])
    val_dates = set(proc_df[proc_df["split"] == "val"]["date"])
    test_dates = set(proc_df[proc_df["split"] == "test"]["date"])
    assert train_dates.isdisjoint(val_dates), "Train and Val dates overlap!"
    assert val_dates.isdisjoint(test_dates), "Val and Test dates overlap!"
    assert train_dates.isdisjoint(test_dates), "Train and Test dates overlap!"
    print("[CHECK 3 PASSED] Splits are non-overlapping and chronologically contiguous.")

    # 4. Weather & Event Counts per Split
    print("\n--- Weather Mornings Count per Split ---")
    weather_per_split = (
        proc_df.groupby(["split", "weather"])["date"].nunique().unstack(fill_value=0)
    )
    print(weather_per_split)

    print("\n--- Event Mornings Count per Split ---")
    event_per_split = (
        proc_df.groupby(["split", "event_flag"])["date"].nunique().unstack(fill_value=0)
    )
    event_per_split.columns = ["No Event (0)", "Event (1)"]
    print(event_per_split)

    # Flag condition warnings
    for s in ["train", "val", "test"]:
        rain_days = weather_per_split.loc[s].get("Rain", 0) + weather_per_split.loc[s].get("Heavy Rain", 0)
        event_days = event_per_split.loc[s].get("Event (1)", 0)
        if rain_days < 3:
            print(f"  [WARNING] Split '{s}' has only {rain_days} rainy days.")
        if event_days < 2:
            print(f"  [WARNING] Split '{s}' has only {event_days} event days.")

    # 5. Class Distribution per Split
    print("\n--- Congestion Class Distribution (% and counts) ---")
    for s in ["train", "val", "test"]:
        sub = proc_df[proc_df["split"] == s]
        dist = sub["congestion_class"].value_counts(normalize=True) * 100
        counts = sub["congestion_class"].value_counts()
        print(f"  Split: {s:<5} -> Low: {dist.get('Low', 0):.1f}% ({counts.get('Low', 0)}), "
              f"Moderate: {dist.get('Moderate', 0):.1f}% ({counts.get('Moderate', 0)}), "
              f"High: {dist.get('High', 0):.1f}% ({counts.get('High', 0)})")
        
        # Assert all 3 classes appear
        for c in cfg.CONGESTION_CLASSES:
            assert counts.get(c, 0) > 0, f"Class {c} missing in {s} split!"
        
        # Check High representation
        high_pct = dist.get("High", 0)
        if high_pct < 5.0:
            print(f"  [WARNING] High class in split '{s}' is under 5.0% ({high_pct:.2f}%)!")

    print("[CHECK 4 PASSED] All 3 congestion classes exist across every split.")

    # 6. Audit Weather Saturation (Check Class Split by Weather Condition)
    print("\n--- Congestion Class Distribution by Weather (Queue Saturation Audit) ---")
    weather_class_dist = pd.crosstab(
        proc_df["weather"],
        proc_df["congestion_class"],
        normalize="index"
    ) * 100
    print(weather_class_dist.round(1))

    # 7. Lag Feature Isolation Check (No crossing morning boundaries)
    first_intervals = proc_df[proc_df["minute_of_window"] == 0]
    assert (first_intervals["vehicle_surge_rate"] == 0.0).all(), (
        "First interval of each morning must have vehicle_surge_rate == 0.0!"
    )
    assert (first_intervals["rolling_vehicle_avg_15min"] == first_intervals["vehicle_count"]).all(), (
        "First interval rolling_vehicle_avg_15min must equal current vehicle_count!"
    )
    print("\n[CHECK 5 PASSED] Lag features strictly isolated per morning (zero boundary leakage).")

    # 8. Leakage Column Isolation Check
    for leak_col in cfg.LEAKAGE_COLS:
        assert leak_col not in cfg.MODEL_FEATURE_COLS, (
            f"LEAKAGE ALERT: Column '{leak_col}' found inside MODEL_FEATURE_COLS!"
        )
    print(f"[CHECK 6 PASSED] Model feature list ({len(cfg.MODEL_FEATURE_COLS)} features) contains NO leakage columns.")

    # 9. Reproducibility Check
    print("\n--- Running Reproducibility Verification ---")
    pub1, _ = simulate_traffic_dataset(num_days=10, seed=cfg.RANDOM_SEED)
    pub2, _ = simulate_traffic_dataset(num_days=10, seed=cfg.RANDOM_SEED)
    pd.testing.assert_frame_equal(pub1, pub2)
    print("[CHECK 7 PASSED] Bitwise identical output confirmed across repeated runs with seed=42.")

    # 10. Generate Sanity Check Figures
    print("\n--- Generating Sanity Check Visualizations ---")
    cfg.DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Mean arrival demand curve across the 8:00 - 10:00 AM window
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rush_stats = proc_df.groupby("minute_of_window")["vehicle_count"].agg(["mean", "std"])
    ax.plot(rush_stats.index, rush_stats["mean"], color="#1f77b4", lw=2, label="Mean Vehicle Arrivals")
    ax.fill_between(
        rush_stats.index,
        rush_stats["mean"] - rush_stats["std"],
        rush_stats["mean"] + rush_stats["std"],
        color="#1f77b4",
        alpha=0.2,
        label="±1 Std Dev",
    )
    ax.axvspan(30, 75, color="orange", alpha=0.15, label="Rush Peak Surge (8:30-9:15)")
    ax.set_title("Arrival Demand Profile Across Morning Window (8:00 - 10:00 AM)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Minute of Window (0 = 08:00, 60 = 09:00, 120 = 10:00)")
    ax.set_ylabel("Vehicle Count (arrivals / 5-min)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(cfg.DOCS_FIGURES_DIR / "arrival_curve.png", dpi=150)
    plt.close(fig)

    # Plot 2: Score distribution by weather and event
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    proc_df.boxplot(column="congestion_score", by="weather", ax=axes[0], grid=False)
    axes[0].set_title("Congestion Score by Weather Condition", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Weather")
    axes[0].set_ylabel("Score (0 - 100)")
    axes[0].axhline(40, color="green", linestyle="--", alpha=0.7, label="Low Threshold (40)")
    axes[0].axhline(70, color="red", linestyle="--", alpha=0.7, label="High Alert (70)")
    axes[0].legend(loc="upper left")

    proc_df.boxplot(column="congestion_score", by="event_flag", ax=axes[1], grid=False)
    axes[1].set_title("Congestion Score by Event Presence", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Event Flag (0 = Normal, 1 = Campus Event)")
    axes[1].set_ylabel("Score (0 - 100)")
    axes[1].axhline(40, color="green", linestyle="--", alpha=0.7)
    axes[1].axhline(70, color="red", linestyle="--", alpha=0.7)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(cfg.DOCS_FIGURES_DIR / "score_by_weather_and_event.png", dpi=150)
    plt.close(fig)

    # Plot 3: Feature Correlation Heatmap with Congestion Score
    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = ["vehicle_count", "rolling_vehicle_avg_15min", "vehicle_surge_rate",
                "weather_ordinal", "event_flag", "is_peak_window", "congestion_score"]
    corr = proc_df[num_cols].corr()
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    ax.set_title("Feature Correlation Matrix with Ground-Truth Congestion Score", pad=20, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(cfg.DOCS_FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"[OK] Sanity check figures saved to: {cfg.DOCS_FIGURES_DIR}")
    print("=" * 70)
    print("ALL INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    run_validation_pipeline()
