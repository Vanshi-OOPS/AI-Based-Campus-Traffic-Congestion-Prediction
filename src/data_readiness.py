"""Data Readiness and Quality Auditor for Live Traffic Feeds.

Audits data/live/traffic_weather_live.csv to evaluate collection completeness,
missing poll slots, provider error rates, class representation, and retraining readiness.
"""

from typing import Dict, Any, Optional
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg


def analyze_live_data_readiness(csv_path: Optional[Path] = None) -> Dict[str, Any]:
    """Audit the real live collection log and compute data readiness metrics."""
    target_path = csv_path or cfg.LIVE_DATA_PATH

    if not target_path.exists() or target_path.stat().st_size == 0:
        return {
            "status": "NO_DATA",
            "total_rows": 0,
            "real_rows": 0,
            "distinct_days": 0,
            "date_range": ("N/A", "N/A"),
            "failed_polls_pct": 0.0,
            "class_counts": {},
            "has_high_congestion": False,
            "missing_slots_summary": "No data logged yet",
            "verdict": "NOT READY: No live records found. Run live_collector.py to begin collection.",
        }

    # Load CSV
    df = pd.read_csv(target_path, dtype=str)

    # Filter out illustrative mock rows if accidentally present
    if "is_illustrative" in df.columns:
        df_real = df[df["is_illustrative"].str.lower() != "true"].copy()
    else:
        df_real = df.copy()

    total_rows = len(df_real)
    if total_rows == 0:
        return {
            "status": "EMPTY_REAL_DATA",
            "total_rows": 0,
            "real_rows": 0,
            "distinct_days": 0,
            "date_range": ("N/A", "N/A"),
            "failed_polls_pct": 0.0,
            "class_counts": {},
            "has_high_congestion": False,
            "missing_slots_summary": "0 real rows",
            "verdict": "NOT READY: Real collection CSV contains 0 non-mock observations.",
        }

    # Parse timestamps
    df_real["dt_local"] = pd.to_datetime(df_real["local_timestamp"], errors="coerce")
    df_real["date_local"] = df_real["dt_local"].dt.date

    distinct_days = int(df_real["date_local"].nunique())
    min_date = str(df_real["date_local"].min())
    max_date = str(df_real["date_local"].max())

    # Provider failures
    failed_traffic = (df_real["traffic_status"] != "OK").sum()
    failed_weather = (df_real["weather_status"] != "OK").sum()
    any_failure = ((df_real["traffic_status"] != "OK") | (df_real["weather_status"] != "OK")).sum()
    failed_pct = float(round((any_failure / total_rows) * 100.0, 1))

    # Missing slots audit for 08:00 - 10:00 AM window (25 slots per morning)
    expected_slots_per_day = 25
    daily_counts = df_real.groupby("date_local").size().to_dict()
    missing_slots_detail = {}
    for d, count in daily_counts.items():
        missing_slots_detail[str(d)] = {
            "logged_slots": int(count),
            "expected_slots": expected_slots_per_day,
            "missing_slots": max(0, expected_slots_per_day - int(count)),
        }

    # Speed Index Class distribution
    class_series = df_real["live_index_class"].dropna()
    class_series = class_series[class_series != ""]
    class_counts = class_series.value_counts().to_dict()

    state_counts = df_real["live_state"].value_counts().to_dict()
    has_high = bool(class_counts.get("High", 0) > 0 or state_counts.get("road_closure", 0) > 0)

    # Retraining threshold: minimum ~500 clean observations across at least 15 distinct mornings
    min_required_rows = 500
    min_required_days = 15

    if total_rows < 50:
        verdict = (
            f"NOT READY TO RETRAIN: Insufficient volume ({total_rows} rows logged across {distinct_days} day(s)). "
            f"A single week of 5-minute morning monitoring yields ~175 rows. "
            f"Plan the next data readiness audit after 2-3 weeks of continuous weekday logging (target: >={min_required_rows} rows)."
        )
    elif distinct_days < min_required_days or not has_high:
        verdict = (
            f"PARTIALLY COLLECTED BUT NOT READY: {total_rows} rows logged across {distinct_days} day(s). "
            f"Need >= {min_required_days} distinct morning sessions and at least several High congestion intervals. "
            f"High congestion occurrences so far: {class_counts.get('High', 0)}."
        )
    else:
        verdict = (
            f"READY FOR FEASIBILITY RETRAINING REVIEW: {total_rows} rows logged across {distinct_days} mornings "
            f"with observed variation (High classes: {class_counts.get('High', 0)})."
        )

    return {
        "status": "AUDITED",
        "total_rows": total_rows,
        "distinct_days": distinct_days,
        "date_range": (min_date, max_date),
        "failed_polls_pct": failed_pct,
        "failed_traffic_count": int(failed_traffic),
        "failed_weather_count": int(failed_weather),
        "class_counts": class_counts,
        "state_counts": state_counts,
        "has_high_congestion": has_high,
        "daily_slots": missing_slots_detail,
        "verdict": verdict,
    }


def print_readiness_report() -> None:
    """Print formatted terminal report for data readiness audit."""
    res = analyze_live_data_readiness()

    print("=" * 75)
    print("LIVE TRAFFIC & WEATHER DATA READINESS AUDIT (BEST-EFFORT AUDIT TRAIL)")
    print("=" * 75)
    print(f"Target CSV: {cfg.LIVE_DATA_PATH}")
    print(f"Total Real Rows Logged: {res['total_rows']}")
    print(f"Distinct Morning Sessions: {res['distinct_days']}")
    print(f"Date Range: {res['date_range'][0]} to {res['date_range'][1]}")
    print(f"Provider Poll Failure Rate: {res['failed_polls_pct']}%")
    if res.get("failed_traffic_count"):
        print(f"  - TomTom Failures: {res['failed_traffic_count']}")
    if res.get("failed_weather_count"):
        print(f"  - Open-Meteo Failures: {res['failed_weather_count']}")

    print("\n--- Live Speed-Based Congestion Index Distribution ---")
    if res.get("class_counts"):
        for cls, count in res["class_counts"].items():
            print(f"  - Class '{cls}': {count} interval(s)")
    else:
        print("  - No valid classes logged yet.")

    print("\n--- Overarching Live State Breakdown ---")
    if res.get("state_counts"):
        for state, count in res["state_counts"].items():
            print(f"  - State '{state}': {count}")

    print("\n--- Missing Morning Poll Slots Audit ---")
    if res.get("daily_slots"):
        for day, info in list(res["daily_slots"].items())[:5]:
            print(f"  - {day}: {info['logged_slots']}/{info['expected_slots']} slots (missing: {info['missing_slots']})")
    else:
        print("  - No morning sessions to audit.")

    print("\n" + "=" * 75)
    print("READINESS VERDICT:")
    print(f"  >> {res['verdict']}")
    print("=" * 75)


if __name__ == "__main__":
    print_readiness_report()
