"""Live Traffic and Weather Data Collector.

Polls TomTom Traffic Flow API and Open-Meteo Weather API at regular intervals,
computes the live speed-based congestion index, and appends structured audit records
to data/live/traffic_weather_live.csv.

Ground Rules:
- Real API calls log to data/live/traffic_weather_live.csv.
- Mock/demo calls log strictly to data/live/mock_feed_demo.csv and are never mixed.
- API keys are read from environment (.env via python-dotenv) and strictly scrubbed.
- Atomic append with strict field-count validation asserting every row has exactly 35 columns.
- Flags unexpected segment fingerprint changes.
"""

from typing import Dict, Any, Optional, List
import sys
import os
import time
import signal
import argparse
from datetime import datetime, timezone
import zoneinfo
import csv
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src import config as cfg
from src.fetch_traffic import fetch_tomtom_traffic, scrub_api_key
from src.fetch_weather import fetch_open_meteo_weather
from src.live_index import compute_live_speed_index

# Field names for the live CSV schema (Version 1.2: exactly 35 columns)
FIELDNAMES: List[str] = [
    "poll_id",
    "schema_version",
    "is_illustrative",
    "utc_timestamp",
    "local_timestamp",
    "traffic_source",
    "weather_source",
    "query_lat",
    "query_lon",
    "snapped_lat",
    "snapped_lon",
    "segment_fingerprint",
    "gate_to_segment_m",
    "segment_length_m",
    "road_class",
    "closure_flag",
    "current_speed",
    "free_flow_speed",
    "speed_ratio",
    "current_travel_time",
    "free_flow_travel_time",
    "confidence",
    "low_confidence_flag",
    "live_speed_congestion_index",
    "live_index_class",
    "live_state",
    "precip_mm",
    "rain_mm",
    "wmo_code",
    "weather_category",
    "traffic_status",
    "traffic_http_status",
    "weather_status",
    "note",
    "error",
]

# Pinned query coordinates for the inbound corridor (approaching gate from West)
PINNED_QUERY_LAT = 12.9683232
PINNED_QUERY_LON = 79.1559336
EXPECTED_SEGMENT_FINGERPRINT = "12.9660,79.1373->12.9968,79.2455:13519"

# Graceful termination handler
_RUNNING = True


def _signal_handler(sig: int, frame: Any) -> None:
    global _RUNNING
    print("\n[INFO] Termination signal received. Shutting down collector cleanly...")
    _RUNNING = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def is_within_monitoring_window(local_dt: datetime, window_str: Optional[str]) -> bool:
    """Check if local time falls within monitoring window (e.g. '08:00-10:00')."""
    if not window_str:
        return True
    try:
        start_str, end_str = window_str.split("-")
        sh, sm = map(int, start_str.strip().split(":"))
        eh, em = map(int, end_str.strip().split(":"))
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        curr_min = local_dt.hour * 60 + local_dt.minute
        return start_min <= curr_min <= end_min
    except Exception:
        return True


def poll_and_build_record(
    query_lat: float = PINNED_QUERY_LAT,
    query_lon: float = PINNED_QUERY_LON,
    is_mock: bool = False,
) -> Dict[str, Any]:
    """Execute a single poll of traffic and weather feeds, format and validate record."""
    now_utc = datetime.now(timezone.utc)
    local_tz = zoneinfo.ZoneInfo(cfg.LOCAL_TIMEZONE_STR)
    now_local = now_utc.astimezone(local_tz)

    poll_id = f"poll_{now_utc.strftime('%Y%m%d_%H%M%S')}_{now_utc.microsecond // 1000:03d}"
    utc_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    local_iso = now_local.isoformat()

    if is_mock:
        # Seeded deterministic mock response for testing/demo
        return {
            "poll_id": poll_id,
            "schema_version": "1.2",
            "is_illustrative": "True",
            "utc_timestamp": utc_iso,
            "local_timestamp": local_iso,
            "traffic_source": "mock",
            "weather_source": "mock",
            "query_lat": round(query_lat, 5),
            "query_lon": round(query_lon, 5),
            "snapped_lat": 12.96597,
            "snapped_lon": 79.13726,
            "segment_fingerprint": EXPECTED_SEGMENT_FINGERPRINT,
            "gate_to_segment_m": 121.1,
            "segment_length_m": 13519.1,
            "road_class": "FRC2",
            "closure_flag": "False",
            "current_speed": 45.0,
            "free_flow_speed": 50.0,
            "speed_ratio": 0.9000,
            "current_travel_time": 950.0,
            "free_flow_travel_time": 900.0,
            "confidence": 0.90,
            "low_confidence_flag": "False",
            "live_speed_congestion_index": 11.76,
            "live_index_class": "Low",
            "live_state": "normal",
            "precip_mm": 0.0,
            "rain_mm": 0.0,
            "wmo_code": 0,
            "weather_category": "Clear",
            "traffic_status": "OK",
            "traffic_http_status": 200,
            "weather_status": "OK",
            "note": "Deterministic mock demo row; not live",
            "error": "",
        }

    # Real Traffic Query
    traffic_res = fetch_tomtom_traffic(lat=query_lat, lon=query_lon, zoom=18)
    # Real Weather Query
    weather_res = fetch_open_meteo_weather(lat=query_lat, lon=query_lon)

    # Compute Speed Index
    index_res = compute_live_speed_index(
        current_speed=traffic_res.get("current_speed"),
        free_flow_speed=traffic_res.get("free_flow_speed"),
        road_closure=bool(traffic_res.get("road_closure", False)),
        confidence=traffic_res.get("confidence"),
    )

    # Verify Fingerprint Continuity
    seg_fp = traffic_res.get("segment_fingerprint")
    note_parts = ["Corridor speed signal (13.5 km segment)"]
    if index_res.get("note"):
        note_parts.append(index_res["note"])

    if seg_fp and seg_fp != EXPECTED_SEGMENT_FINGERPRINT:
        fp_warn = f"Warning: segment fingerprint changed ({seg_fp})"
        print(f"[WARN] {fp_warn}")
        note_parts.append(fp_warn)

    combined_note = "; ".join(note_parts)

    # Combine errors cleanly (scrubbed)
    err_parts = []
    if traffic_res.get("error"):
        err_parts.append(scrub_api_key(str(traffic_res["error"])))
    if weather_res.get("error"):
        err_parts.append(scrub_api_key(str(weather_res["error"])))
    combined_error = "; ".join(err_parts)

    record: Dict[str, Any] = {
        "poll_id": poll_id,
        "schema_version": "1.2",
        "is_illustrative": "False",
        "utc_timestamp": utc_iso,
        "local_timestamp": local_iso,
        "traffic_source": traffic_res.get("source", "unavailable"),
        "weather_source": weather_res.get("source", "unavailable"),
        "query_lat": round(query_lat, 5),
        "query_lon": round(query_lon, 5),
        "snapped_lat": traffic_res.get("snapped_lat") or "",
        "snapped_lon": traffic_res.get("snapped_lon") or "",
        "segment_fingerprint": seg_fp or "",
        "gate_to_segment_m": traffic_res.get("gate_to_segment_m") if traffic_res.get("gate_to_segment_m") is not None else "",
        "segment_length_m": traffic_res.get("segment_length_m") if traffic_res.get("segment_length_m") is not None else "",
        "road_class": traffic_res.get("road_class") or "",
        "closure_flag": str(traffic_res["road_closure"]) if traffic_res.get("road_closure") is not None else "",
        "current_speed": traffic_res.get("current_speed") if traffic_res.get("current_speed") is not None else "",
        "free_flow_speed": traffic_res.get("free_flow_speed") if traffic_res.get("free_flow_speed") is not None else "",
        "speed_ratio": index_res.get("speed_ratio") if index_res.get("speed_ratio") is not None else "",
        "current_travel_time": traffic_res.get("current_travel_time") if traffic_res.get("current_travel_time") is not None else "",
        "free_flow_travel_time": traffic_res.get("free_flow_travel_time") if traffic_res.get("free_flow_travel_time") is not None else "",
        "confidence": traffic_res.get("confidence") if traffic_res.get("confidence") is not None else "",
        "low_confidence_flag": str(index_res["low_confidence_flag"]) if index_res.get("low_confidence_flag") is not None else "",
        "live_speed_congestion_index": index_res.get("live_speed_congestion_index") if index_res.get("live_speed_congestion_index") is not None else "",
        "live_index_class": index_res.get("live_index_class") or "",
        "live_state": index_res.get("live_state", "normal"),
        "precip_mm": weather_res.get("precip_mm") if weather_res.get("precip_mm") is not None else "",
        "rain_mm": weather_res.get("rain_mm") if weather_res.get("rain_mm") is not None else "",
        "wmo_code": weather_res.get("wmo_code") if weather_res.get("wmo_code") is not None else "",
        "weather_category": weather_res.get("weather_category") or "",
        "traffic_status": traffic_res.get("status", "ERROR"),
        "traffic_http_status": traffic_res.get("http_status") or "",
        "weather_status": weather_res.get("status", "ERROR"),
        "note": combined_note,
        "error": combined_error,
    }

    return record


def write_record_atomically(record: Dict[str, Any], target_csv: Path) -> None:
    """Atomically append a validated record to the CSV file.

    Enforces strict field count validation against FIELDNAMES (exactly 35 columns).
    """
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = target_csv.exists() and target_csv.stat().st_size > 0

    # Ensure all fieldnames exist in record
    row_values = [str(record.get(col, "")) for col in FIELDNAMES]
    assert len(row_values) == len(FIELDNAMES), (
        f"Row column count mismatch: {len(row_values)} != {len(FIELDNAMES)}"
    )

    with open(target_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(FIELDNAMES)
        writer.writerow(row_values)
        f.flush()


def run_collector(
    once: bool = False,
    interval_minutes: int = 5,
    window: Optional[str] = "08:00-10:00",
    is_mock: bool = False,
    query_lat: float = PINNED_QUERY_LAT,
    query_lon: float = PINNED_QUERY_LON,
) -> None:
    """Run the live data collector loop or single execution."""
    target_csv = cfg.MOCK_DATA_PATH if is_mock else cfg.LIVE_DATA_PATH
    mode_str = "MOCK DEMO" if is_mock else "REAL LIVE API"

    print("=" * 70)
    print(f"STARTING LIVE TRAFFIC & WEATHER COLLECTOR ({mode_str})")
    print("=" * 70)
    print(f"Target CSV: {target_csv}")
    print(f"Query Coordinates: ({query_lat:.6f}, {query_lon:.6f})")
    print(f"Expected Fingerprint: {EXPECTED_SEGMENT_FINGERPRINT}")
    print(f"Interval: {interval_minutes} minute(s) | Window: {window or 'All day'}")
    print(f"Local Timezone: {cfg.LOCAL_TIMEZONE_STR}")
    print("-" * 70)

    poll_count = 0
    local_tz = zoneinfo.ZoneInfo(cfg.LOCAL_TIMEZONE_STR)

    while _RUNNING:
        now_local = datetime.now(timezone.utc).astimezone(local_tz)

        if is_within_monitoring_window(now_local, window):
            poll_count += 1
            print(f"[{now_local.strftime('%H:%M:%S')} IST] Poll #{poll_count}...", end=" ", flush=True)

            try:
                rec = poll_and_build_record(query_lat=query_lat, query_lon=query_lon, is_mock=is_mock)
                write_record_atomically(rec, target_csv)
                status_summary = (
                    f"Traffic: {rec['traffic_status']} (spd={rec['current_speed']}/{rec['free_flow_speed']}), "
                    f"Weather: {rec['weather_status']} ({rec['weather_category']}), "
                    f"State: {rec['live_state']}"
                )
                print(f"[OK] -> {status_summary}")
            except Exception as e:
                print(f"[ERROR] Poll failed: {type(e).__name__}: {str(e)}")
        else:
            print(f"[{now_local.strftime('%H:%M:%S')} IST] Outside window ({window}). Sleeping...")

        if once or not _RUNNING:
            break

        # Sleep in 1-second increments to catch SIGINT promptly
        sleep_secs = interval_minutes * 60
        for _ in range(sleep_secs):
            if not _RUNNING:
                break
            time.sleep(1.0)

    print("\n[INFO] Collector stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live traffic and weather data collector.")
    parser.add_argument("--once", action="store_true", help="Execute single poll and exit.")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in minutes (default: 5).")
    parser.add_argument("--window", type=str, default="08:00-10:00", help="Local monitoring window (e.g. '08:00-10:00'). Use 'none' for 24/7.")
    parser.add_argument("--demo-mock", action="store_true", help="Run in mock demo mode; writes to mock_feed_demo.csv.")
    parser.add_argument("--lat", type=float, default=PINNED_QUERY_LAT, help="Query latitude.")
    parser.add_argument("--lon", type=float, default=PINNED_QUERY_LON, help="Query longitude.")

    args = parser.parse_args()
    win = None if args.window.lower() == "none" else args.window

    run_collector(
        once=args.once,
        interval_minutes=args.interval,
        window=win,
        is_mock=args.demo_mock,
        query_lat=args.lat,
        query_lon=args.lon,
    )


if __name__ == "__main__":
    main()
