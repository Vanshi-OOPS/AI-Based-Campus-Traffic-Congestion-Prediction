"""Live data ingestion and cache manager for the Streamlit dashboard.

Manages cached live queries to TomTom and Open-Meteo APIs (with TTL caching
to prevent API quota exhaustion on UI reruns) or falls back to persisted live records
and deterministic mock demo feeds.
"""

from typing import Dict, Any, Optional
import os
import csv
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st

from src import config as cfg
from src.fetch_traffic import fetch_tomtom_traffic, scrub_api_key
from src.fetch_weather import fetch_open_meteo_weather
from src.live_index import compute_live_speed_index


def get_latest_logged_record(csv_path: Path) -> Optional[Dict[str, Any]]:
    """Read the most recent row from a live CSV log."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return None
            last_row = reader[-1]
            # Convert numeric strings back to float/int where appropriate
            typed_row: Dict[str, Any] = dict(last_row)
            for k in [
                "current_speed",
                "free_flow_speed",
                "speed_ratio",
                "current_travel_time",
                "free_flow_travel_time",
                "confidence",
                "live_speed_congestion_index",
                "precip_mm",
                "rain_mm",
                "gate_to_segment_m",
                "segment_length_m",
            ]:
                if typed_row.get(k) not in [None, ""]:
                    try:
                        typed_row[k] = float(typed_row[k])
                    except (ValueError, TypeError):
                        pass
            if typed_row.get("wmo_code") not in [None, ""]:
                try:
                    typed_row["wmo_code"] = int(typed_row["wmo_code"])
                except (ValueError, TypeError):
                    pass
            return typed_row
    except Exception:
        return None


def generate_deterministic_mock_record() -> Dict[str, Any]:
    """Generate deterministic mock demo record when real feeds are unavailable."""
    now_utc = datetime.now(timezone.utc)
    return {
        "poll_id": "mock_demo_poll",
        "schema_version": "1.2",
        "is_illustrative": True,
        "utc_timestamp": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_timestamp": now_utc.astimezone().isoformat(),
        "traffic_source": "mock",
        "weather_source": "mock",
        "query_lat": round(cfg.GATE_COORDINATES[0], 5),
        "query_lon": round(cfg.GATE_COORDINATES[1], 5),
        "snapped_lat": round(cfg.GATE_COORDINATES[0], 5),
        "snapped_lon": round(cfg.GATE_COORDINATES[1], 5),
        "segment_fingerprint": "mock_fingerprint",
        "gate_to_segment_m": 0.0,
        "segment_length_m": 13519.0,
        "road_class": "FRC2",
        "closure_flag": False,
        "current_speed": 45.0,
        "free_flow_speed": 50.0,
        "speed_ratio": 0.90,
        "current_travel_time": 950.0,
        "free_flow_travel_time": 900.0,
        "confidence": 0.90,
        "low_confidence_flag": False,
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
        "note": "Deterministic mock demo feed; not live",
        "error": "",
    }


@st.cache_data(ttl=cfg.CACHE_TTL_SECONDS)
def fetch_live_feed_cached(
    query_lat: float,
    query_lon: float,
    use_mock: bool = False,
    cache_bust_key: int = 0,
) -> Dict[str, Any]:
    """Fetch live data with TTL caching to protect free-tier API quotas.
    
    If use_mock is True, returns deterministic mock feed.
    Otherwise queries TomTom and Open-Meteo APIs, falling back cleanly to logged CSV.
    """
    if use_mock:
        # Check if persisted mock file exists, or generate
        mock_rec = get_latest_logged_record(cfg.MOCK_DATA_PATH)
        if mock_rec:
            return mock_rec
        return generate_deterministic_mock_record()

    # Real API Query
    tomtom_key = os.getenv("TOMTOM_API_KEY")
    if not tomtom_key and hasattr(st, "secrets") and "TOMTOM_API_KEY" in st.secrets:
        tomtom_key = st.secrets["TOMTOM_API_KEY"]

    traffic_res = fetch_tomtom_traffic(lat=query_lat, lon=query_lon, api_key=tomtom_key)
    weather_res = fetch_open_meteo_weather(lat=query_lat, lon=query_lon)

    # If both APIs failed completely (e.g. offline sandbox or no key), check if we have recent logged CSV
    if traffic_res.get("status") == "ERROR" and weather_res.get("status") == "ERROR":
        logged_rec = get_latest_logged_record(cfg.LIVE_DATA_PATH)
        if logged_rec:
            return logged_rec

    # Compute Index
    index_res = compute_live_speed_index(
        current_speed=traffic_res.get("current_speed"),
        free_flow_speed=traffic_res.get("free_flow_speed"),
        road_closure=bool(traffic_res.get("road_closure", False)),
        confidence=traffic_res.get("confidence"),
    )

    now_utc = datetime.now(timezone.utc)
    seg_fp = traffic_res.get("segment_fingerprint")
    note_parts = ["Corridor speed signal (13.5 km segment)"]
    if index_res.get("note"):
        note_parts.append(index_res["note"])

    err_parts = []
    if traffic_res.get("error"):
        err_parts.append(scrub_api_key(str(traffic_res["error"])))
    if weather_res.get("error"):
        err_parts.append(scrub_api_key(str(weather_res["error"])))

    return {
        "poll_id": f"dash_{now_utc.strftime('%Y%m%d_%H%M%S')}",
        "schema_version": "1.2",
        "is_illustrative": False,
        "utc_timestamp": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "local_timestamp": now_utc.astimezone().isoformat(),
        "traffic_source": traffic_res.get("source", "unavailable"),
        "weather_source": weather_res.get("source", "unavailable"),
        "query_lat": round(query_lat, 5),
        "query_lon": round(query_lon, 5),
        "snapped_lat": traffic_res.get("snapped_lat") or "",
        "snapped_lon": traffic_res.get("snapped_lon") or "",
        "segment_fingerprint": seg_fp or "",
        "gate_to_segment_m": traffic_res.get("gate_to_segment_m"),
        "segment_length_m": traffic_res.get("segment_length_m"),
        "road_class": traffic_res.get("road_class") or "",
        "closure_flag": traffic_res.get("road_closure", False),
        "current_speed": traffic_res.get("current_speed"),
        "free_flow_speed": traffic_res.get("free_flow_speed"),
        "speed_ratio": index_res.get("speed_ratio"),
        "current_travel_time": traffic_res.get("current_travel_time"),
        "free_flow_travel_time": traffic_res.get("free_flow_travel_time"),
        "confidence": traffic_res.get("confidence"),
        "low_confidence_flag": index_res.get("low_confidence_flag", False),
        "live_speed_congestion_index": index_res.get("live_speed_congestion_index"),
        "live_index_class": index_res.get("live_index_class") or "",
        "live_state": index_res.get("live_state", "normal"),
        "precip_mm": weather_res.get("precip_mm"),
        "rain_mm": weather_res.get("rain_mm"),
        "wmo_code": weather_res.get("wmo_code"),
        "weather_category": weather_res.get("weather_category") or "",
        "traffic_status": traffic_res.get("status", "ERROR"),
        "traffic_http_status": traffic_res.get("http_status"),
        "weather_status": weather_res.get("status", "ERROR"),
        "note": "; ".join(note_parts),
        "error": "; ".join(err_parts),
    }
