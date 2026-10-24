"""TomTom Traffic Flow Segment Data API Client.

Queries TomTom Traffic Flow API for real-time speed, travel time, and functional road class
near specified coordinates. Follows current official TomTom API documentation.

Endpoint:
https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/10/json

Documentation:
https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data

Free-Tier Terms & Limits:
- Free evaluation tier: 2,500 non-tile requests/day (subject to evaluation QPS rate limits).
- Non-commercial evaluation license; no API key or sensitive credentials ever printed or logged.
"""

from typing import Dict, Any, Optional
import os
import re
import time
import math
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

from src import config as cfg

# Load .env locally if present
load_dotenv()


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in meters using Haversine formula."""
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return float(r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def scrub_api_key(text: str, key: Optional[str] = None) -> str:
    """Scrub raw API keys from URLs, error messages, and exception traces."""
    if not text:
        return ""
    scrubbed = re.sub(r"key=[^&\s'\"]+", "key=[REDACTED]", text, flags=re.IGNORECASE)
    if key and len(key) >= 6:
        scrubbed = scrubbed.replace(key, "[REDACTED]")
    return scrubbed


def fetch_tomtom_traffic(
    lat: float = cfg.GATE_COORDINATES[0],
    lon: float = cfg.GATE_COORDINATES[1],
    api_key: Optional[str] = None,
    zoom: int = 10,
    unit: str = "kmph",
    timeout: float = 10.0,
    max_retries: int = 2,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Query TomTom Traffic Flow Segment Data API for a geographic coordinate.

    Parameters
    ----------
    lat : float
        WGS84 latitude.
    lon : float
        WGS84 longitude.
    api_key : Optional[str]
        TomTom developer API key. Defaults to TOMTOM_API_KEY environment variable.
    zoom : int
        Zoom level (0-22). Default 10.
    unit : str
        Speed unit ('kmph' or 'mph'). Default 'kmph'.
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Maximum retries on 5xx server errors or transient timeouts.
    session : Optional[requests.Session]
        Optional reusable HTTP session.

    Returns
    -------
    Dict[str, Any]
        Standardized dictionary containing observed speeds, confidence, road class,
        road closure flag, snapped coordinates, status, and scrubbed error messages.
    """
    key = api_key or os.environ.get("TOMTOM_API_KEY", "").strip()

    fetch_time_utc = datetime.now(timezone.utc).isoformat()

    # Graceful degradation if no API key is configured
    if not key or key == "your_tomtom_api_key_here":
        return {
            "status": "NO_KEY",
            "source": "unavailable",
            "query_lat": float(lat),
            "query_lon": float(lon),
            "current_speed": None,
            "free_flow_speed": None,
            "current_travel_time": None,
            "free_flow_travel_time": None,
            "confidence": None,
            "road_class": "",
            "road_closure": None,
            "snapped_lat": None,
            "snapped_lon": None,
            "segment_fingerprint": None,
            "segment_length_m": None,
            "gate_to_segment_m": None,
            "fetch_time_utc": fetch_time_utc,
            "http_status": None,
            "error": "TOMTOM_API_KEY is not set or configured in .env",
            "attribution": "TomTom Traffic (API key missing)",
        }

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/{zoom}/json"
    params = {
        "point": f"{lat:.5f},{lon:.5f}",
        "unit": unit,
        "key": key,
    }

    client = session or requests

    last_error = ""
    http_status = 0

    for attempt in range(max_retries + 1):
        try:
            resp = client.get(url, params=params, timeout=timeout)
            http_status = resp.status_code

            # 1. Rate Limiting (429)
            if http_status == 429:
                retry_after = resp.headers.get("Retry-After", "5")
                return {
                    "status": "RATE_LIMITED",
                    "source": "unavailable",
                    "query_lat": float(lat),
                    "query_lon": float(lon),
                    "current_speed": None,
                    "free_flow_speed": None,
                    "current_travel_time": None,
                    "free_flow_travel_time": None,
                    "confidence": None,
                    "road_class": "",
                    "road_closure": None,
                    "snapped_lat": None,
                    "snapped_lon": None,
                    "segment_fingerprint": None,
                    "segment_length_m": None,
                    "gate_to_segment_m": None,
                    "fetch_time_utc": fetch_time_utc,
                    "http_status": 429,
                    "error": f"TomTom rate limit reached (HTTP 429). Retry-After: {retry_after}s",
                    "attribution": "TomTom Traffic (rate limited)",
                }

            # 2. Client Errors (401, 403, etc.) - Do not retry
            if 400 <= http_status < 500:
                raw_err = scrub_api_key(resp.text, key)
                return {
                    "status": f"HTTP_{http_status}",
                    "source": "unavailable",
                    "query_lat": float(lat),
                    "query_lon": float(lon),
                    "current_speed": None,
                    "free_flow_speed": None,
                    "current_travel_time": None,
                    "free_flow_travel_time": None,
                    "confidence": None,
                    "road_class": "",
                    "road_closure": None,
                    "snapped_lat": None,
                    "snapped_lon": None,
                    "segment_fingerprint": None,
                    "segment_length_m": None,
                    "gate_to_segment_m": None,
                    "fetch_time_utc": fetch_time_utc,
                    "http_status": http_status,
                    "error": f"TomTom client error (HTTP {http_status}): {raw_err[:120]}",
                    "attribution": "TomTom Traffic (unauthorized/error)",
                }

            # 3. Server Errors (5xx) - Retry with backoff
            if http_status >= 500:
                last_error = f"TomTom server error (HTTP {http_status})"
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue

            resp.raise_for_status()
            data = resp.json()

            # 4. Parse flowSegmentData payload
            flow = data.get("flowSegmentData", {})
            if not flow:
                return {
                    "status": "MALFORMED_RESPONSE",
                    "source": "unavailable",
                    "query_lat": float(lat),
                    "query_lon": float(lon),
                    "current_speed": None,
                    "free_flow_speed": None,
                    "current_travel_time": None,
                    "free_flow_travel_time": None,
                    "confidence": None,
                    "road_class": "",
                    "road_closure": None,
                    "snapped_lat": None,
                    "snapped_lon": None,
                    "segment_length_m": None,
                    "gate_to_segment_m": None,
                    "fetch_time_utc": fetch_time_utc,
                    "http_status": 200,
                    "error": "Missing flowSegmentData object in TomTom response",
                    "attribution": "TomTom Traffic",
                }

            frc = flow.get("frc", "")
            current_spd = flow.get("currentSpeed")
            free_flow_spd = flow.get("freeFlowSpeed")
            curr_time = flow.get("currentTravelTime")
            free_time = flow.get("freeFlowTravelTime")
            confidence = float(flow.get("confidence")) if flow.get("confidence") is not None else None
            road_closure = bool(flow.get("roadClosure")) if flow.get("roadClosure") is not None else None

            # Extract snapped coordinates
            snapped_lat, snapped_lon = None, None
            coords_obj = flow.get("coordinates", {})
            coords_list = coords_obj.get("coordinate", []) if isinstance(coords_obj, dict) else []
            if coords_list and isinstance(coords_list, list) and len(coords_list) > 0:
                snapped_lat = coords_list[0].get("latitude")
                snapped_lon = coords_list[0].get("longitude")

            # Calculate segment length and distance from gate to nearest segment polyline point
            seg_len_m = 0.0
            gate_lat, gate_lon = cfg.GATE_COORDINATES
            min_d_gate = float("inf")
            if coords_list and isinstance(coords_list, list) and len(coords_list) > 0:
                for idx, pt in enumerate(coords_list):
                    p_lat, p_lon = pt.get("latitude"), pt.get("longitude")
                    if p_lat is not None and p_lon is not None:
                        d = haversine_distance_m(gate_lat, gate_lon, p_lat, p_lon)
                        if d < min_d_gate:
                            min_d_gate = d
                        if idx > 0:
                            prev_lat = coords_list[idx - 1].get("latitude")
                            prev_lon = coords_list[idx - 1].get("longitude")
                            if prev_lat is not None and prev_lon is not None:
                                seg_len_m += haversine_distance_m(prev_lat, prev_lon, p_lat, p_lon)
                gate_to_segment_m = round(min_d_gate, 1) if min_d_gate != float("inf") else None
                segment_length_m = round(seg_len_m, 1)
                first_p = coords_list[0]
                last_p = coords_list[-1]
                segment_fingerprint = f"{first_p.get('latitude', 0.0):.4f},{first_p.get('longitude', 0.0):.4f}->{last_p.get('latitude', 0.0):.4f},{last_p.get('longitude', 0.0):.4f}:{segment_length_m:.0f}"
            else:
                gate_to_segment_m = None
                segment_length_m = None
                segment_fingerprint = None

            # Validate speed sanity
            if free_flow_spd is not None and free_flow_spd <= 0:
                last_error = f"Invalid freeFlowSpeed ({free_flow_spd}) returned by provider."

            return {
                "status": "OK",
                "source": "tomtom",
                "query_lat": float(lat),
                "query_lon": float(lon),
                "current_speed": float(current_spd) if current_spd is not None else None,
                "free_flow_speed": float(free_flow_spd) if free_flow_spd is not None else None,
                "current_travel_time": float(curr_time) if curr_time is not None else None,
                "free_flow_travel_time": float(free_time) if free_time is not None else None,
                "confidence": confidence,
                "road_class": frc,
                "road_closure": road_closure,
                "snapped_lat": float(snapped_lat) if snapped_lat is not None else lat,
                "snapped_lon": float(snapped_lon) if snapped_lon is not None else lon,
                "segment_fingerprint": segment_fingerprint,
                "segment_length_m": segment_length_m,
                "gate_to_segment_m": gate_to_segment_m,
                "fetch_time_utc": fetch_time_utc,
                "http_status": 200,
                "error": last_error,
                "attribution": "Real-time traffic flow data reported by TomTom",
            }

        except requests.exceptions.Timeout:
            last_error = f"TomTom request timed out after {timeout}s"
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue
        except Exception as e:
            last_error = scrub_api_key(f"TomTom request failed: {type(e).__name__}: {str(e)}", key)
            if attempt < max_retries:
                time.sleep(1.0 * (attempt + 1))
                continue

    return {
        "status": "TIMEOUT" if "timed out" in last_error else "ERROR",
        "source": "unavailable",
        "query_lat": float(lat),
        "query_lon": float(lon),
        "current_speed": None,
        "free_flow_speed": None,
        "current_travel_time": None,
        "free_flow_travel_time": None,
        "confidence": None,
        "road_class": "",
        "road_closure": None,
        "snapped_lat": None,
        "snapped_lon": None,
        "segment_fingerprint": None,
        "segment_length_m": None,
        "gate_to_segment_m": None,
        "fetch_time_utc": fetch_time_utc,
        "http_status": http_status or None,
        "error": last_error,
        "attribution": "TomTom Traffic (unavailable)",
    }
