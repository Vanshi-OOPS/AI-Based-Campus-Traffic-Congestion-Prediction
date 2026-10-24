"""Unit and Mock Tests for Phase 4 Live Collector and Components.

Strict Testing Constraints:
- Zero real network requests permitted during automated tests.
- Verifies exact CSV column counts, non-overlapping WMO table, index formulas,
  and API key scrubbing.
"""

import csv
from pathlib import Path
import pytest
import numpy as np

from src import config as cfg
from src.fetch_traffic import scrub_api_key, haversine_distance_m
from src.fetch_weather import WMO_CODE_MAP, map_wmo_code_to_category
from src.live_index import compute_live_speed_index


def test_sample_live_log_schema_and_alignment():
    """Verify that docs/sample_live_log.ILLUSTRATIVE.csv strictly matches the 35-column schema."""
    sample_path = cfg.DOCS_DIR / "sample_live_log.ILLUSTRATIVE.csv"
    assert sample_path.exists(), "docs/sample_live_log.ILLUSTRATIVE.csv not found"

    with open(sample_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    header = reader[0]
    expected_col_count = 35
    assert len(header) == expected_col_count, f"Header has {len(header)} cols, expected {expected_col_count}"

    # Verify column presence
    assert header[0] == "poll_id"
    assert header[2] == "is_illustrative"
    assert header[7] == "query_lat"
    assert header[8] == "query_lon"
    assert header[11] == "segment_fingerprint"
    assert header[12] == "gate_to_segment_m"
    assert header[13] == "segment_length_m"
    assert header[23] == "live_speed_congestion_index"
    assert header[24] == "live_index_class"
    assert header[25] == "live_state"

    # Verify every data row
    data_rows = reader[1:]
    assert len(data_rows) == 5, f"Expected 5 sample rows, found {len(data_rows)}"

    for i, row in enumerate(data_rows, start=1):
        assert len(row) == expected_col_count, (
            f"Row {i} has {len(row)} columns, expected exactly {expected_col_count}. "
            f"Row content: {row}"
        )
        assert row[2] == "True"  # is_illustrative

    # Specific check for Row 3 (timeout poll): unobserved speeds are blank
    row_3 = data_rows[2]
    assert row_3[5] == "unavailable"  # traffic_source
    assert row_3[16] == ""            # current_speed
    assert row_3[17] == ""            # free_flow_speed
    assert row_3[23] == ""            # live_speed_congestion_index
    assert row_3[24] == ""            # live_index_class
    assert row_3[25] == "traffic_unavailable"  # live_state

    # Specific check for Row 4 (road closure): index and class must be blank
    row_4 = data_rows[3]
    assert row_4[15] == "True"        # closure_flag
    assert row_4[23] == ""            # live_speed_congestion_index blank
    assert row_4[24] == ""            # live_index_class blank
    assert row_4[25] == "road_closure"  # live_state
    assert row_4[33] == "Road closure reported by provider"  # note column
    assert row_4[34] == ""            # error must NOT contain closure text


def test_wmo_code_mapping_non_overlapping():
    """Verify that every WMO code maps to exactly one category without overlap."""
    assert WMO_CODE_MAP[0][0] == "Clear"
    assert WMO_CODE_MAP[1][0] == "Clear"  # Documented okta assumption
    assert WMO_CODE_MAP[2][0] == "Cloudy"
    assert WMO_CODE_MAP[3][0] == "Cloudy"

    # Fog, snow, and hail map to Other/Unknown
    assert WMO_CODE_MAP[45][0] == "Other/Unknown"
    assert WMO_CODE_MAP[48][0] == "Other/Unknown"
    assert WMO_CODE_MAP[71][0] == "Other/Unknown"
    assert WMO_CODE_MAP[96][0] == "Other/Unknown"

    # Rain and Heavy Rain
    assert WMO_CODE_MAP[61][0] == "Rain"
    assert WMO_CODE_MAP[63][0] == "Rain"
    assert WMO_CODE_MAP[65][0] == "Heavy Rain"  # Non-overlapping
    assert WMO_CODE_MAP[82][0] == "Heavy Rain"

    # Test unknown codes
    assert map_wmo_code_to_category(999) == "Other/Unknown"
    assert map_wmo_code_to_category(None) == "Other/Unknown"

    # Test precipitation threshold assumption (5.0 mm in 15-min window elevates Rain to Heavy Rain)
    assert map_wmo_code_to_category(61, precip_mm=1.0) == "Rain"
    assert map_wmo_code_to_category(61, precip_mm=5.0) == "Heavy Rain"
    assert map_wmo_code_to_category(61, precip_mm=7.5) == "Heavy Rain"


def test_live_speed_congestion_index_calculations():
    """Verify live speed index formula and edge cases."""
    # 1. Unimpeded free flow (r = 1.0 -> index = 0.0)
    idx_1 = compute_live_speed_index(50.0, 50.0, road_closure=False, confidence=1.0)
    assert idx_1["speed_ratio"] == 1.0
    assert idx_1["live_speed_congestion_index"] == 0.0
    assert idx_1["live_index_class"] == "Low"
    assert idx_1["live_state"] == "normal"

    # 2. Moderate slowdown (r = 0.8 -> index ≈ 23.53)
    idx_2 = compute_live_speed_index(40.0, 50.0, road_closure=False, confidence=0.9)
    assert idx_2["speed_ratio"] == 0.8
    assert idx_2["live_speed_congestion_index"] == pytest.approx(23.53, abs=0.05)
    assert idx_2["live_index_class"] == "Low"

    # 3. Congestion threshold (r = 0.5 -> index ≈ 58.82)
    idx_3 = compute_live_speed_index(25.0, 50.0, road_closure=False, confidence=0.85)
    assert idx_3["speed_ratio"] == 0.5
    assert idx_3["live_speed_congestion_index"] == pytest.approx(58.82, abs=0.05)
    assert idx_3["live_index_class"] == "Moderate"

    # 4. Severe congestion (r = 0.3 -> index ≈ 82.35)
    idx_4 = compute_live_speed_index(15.0, 50.0, road_closure=False, confidence=0.8)
    assert idx_4["speed_ratio"] == 0.3
    assert idx_4["live_speed_congestion_index"] == pytest.approx(82.35, abs=0.05)
    assert idx_4["live_index_class"] == "High"

    # 5. Standstill (r = 0.0 -> index = 100.0)
    idx_5 = compute_live_speed_index(0.0, 50.0, road_closure=False, confidence=0.7)
    assert idx_5["speed_ratio"] == 0.0
    assert idx_5["live_speed_congestion_index"] == 100.0
    assert idx_5["live_index_class"] == "High"

    # 6. Road Closure: index and class MUST be None (blank)
    idx_closure = compute_live_speed_index(0.0, 50.0, road_closure=True, confidence=1.0)
    assert idx_closure["live_speed_congestion_index"] is None
    assert idx_closure["live_index_class"] is None
    assert idx_closure["live_state"] == "road_closure"
    assert "closure" in idx_closure["note"].lower()

    # 7. Low confidence handling (< 0.50)
    idx_low_conf = compute_live_speed_index(30.0, 50.0, road_closure=False, confidence=0.35)
    assert idx_low_conf["live_state"] == "low_confidence"
    assert idx_low_conf["low_confidence_flag"] is True


def test_api_key_scrubbing():
    """Verify that raw secret keys are scrubbed from query strings and errors."""
    sentinel_key = "SECRET_KEY_XYZ_12345"
    raw_url = f"https://api.tomtom.com/traffic/services/4/flow?point=12.96,79.15&key={sentinel_key}&unit=kmph"
    scrubbed = scrub_api_key(raw_url, sentinel_key)
    assert sentinel_key not in scrubbed
    assert "key=[REDACTED]" in scrubbed

    raw_err = f"Failed to connect using key {sentinel_key} during HTTP request."
    scrubbed_err = scrub_api_key(raw_err, sentinel_key)
    assert sentinel_key not in scrubbed_err
    assert "[REDACTED]" in scrubbed_err


def test_haversine_distance():
    """Verify geodesic distance calculation between two points."""
    # Distance between gate and 100m south offset
    gate_lat, gate_lon = 12.9692232, 79.1559336
    offset_lat, offset_lon = 12.9683232, 79.1559336
    dist = haversine_distance_m(gate_lat, gate_lon, offset_lat, offset_lon)
    assert dist == pytest.approx(100.1, abs=1.0)


def test_atomic_write_and_strict_field_count(tmp_path):
    """Verify that write_record_atomically writes exactly 35 columns and appends correctly."""
    from src.live_collector import write_record_atomically, FIELDNAMES, poll_and_build_record

    test_csv = tmp_path / "test_live_append.csv"
    mock_rec = poll_and_build_record(is_mock=True)

    # 1. First write creates header
    write_record_atomically(mock_rec, test_csv)
    assert test_csv.exists()

    with open(test_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 2  # header + 1 row
    assert len(reader[0]) == len(FIELDNAMES) == 35
    assert len(reader[1]) == len(FIELDNAMES) == 35

    # 2. Second write appends without duplicating header
    mock_rec_2 = poll_and_build_record(is_mock=True)
    write_record_atomically(mock_rec_2, test_csv)

    with open(test_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 3  # header + 2 rows
    assert len(reader[2]) == 35


def test_data_readiness_auditor(tmp_path):
    """Verify data readiness reporting on empty, partial, and sufficient datasets."""
    from src.data_readiness import analyze_live_data_readiness

    # 1. Non-existent file
    res_empty = analyze_live_data_readiness(tmp_path / "nonexistent.csv")
    assert res_empty["status"] == "NO_DATA"
    assert "NOT READY" in res_empty["verdict"]

    # 2. File with a single row
    from src.live_collector import write_record_atomically, poll_and_build_record
    test_csv = tmp_path / "traffic_weather_live.csv"
    rec = poll_and_build_record(is_mock=True)
    rec["is_illustrative"] = "False"  # treat as real for test
    write_record_atomically(rec, test_csv)

    res_single = analyze_live_data_readiness(test_csv)
    assert res_single["status"] == "AUDITED"
    assert res_single["total_rows"] == 1
    assert "NOT READY TO RETRAIN" in res_single["verdict"]


def test_status_pill_combinations():
    """Verify all 4 status pill string combinations."""
    from app.components.source_status import determine_status_pill

    assert determine_status_pill("tomtom", "open-meteo") == "Live (TomTom + Open-Meteo)"
    assert determine_status_pill("unavailable", "open-meteo") == "Partial live: weather only (traffic not live)"
    assert determine_status_pill("tomtom", "unavailable") == "Partial live: traffic only (weather not live)"
    assert determine_status_pill("mock", "mock") == "Demo feed (mock), not live"
    assert determine_status_pill("unavailable", "unavailable") == "Demo feed (mock), not live"


def test_staleness_and_age_computation():
    """Verify data age computation and staleness flagging at 15 minutes."""
    from datetime import datetime, timezone, timedelta
    from app.components.source_status import compute_age_minutes, is_reading_stale

    now = datetime.now(timezone.utc)
    ts_fresh = now - timedelta(minutes=4)
    ts_stale = now - timedelta(minutes=16)

    assert compute_age_minutes(ts_fresh, now) == pytest.approx(4.0, abs=0.1)
    assert not is_reading_stale(compute_age_minutes(ts_fresh, now))

    assert compute_age_minutes(ts_stale, now) == pytest.approx(16.0, abs=0.1)
    assert is_reading_stale(compute_age_minutes(ts_stale, now))


def test_traffic_fetcher_mocked_http_responses(monkeypatch):
    """Verify fetch_tomtom_traffic behavior under all HTTP statuses and edge cases."""
    from src.fetch_traffic import fetch_tomtom_traffic
    import requests

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code

        def json(self):
            return self._json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    # 1. Success 200
    valid_payload = {
        "flowSegmentData": {
            "frc": "FRC2",
            "currentSpeed": 45,
            "freeFlowSpeed": 50,
            "currentTravelTime": 920,
            "freeFlowTravelTime": 900,
            "confidence": 0.95,
            "roadClosure": False,
            "coordinates": {
                "coordinate": [
                    {"latitude": 12.965, "longitude": 79.137},
                    {"latitude": 12.996, "longitude": 79.245},
                ]
            },
        }
    }
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse(valid_payload, 200))
    res = fetch_tomtom_traffic(lat=12.968, lon=79.155, api_key="dummy_key")
    assert res["status"] == "OK"
    assert res["current_speed"] == 45
    assert res["free_flow_speed"] == 50
    assert res["confidence"] == 0.95
    assert res["source"] == "tomtom"

    # 2. HTTP 403 Forbidden
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse({"error": "Forbidden"}, 403))
    res_403 = fetch_tomtom_traffic(lat=12.968, lon=79.155, api_key="dummy_key")
    assert res_403["status"] == "ERROR"
    assert res_403["source"] == "unavailable"
    assert res_403["http_status"] == 403

    # 3. HTTP 429 Rate Limited
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse({"error": "Too Many Requests"}, 429))
    res_429 = fetch_tomtom_traffic(lat=12.968, lon=79.155, api_key="dummy_key")
    assert res_429["status"] == "ERROR"
    assert res_429["source"] == "unavailable"

    # 4. ConnectionError / Timeout
    def mock_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("Request timed out")

    monkeypatch.setattr(requests, "get", mock_timeout)
    res_timeout = fetch_tomtom_traffic(lat=12.968, lon=79.155, api_key="dummy_key")
    assert res_timeout["status"] == "TIMEOUT"
    assert res_timeout["source"] == "unavailable"
    assert "timed out" in res_timeout["error"].lower()

    # 5. Zero freeFlowSpeed handling
    payload_zero_ff = {
        "flowSegmentData": {
            "frc": "FRC2",
            "currentSpeed": 0,
            "freeFlowSpeed": 0,
            "currentTravelTime": 0,
            "freeFlowTravelTime": 0,
            "confidence": 1.0,
            "roadClosure": False,
            "coordinates": {"coordinate": [{"latitude": 12.965, "longitude": 79.137}]},
        }
    }
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse(payload_zero_ff, 200))
    res_zero = fetch_tomtom_traffic(lat=12.968, lon=79.155, api_key="dummy_key")
    assert res_zero["status"] == "OK"
    assert res_zero["free_flow_speed"] == 0.0

    # Index handling of freeFlowSpeed = 0
    from src.live_index import compute_live_speed_index

    idx_zero = compute_live_speed_index(0.0, 0.0, road_closure=False, confidence=1.0)
    assert idx_zero["speed_ratio"] is None
    assert idx_zero["live_speed_congestion_index"] is None
    assert idx_zero["live_state"] == "traffic_unavailable"


def test_weather_fetcher_mocked_responses(monkeypatch):
    """Verify fetch_open_meteo_weather extraction and error handling."""
    from src.fetch_weather import fetch_open_meteo_weather
    import requests

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code

        def json(self):
            return self._json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    # Success
    payload = {
        "current": {
            "precipitation": 0.2,
            "rain": 0.2,
            "weather_code": 51,
            "interval": 900,
        }
    }
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse(payload))
    res = fetch_open_meteo_weather(lat=12.968, lon=79.155)
    assert res["status"] == "OK"
    assert res["precip_mm"] == 0.2
    assert res["wmo_code"] == 51
    assert res["weather_category"] == "Rain"
    assert res["source"] == "open-meteo"


def test_secret_scrubbing_with_sentinel_key():
    """Verify that a sentinel key is scrubbed in all log representations."""
    from src.fetch_traffic import scrub_api_key

    sentinel = "MY_SUPER_SECRET_TOMTOM_KEY_98765"
    test_str = f"GET https://api.tomtom.com/traffic?key={sentinel}&point=12.9,79.1"
    scrubbed = scrub_api_key(test_str, key=sentinel)
    assert sentinel not in scrubbed
    assert "key=[REDACTED]" in scrubbed
