"""Open-Meteo Public Weather API Client.

Fetches real-time weather conditions without requiring an API key.
Provides explicit mapping for all WMO 4677 weather interpretation codes
documented by Open-Meteo into the project's categorical weather schema:
'Clear', 'Cloudy', 'Rain', 'Heavy Rain', or 'Unknown'.

Attribution Requirement:
Data provided by Open-Meteo under the Creative Commons Attribution 4.0
International (CC BY 4.0) license (https://open-meteo.com/).
"""

from typing import Dict, Any, Optional
import time
import requests

from src import config as cfg


# Explicit Non-Overlapping WMO Weather Interpretation Code Table (Open-Meteo Documentation)
# Every documented WMO code maps to exactly one category without overlap.
# Fog, freezing precipitation, snow, and hail map to 'Other/Unknown' as they are outside
# the 4 campus transit target categories (Clear, Cloudy, Rain, Heavy Rain).
WMO_CODE_MAP = {
    0: ("Clear", "Clear sky"),
    1: ("Clear", "Mainly clear (Project assumption: cloud cover < 2/8 oktas, minimal solar/road friction effect)"),
    2: ("Cloudy", "Partly cloudy"),
    3: ("Cloudy", "Overcast"),
    45: ("Other/Unknown", "Fog"),
    48: ("Other/Unknown", "Depositing rime fog"),
    51: ("Rain", "Light drizzle"),
    53: ("Rain", "Moderate drizzle"),
    55: ("Rain", "Dense drizzle"),
    56: ("Other/Unknown", "Light freezing drizzle"),
    57: ("Other/Unknown", "Dense freezing drizzle"),
    61: ("Rain", "Slight rain"),
    63: ("Rain", "Moderate rain"),
    65: ("Heavy Rain", "Heavy rain"),
    66: ("Other/Unknown", "Light freezing rain"),
    67: ("Other/Unknown", "Heavy freezing rain"),
    71: ("Other/Unknown", "Slight snow fall"),
    73: ("Other/Unknown", "Moderate snow fall"),
    75: ("Other/Unknown", "Heavy snow fall"),
    77: ("Other/Unknown", "Snow grains"),
    80: ("Rain", "Slight rain showers"),
    81: ("Rain", "Moderate rain showers"),
    82: ("Heavy Rain", "Violent rain showers"),
    85: ("Other/Unknown", "Slight snow showers"),
    86: ("Other/Unknown", "Heavy snow showers"),
    95: ("Rain", "Slight or moderate thunderstorm"),
    96: ("Other/Unknown", "Thunderstorm with slight hail"),
    99: ("Other/Unknown", "Thunderstorm with heavy hail"),
}


def map_wmo_code_to_category(wmo_code: Optional[int], precip_mm: float = 0.0) -> str:
    """Map a WMO weather code and precipitation accumulation to a project category.

    Parameters
    ----------
    wmo_code : Optional[int]
        WMO weather code from Open-Meteo.
    precip_mm : float
        Precipitation accumulation in millimeters over Open-Meteo's 15-minute
        observation window. A threshold of >= 5.0 mm in 15 min (equivalent to an
        hourly downpour rate of ~20 mm/h) elevates 'Rain' to 'Heavy Rain'
        as a documented physical assumption.

    Returns
    -------
    str
        'Clear' | 'Cloudy' | 'Rain' | 'Heavy Rain' | 'Other/Unknown'
    """
    if wmo_code is None or wmo_code not in WMO_CODE_MAP:
        return "Other/Unknown"

    base_cat = WMO_CODE_MAP[wmo_code][0]

    # Elevate Rain to Heavy Rain if precipitation accumulation exceeds 5.0 mm in 15 min
    if base_cat == "Rain" and precip_mm >= cfg.HEAVY_RAIN_PRECIP_THRESHOLD_MM:
        return "Heavy Rain"

    return base_cat


def fetch_open_meteo_weather(
    lat: float = cfg.GATE_COORDINATES[0],
    lon: float = cfg.GATE_COORDINATES[1],
    timeout: float = 8.0,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Query Open-Meteo Current Weather API for a geographic coordinate.

    Parameters
    ----------
    lat : float
        WGS84 latitude.
    lon : float
        WGS84 longitude.
    timeout : float
        Request timeout in seconds.
    session : Optional[requests.Session]
        Optional reusable HTTP session.

    Returns
    -------
    Dict[str, Any]
        Standardized dictionary containing weather observables, mapped category,
        status, and attribution.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "current": "precipitation,rain,weather_code,temperature_2m,wind_speed_10m",
        "timezone": "Asia/Kolkata",
    }

    client = session or requests
    last_err = ""

    for attempt in range(2):
        try:
            response = client.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()

            current = payload.get("current", {})
            precip_mm = float(current.get("precipitation", 0.0) or 0.0)
            rain_mm = float(current.get("rain", 0.0) or 0.0)
            wmo_code = current.get("weather_code")
            wmo_int = int(wmo_code) if wmo_code is not None else None
            temp_c = current.get("temperature_2m")
            wind_kmh = current.get("wind_speed_10m")
            time_iso = current.get("time", "")

            weather_category = map_wmo_code_to_category(wmo_int, precip_mm)

            return {
                "status": "OK",
                "source": "open-meteo",
                "weather_category": weather_category,
                "precip_mm": precip_mm,
                "rain_mm": rain_mm,
                "wmo_code": wmo_int,
                "temperature_c": float(temp_c) if temp_c is not None else None,
                "wind_kmh": float(wind_kmh) if wind_kmh is not None else None,
                "observation_time": time_iso,
                "error": "",
                "attribution": "API-provided weather estimates (Open-Meteo, CC BY 4.0)",
            }
        except requests.exceptions.Timeout:
            last_err = f"Open-Meteo request timed out after {timeout}s"
            if attempt < 1:
                time.sleep(1.0)
                continue
        except Exception as e:
            last_err = f"Open-Meteo query failed: {type(e).__name__}"
            if attempt < 1:
                time.sleep(1.0)
                continue

    return {
        "status": "TIMEOUT" if "timed out" in last_err else "ERROR",
        "source": "unavailable",
        "weather_category": None,
        "precip_mm": None,
        "rain_mm": None,
        "wmo_code": None,
        "temperature_c": None,
        "wind_kmh": None,
        "observation_time": "",
        "error": last_err,
        "attribution": "Open-Meteo (unavailable)",
    }
