"""Source status monitoring and badge rendering for Live Mode.

Tracks provider availability, age of data, staleness, and overall status pills
for TomTom Traffic Flow and Open-Meteo Weather feeds.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import zoneinfo
import streamlit as st

from src import config as cfg


def determine_status_pill(traffic_source: str, weather_source: str) -> str:
    """Return the exact status pill text according to the 4 provider combinations."""
    is_traffic_real = traffic_source == "tomtom"
    is_weather_real = weather_source == "open-meteo"

    if is_traffic_real and is_weather_real:
        return "Live (TomTom + Open-Meteo)"
    elif is_weather_real:
        return "Partial live: weather only (traffic not live)"
    elif is_traffic_real:
        return "Partial live: traffic only (weather not live)"
    else:
        return "Demo feed (mock), not live"


def compute_age_minutes(timestamp_dt: datetime, now_dt: Optional[datetime] = None) -> float:
    """Compute elapsed time in minutes from reading timestamp to now."""
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    if timestamp_dt.tzinfo is None:
        # Assume UTC if naive
        timestamp_dt = timestamp_dt.replace(tzinfo=timezone.utc)
    delta_seconds = (now_dt - timestamp_dt).total_seconds()
    return max(0.0, delta_seconds / 60.0)


def is_reading_stale(
    age_minutes: float,
    threshold_minutes: float = cfg.STALENESS_THRESHOLD_MINUTES,
) -> bool:
    """Check if reading exceeds the staleness threshold."""
    return age_minutes > threshold_minutes


def render_source_status_badges(feed_data: Dict[str, Any]) -> None:
    """Render per-source status badges and staleness alerts in Streamlit."""
    traffic_src = feed_data.get("traffic_source", "unavailable")
    weather_src = feed_data.get("weather_source", "unavailable")
    pill_text = determine_status_pill(traffic_src, weather_src)

    # Compute freshness
    utc_str = feed_data.get("utc_timestamp", "")
    age_min = 0.0
    stale = False
    time_label = "Just now"

    if utc_str:
        try:
            ts = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
            age_min = compute_age_minutes(ts)
            stale = is_reading_stale(age_min)
            if age_min < 1.0:
                time_label = "< 1 min ago"
            else:
                time_label = f"{int(round(age_min))} min ago"
        except Exception:
            time_label = "Unknown age"

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        st.markdown(f"**Live Status:** `{pill_text}`")
        if stale:
            st.warning(f"⚠️ Data is STALE ({time_label}). Exceeds {cfg.STALENESS_THRESHOLD_MINUTES} min threshold.")
        else:
            st.caption(f"Last updated: {time_label}")

    with col2:
        t_status = feed_data.get("traffic_status", "ERROR")
        t_icon = "🟢" if t_status == "OK" and traffic_src == "tomtom" else ("🟡" if traffic_src == "mock" else "🔴")
        st.markdown(f"**Traffic:** {t_icon} `{traffic_src}` ({t_status})")
        st.caption("Flow Segment Data (reported by TomTom)")

    with col3:
        w_status = feed_data.get("weather_status", "ERROR")
        w_icon = "🟢" if w_status == "OK" and weather_src == "open-meteo" else ("🟡" if weather_src == "mock" else "🔴")
        st.markdown(f"**Weather:** {w_icon} `{weather_src}` ({w_status})")
        st.caption("Forecast & Current (estimate from Open-Meteo)")
