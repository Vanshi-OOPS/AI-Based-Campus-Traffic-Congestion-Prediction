"""Geospatial Road Segment Mock-up Component using Folium.

Renders an illustrative road segment approaching the campus main gate,
dynamically colored Green (Low), Amber (Moderate), or Red (High).

Note: Coordinates and geometry are illustrative mock-ups (placeholders until field verified).
Map tiles require an internet connection to render OpenStreetMap tiles.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import folium

from src import config as cfg


def render_map_component(m: folium.Map, width: int = 380, height: int = 320) -> Any:
    """Render Folium map safely in Streamlit, bypassing custom frontend in headless test mode."""
    if os.environ.get("STREAMLIT_TESTING", "0") == "1":
        return None
    try:
        from streamlit_folium import st_folium
        return st_folium(m, width=width, height=height, returned_objects=[])
    except Exception:
        return None


def create_segment_map(
    predicted_class: str,
    congestion_score: float,
    vehicle_count: int,
    weather: str,
    minute_of_window: int,
) -> folium.Map:
    """Build a Folium map centered on the campus gate with a colored road segment."""
    gate_lat, gate_lon = cfg.GATE_COORDINATES

    # Map styling color by predicted class
    color_map = {
        "Low": "#27ae60",       # Green
        "Moderate": "#f39c12",  # Amber / Orange
        "High": "#e74c3c",      # Red
    }
    segment_color = color_map.get(predicted_class, "#3498db")

    hour = cfg.WINDOW_START_HOUR + (minute_of_window // 60)
    minute = minute_of_window % 60
    time_str = f"{hour:02d}:{minute:02d}"

    # Initialize Folium Map
    m = folium.Map(
        location=[gate_lat, gate_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Road Segment Polyline (illustrative approach road)
    folium.PolyLine(
        locations=cfg.ROAD_SEGMENT_POINTS,
        color=segment_color,
        weight=8,
        opacity=0.85,
        tooltip=f"Gate Road Approach Segment (Status: {predicted_class})",
    ).add_to(m)

    # Gate Marker with Popup Telemetry
    popup_html = f"""
    <div style="font-family: sans-serif; min-width: 180px;">
        <h4 style="margin: 0 0 5px 0; color: #2c3e50;">{cfg.CAMPUS_NAME}</h4>
        <b>Time:</b> {time_str} AM<br/>
        <b>Arrivals:</b> {vehicle_count} veh/5-min<br/>
        <b>Weather:</b> {weather}<br/>
        <hr style="margin: 5px 0;"/>
        <b>Score:</b> {congestion_score:.1f} / 100<br/>
        <b>Status:</b> <span style="color: {segment_color}; font-weight: bold;">{predicted_class}</span><br/>
        <small style="color: gray;">(Simulated Prototype Mock-up)</small>
    </div>
    """

    folium.Marker(
        location=[gate_lat, gate_lon],
        popup=folium.Popup(popup_html, max_width=250),
        tooltip="Campus Main Gate Checkpoint",
        icon=folium.Icon(color="red" if predicted_class == "High" else "blue", icon="info-sign"),
    ).add_to(m)

    return m
