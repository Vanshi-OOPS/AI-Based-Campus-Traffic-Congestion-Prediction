"""Live speed cross-check panel and comparative alert display.

Presents the corridor speed signal from TomTom Flow Segment Data alongside
the Phase 2 ML model prediction without conflating speed rules with ML inference.
"""

from typing import Dict, Any, Optional
import streamlit as st

from src import config as cfg


def render_live_speed_cross_check(
    live_data: Dict[str, Any],
    ml_pred_class: Optional[str] = None,
) -> None:
    """Render the live speed cross-check panel and class comparison."""
    st.subheader("⚡ Live Speed Cross-Check")
    st.caption("Corridor speed signal (13.5 km segment) reported by TomTom • Speed-based rule, not an ML prediction")

    traffic_status = live_data.get("traffic_status", "ERROR")
    live_state = live_data.get("live_state", "normal")

    if traffic_status != "OK" or live_state == "traffic_unavailable":
        st.info("ℹ️ Live traffic speed signal currently unavailable from provider.")
        return

    # Special state: Road closure
    if live_state == "road_closure" or live_data.get("closure_flag") in [True, "True", "true"]:
        st.error("🚫 **Road Closure Reported by TomTom:** Arterial segment is closed to traffic. Live speed index not applicable.")
        return

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)

    curr_speed = live_data.get("current_speed")
    ff_speed = live_data.get("free_flow_speed")
    ratio = live_data.get("speed_ratio")
    idx_val = live_data.get("live_speed_congestion_index")
    live_cls = live_data.get("live_index_class") or "Unknown"

    with c1:
        val_str = f"{curr_speed:.0f} km/h" if curr_speed is not None and curr_speed != "" else "N/A"
        st.metric("Current Speed", val_str, help="Harmonic average speed reported by TomTom")

    with c2:
        val_str = f"{ff_speed:.0f} km/h" if ff_speed is not None and ff_speed != "" else "N/A"
        st.metric("Free-Flow Speed", val_str, help="Reference speed under uncongested conditions reported by TomTom")

    with c3:
        val_str = f"{ratio:.2f}" if ratio is not None and ratio != "" else "N/A"
        st.metric("Speed Ratio", val_str, help="Ratio of current speed to free flow speed")

    with c4:
        val_str = f"{idx_val:.1f} ({live_cls})" if idx_val is not None and idx_val != "" else f"N/A ({live_cls})"
        st.metric("Speed Index", val_str, help="Provisional speed index (0-100), provisional thresholds")

    # Low confidence warning
    if live_state == "low_confidence" or live_data.get("low_confidence_flag") in [True, "True", "true"]:
        st.warning("⚠️ Low confidence traffic reading (TomTom confidence < 0.50). Treat signal with caution.")

    # Cross-check comparison against Phase 2 ML Model
    if ml_pred_class is not None and live_cls in ["Low", "Moderate", "High"]:
        if live_cls == ml_pred_class:
            st.success(
                f"✅ **Signal Alignment:** Both the Phase 2 ML model prediction (`{ml_pred_class}`) "
                f"and the live speed rule (`{live_cls}`) indicate concordant congestion states."
            )
        else:
            st.info(
                f"ℹ️ **Signal Difference (Neutral Note):** Phase 2 ML model predicts `{ml_pred_class}` "
                f"(based on vehicle volume and gate dynamics), while live corridor speed rule indicates `{live_cls}` "
                f"(reflecting regional 13.5 km trunk corridor velocity). These monitor different physical phenomena."
            )


def render_live_alert_banner(live_data: Dict[str, Any]) -> None:
    """Render neutral, speed-rule operational advisory banner in Live Mode."""
    live_state = live_data.get("live_state", "normal")
    live_cls = live_data.get("live_index_class")
    closure = live_data.get("closure_flag") in [True, "True", "true"]

    if closure or live_state == "road_closure":
        st.error(
            "🚨 **Operational Advisory (Road Closure):** Road closure reported on the Main Gate approach corridor. "
            "Consider activating alternative campus transit access routes."
        )
    elif live_cls == "High":
        st.warning(
            "⚠️ **Operational Advisory (Speed-Based Rule):** Slow traffic reported near the Main Gate corridor. "
            "Consider the approved campus traffic-management procedure."
        )
    elif live_cls == "Moderate":
        st.info(
            "ℹ️ **Operational Advisory (Speed-Based Rule):** Moderate speed reduction reported on the approach corridor. "
            "Allow additional transit travel time."
        )
