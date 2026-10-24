"""Data and Model Provenance Panel Component.

Displays execution provenance, data mode, model metadata, and environment hashes.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import streamlit as st


def render_provenance_panel(
    run_metadata: Dict[str, Any],
    feature_meta: Dict[str, Any],
    input_source: str = "Manual Form Input (CORE)",
    model_key: str = "Selected Logistic Regression",
    live_feed_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the expandable data provenance and environment audit panel."""
    with st.expander("ℹ️ Data & Model Provenance", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            env_mode = "Hybrid Live Mode" if live_feed_data is not None else "Simulated Prototype"
            st.markdown(f"**Data Environment:** `{env_mode}`")
            st.markdown(f"**Input Source:** `{input_source}`")
            st.markdown(f"**Model In Use:** `{model_key}`")
            st.markdown(f"**Dataset Hash:** `{run_metadata.get('data_hash_processed', 'unknown')}`")
        with c2:
            st.markdown(f"**Inference Timestamp:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
            if live_feed_data is not None:
                t_src = live_feed_data.get("traffic_source", "unavailable")
                w_src = live_feed_data.get("weather_source", "unavailable")
                poll_ts = live_feed_data.get("local_timestamp", "N/A")
                st.markdown(f"**Live Feed Sources:** `Traffic: {t_src} | Weather: {w_src}`")
                st.caption(f"Last Poll: {poll_ts}")
            else:
                st.markdown(f"**Last Data Update:** `N/A (Simulated Mode - No Live Feed)`")
            st.markdown(f"**Random Seed:** `{run_metadata.get('random_seed', 42)}`")
            lib_vers = run_metadata.get("library_versions", {})
            st.caption(f"Stack: sklearn {lib_vers.get('scikit_learn', '')} | xgboost {lib_vers.get('xgboost', '')} | shap {lib_vers.get('shap', '')}")
