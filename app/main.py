"""Smart Campus Traffic Congestion Alert Dashboard.

Streamlit web application for monitoring and predicting traffic congestion at a college campus gate.
Operates on pre-trained Phase 2 artifacts from models/.
Zero retraining or on-the-fly metric computation on startup.
"""

import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

from src import config as cfg
from src.features import compute_morning_lag_features
from app.components.artifacts import load_all_artifacts, ArtifactValidationError
from app.components.alert_logic import evaluate_operational_alert
from app.components.map_view import create_segment_map, render_map_component
from app.components.factor_chart import (
    render_regressor_factor_chart,
    render_classifier_probability_chart,
)
from app.components.provenance import render_provenance_panel


def main() -> None:
    # 1. Page Configuration & Styling
    st.set_page_config(
        page_title=cfg.APP_TITLE,
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Header & Status Banners
    st.title(f"🚦 {cfg.APP_TITLE}")
    st.markdown(f"*{cfg.APP_SUBTITLE}*")

    h_col1, h_col2, h_col3 = st.columns([2, 2, 2])
    with h_col1:
        st.info(f"**Environment:** `{cfg.APP_STATUS_PILL}`")
    with h_col2:
        st.info(f"**Monitoring Window:** `{cfg.MONITORING_WINDOW_LABEL}`")
    with h_col3:
        st.toggle(
            "Live Traffic & Weather Feed",
            value=False,
            disabled=True,
            help=cfg.LIVE_TOGGLE_DISABLED_LABEL,
        )

    # 3. Load Model Artifacts
    try:
        artifacts = load_all_artifacts()
    except ArtifactValidationError as e:
        st.error(f"⚠️ Artifact Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Unexpected error loading model artifacts: {e}")
        st.stop()

    models = artifacts["models"]
    eval_metrics = artifacts["eval_metrics"]
    run_metadata = artifacts["run_metadata"]
    feature_meta = artifacts["feature_meta"]
    alert_threshold = artifacts["alert_threshold"]

    clf_core = models["clf_core"]
    reg_core = models["reg_core"]
    clf_full = models["clf_full"]
    reg_full = models["reg_full"]
    adv_model = models.get("adv_model")
    explainer = models["explainer"]

    # 4. Main Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Real-Time Prediction & Alert (Nowcast)",
        "📈 Morning Rush Timeline (8:00 - 10:00 AM)",
        "📊 Model Evaluation & Benchmarks",
        "📖 System Methodology & Architecture",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Prediction & Alert (CORE Models)
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("### Interactive Gate Telemetry & Congestion Estimation")
        st.caption("Input current observable traffic conditions to predict gate congestion level and view factor attribution.")

        col_input, col_output = st.columns([1, 2], gap="medium")

        with col_input:
            st.markdown("#### Scenario Inputs (CORE Features)")
            
            # Time slider (8:00 AM to 10:00 AM in 5-min intervals)
            selected_minute = st.slider(
                "Time of Morning",
                min_value=0,
                max_value=120,
                value=50,
                step=5,
                format="%d min",
                help="Minutes elapsed since 8:00 AM (e.g. 50 = 8:50 AM)",
            )
            hour = cfg.WINDOW_START_HOUR + (selected_minute // 60)
            minute = selected_minute % 60
            st.caption(f"Selected Time: **{hour:02d}:{minute:02d} AM**")

            # Day of week
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            selected_day_name = st.selectbox("Day of Week", options=day_names, index=1)
            selected_dow = day_names.index(selected_day_name)

            # Vehicle arrival count
            selected_count = st.slider(
                "Vehicle Arrival Count (per 5 min)",
                min_value=0,
                max_value=140,
                value=75,
                step=1,
                help="Estimated vehicles arriving at the gate approach road in this 5-minute interval",
            )

            # Weather Condition
            selected_weather = st.selectbox(
                "Weather Condition",
                options=cfg.WEATHER_CATEGORIES,
                index=0,
                help="Ambient weather affecting pavement friction and driver speeds",
            )

            # Campus Event Flag
            selected_event = st.checkbox(
                "Campus Event Active (Symposium / Placement / Sports)",
                value=False,
                help="Special campus event inducing inbound traffic surges and checkpoint delays",
            )

            # Optional custom diversion text
            with st.expander("⚙️ Campus Dispatch Route Settings", expanded=False):
                custom_route = st.text_area(
                    "Suggested Diversion Route Text",
                    value=cfg.DEFAULT_DIVERSION_ROUTE_EXAMPLE,
                    help="Custom guidance displayed when a High Alert is triggered.",
                )

        with col_output:
            # Prepare Input DataFrame for CORE model
            core_input = pd.DataFrame([{
                "minute_of_window": selected_minute,
                "day_of_week": selected_dow,
                "vehicle_count": selected_count,
                "weather": selected_weather,
                "event_flag": int(selected_event),
            }])

            # Run Inference
            pred_class = str(clf_core.predict(core_input)[0])
            pred_score = float(reg_core.predict(core_input)[0])
            pred_score = float(np.clip(pred_score, 0.0, 100.0))

            # Probabilities
            probs = clf_core.predict_proba(core_input)[0]
            high_idx = cfg.get_class_index(clf_core, "High")
            mod_idx = cfg.get_class_index(clf_core, "Moderate")
            low_idx = cfg.get_class_index(clf_core, "Low")

            prob_high = float(probs[high_idx])
            prob_mod = float(probs[mod_idx])
            prob_low = float(probs[low_idx])

            # Evaluate Operational Alert Banner (ONE centralized rule)
            alert_res = evaluate_operational_alert(
                predicted_class=pred_class,
                congestion_score=pred_score,
                high_probability=prob_high,
                alert_threshold=alert_threshold,
                custom_diversion_route=custom_route,
            )

            # 1. Operational Alert Banner
            b_type = alert_res["banner_type"]
            msg = alert_res["banner_message"]
            if b_type == "error":
                st.error(f"🚨 **{alert_res['alert_level'].upper()} CONGESTION ALERT**\n\n{msg}")
                if alert_res["diversion_route_note"]:
                    st.info(alert_res["diversion_route_note"])
            elif b_type == "warning":
                st.warning(f"⚠️ **{alert_res['alert_level'].upper()} ADVISORY**\n\n{msg}")
            else:
                st.success(f"✅ **{alert_res['alert_level'].upper()} TRAFFIC FLOW**\n\n{msg}")

            if alert_res["disagreement_note"]:
                st.caption(f"ℹ️ {alert_res['disagreement_note']}")

            # 2. Metric Scorecards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Predicted Status", pred_class)
            with m2:
                st.metric("Congestion Score", f"{pred_score:.1f} / 100")
            with m3:
                st.metric(
                    "P(High)",
                    f"{prob_high:.1%}",
                    delta=f"Cutoff: {alert_threshold:.2f}",
                    delta_color="inverse" if prob_high >= alert_threshold else "off",
                )
            with m4:
                est_speed_ratio = cfg.score_to_speed_ratio(pred_score)
                st.metric(
                    "Est. Speed Ratio",
                    f"{est_speed_ratio:.2f}",
                    help=cfg.SPEED_RATIO_DISCLAIMER,
                )

            # 3. Folium Map & Factor Chart Side-by-Side
            sub_col_map, sub_col_chart = st.columns([1, 1], gap="medium")

            with sub_col_map:
                st.markdown("##### Road Segment Geospatial Mock-up")
                st.caption(cfg.MAP_CAPTION)
                try:
                    seg_map = create_segment_map(
                        predicted_class=alert_res["alert_level"],
                        congestion_score=pred_score,
                        vehicle_count=selected_count,
                        weather=selected_weather,
                        minute_of_window=selected_minute,
                    )
                    render_map_component(seg_map, width=380, height=320)
                except Exception as map_err:
                    st.warning(f"Map rendering offline: {map_err}. (Internet connection required for OpenStreetMap tiles).")

            with sub_col_chart:
                st.markdown("##### Factor Attribution (SHAP Score Points)")
                # Explain using explainer
                exp_dict = {
                    "minute_of_window": selected_minute,
                    "day_of_week": selected_dow,
                    "vehicle_count": selected_count,
                    "weather": selected_weather,
                    "event_flag": int(selected_event),
                }
                explanation = explainer.explain_prediction(exp_dict, mode="regressor")
                fig_factor = render_regressor_factor_chart(explanation)
                st.pyplot(fig_factor)
                plt.close(fig_factor)

            # Provenance panel at bottom of Tab 1
            render_provenance_panel(
                run_metadata=run_metadata,
                feature_meta=feature_meta,
                input_source="Manual Form Input (CORE)",
                model_key=feature_meta.get("selected_classifier_core", "logistic_regression_CORE"),
            )

    # -------------------------------------------------------------------------
    # TAB 2: Morning Timeline Simulator (FULL Models)
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### 8:00 AM – 10:00 AM Chronological Rush Timeline")
        st.caption("Sequential 25-step simulation incorporating historical rolling averages and arrival surge rates.")

        # Load processed test data for selection
        raw_df = pd.read_csv(cfg.PROCESSED_DATA_PATH)
        test_df = raw_df[raw_df["split"] == "test"].copy()
        test_dates = test_df["date"].unique().tolist()

        t2_mode = st.radio(
            "Timeline Source Mode",
            options=["Inspect Held-Out Test Morning", "Custom Morning Scenario Generator"],
            horizontal=True,
        )

        if t2_mode == "Inspect Held-Out Test Morning":
            selected_date = st.selectbox("Select Test Date (Unseen Held-Out Split)", options=test_dates, index=0)
            morning_df = test_df[test_df["date"] == selected_date].sort_values("minute_of_window").copy().reset_index(drop=True)
            has_ground_truth = True
        else:
            has_ground_truth = False
            sc_c1, sc_c2 = st.columns(2)
            with sc_c1:
                sc_weather = st.selectbox("Scenario Weather", options=cfg.WEATHER_CATEGORIES, index=2)
            with sc_c2:
                sc_event = st.checkbox("Scenario Campus Event Active", value=True)

            # Build synthetic 25-step sequence using true features.py function
            counts_base = [
                30, 32, 35, 42, 50, 65, 78, 92, 105, 112, 108, 95, 80,
                70, 60, 52, 45, 40, 38, 35, 32, 30, 28, 26, 25
            ]
            if sc_event:
                counts_base = [int(c * 1.3) for c in counts_base]

            morning_records = []
            for idx, count in enumerate(counts_base):
                m_win = idx * 5
                h = cfg.WINDOW_START_HOUR + (m_win // 60)
                m = m_win % 60
                morning_records.append({
                    "date": "2026-Scenario",
                    "timestamp": f"2026-Scenario {h:02d}:{m:02d}:00",
                    "minute_of_window": m_win,
                    "day_of_week": 1,
                    "vehicle_count": count,
                    "weather": sc_weather,
                    "event_flag": int(sc_event),
                    "congestion_class": "Low",  # placeholder
                })
            raw_scenario = pd.DataFrame(morning_records)
            morning_df = compute_morning_lag_features(raw_scenario)

        # Run FULL Model Inference on all 25 intervals
        full_preds_class = clf_full.predict(morning_df)
        full_preds_score = reg_full.predict(morning_df)
        full_probs = clf_full.predict_proba(morning_df)
        full_high_idx = cfg.get_class_index(clf_full, "High")
        full_high_probs = full_probs[:, full_high_idx]

        morning_df["pred_class"] = full_preds_class
        morning_df["pred_score"] = np.round(full_preds_score, 1)
        morning_df["prob_high"] = np.round(full_high_probs, 3)

        # 15-Minute Advance Alert if model exists
        if adv_model is not None:
            # Advance model requires full features
            adv_probs = adv_model.predict_proba(morning_df)[:, 1]
            morning_df["adv_high_soon_prob"] = np.round(adv_probs, 3)
            # Last 3 rows have no forward horizon
            morning_df.loc[22:24, "adv_high_soon_prob"] = np.nan

        # Plot Arrival Curve & Score Timeline
        fig_time, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

        # Plot 1: Arrival Counts
        mins = morning_df["minute_of_window"].values
        counts = morning_df["vehicle_count"].values
        ax1.bar(mins, counts, width=3.5, color="#3498db", alpha=0.85, label="Arrival Count (veh / 5-min)")
        ax1.axvspan(30, 75, color="orange", alpha=0.15, label="Morning Rush Peak (8:30 - 9:15 AM)")
        ax1.set_ylabel("Arrivals / 5-min", fontweight="bold")
        ax1.set_title("5-Minute Vehicle Arrival Demand Across Morning Window", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper left")

        # Plot 2: Congestion Score Timeline
        pred_scores = morning_df["pred_score"].values
        ax2.plot(mins, pred_scores, "r-s", lw=2, label="Predicted Score (ML FULL)")
        if has_ground_truth:
            actual_scores = morning_df["congestion_score"].values
            ax2.plot(mins, actual_scores, "b--o", lw=1.5, alpha=0.8, label="Actual Ground Truth")

        ax2.axhline(cfg.SCORE_LOW_MAX, color="green", linestyle="--", alpha=0.7, label="Low Threshold (40)")
        ax2.axhline(cfg.SCORE_MODERATE_MAX, color="red", linestyle="--", alpha=0.7, label="High Alert Threshold (70)")
        ax2.set_ylabel("Score (0 - 100)", fontweight="bold")
        ax2.set_xlabel("Minute of Window (0 = 8:00 AM, 60 = 9:00 AM, 120 = 10:00 AM)", fontweight="bold")
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="upper left")

        fig_time.tight_layout()
        st.pyplot(fig_time)
        plt.close(fig_time)

        # Timeline Scrubber for Step-by-Step Inspection
        st.markdown("#### Scrubber: Inspect Individual 5-Minute Interval")
        scrub_idx = st.slider("Step Index (0 to 24)", min_value=0, max_value=24, value=10, step=1)
        step_row = morning_df.iloc[scrub_idx]

        h_step = cfg.WINDOW_START_HOUR + (int(step_row["minute_of_window"]) // 60)
        m_step = int(step_row["minute_of_window"]) % 60
        st.markdown(f"**Selected Time:** `{h_step:02d}:{m_step:02d} AM` | "
                    f"Arrivals: `{int(step_row['vehicle_count'])}` | "
                    f"Rolling 15-min Avg: `{step_row['rolling_vehicle_avg_15min']:.1f}` | "
                    f"Surge Rate: `{step_row['vehicle_surge_rate']:+.1f}`")

        # Step Alert Banner using same central logic
        step_alert = evaluate_operational_alert(
            predicted_class=str(step_row["pred_class"]),
            congestion_score=float(step_row["pred_score"]),
            high_probability=float(step_row["prob_high"]),
            alert_threshold=alert_threshold,
        )
        if step_alert["banner_type"] == "error":
            st.error(f"🚨 **{step_alert['alert_level']} Alert:** {step_alert['banner_message']}")
        elif step_alert["banner_type"] == "warning":
            st.warning(f"⚠️ **{step_alert['alert_level']} Alert:** {step_alert['banner_message']}")
        else:
            st.success(f"✅ **{step_alert['alert_level']}:** {step_alert['banner_message']}")

        if adv_model is not None:
            adv_val = step_row["adv_high_soon_prob"]
            if pd.isna(adv_val):
                st.caption("15-Min Advance Forecast: `n/a (no forecast horizon for morning tail)`")
            else:
                adv_flag = "⚠️ High Expected Soon!" if adv_val >= 0.35 else "Flow Stable"
                st.caption(f"15-Min Advance Forecast: P(High at t+15min) = **{adv_val:.1%}** — `{adv_flag}`")

    # -------------------------------------------------------------------------
    # TAB 3: Model Benchmark & Recorded Evaluation Hub
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### Measured Model Performance (Recorded Results Only)")
        st.caption("Results evaluated on the unseen held-out chronological test set (27 mornings, 675 rows). No live re-computation on startup.")

        # Limitation disclaimer
        st.warning(
            "⚠️ **Methodology & Limitation Notice:** "
            "All metrics below reflect performance in the domain-informed simulated environment. "
            "Because labels originate from the simulator formula, high scores demonstrate that the ML model effectively "
            "learns physical arrival and queue dynamics, but they do NOT constitute empirical validation on physical campus sensors."
        )

        st.markdown("#### 1. Test Set Benchmark Table (Baselines vs. ML Models)")
        
        # Build Table from recorded eval_metrics.json
        bench_rows = []
        for b_name, m in eval_metrics.get("baselines", {}).items():
            f1_ci = f"{m['macro_f1']:.3f} [{m['macro_f1_ci95'][0]:.2f}, {m['macro_f1_ci95'][1]:.2f}]"
            rec_ci = f"{m['per_class']['High']['recall']:.3f} [{m['high_recall_ci95'][0]:.2f}, {m['high_recall_ci95'][1]:.2f}]"
            bench_rows.append({
                "Model / Baseline": f"{b_name} (Baseline)",
                "Accuracy": f"{m['accuracy']:.3f}",
                "Macro-F1 (95% CI)": f1_ci,
                "High Recall (95% CI)": rec_ci,
                "False High Alerts": m["false_high_alerts"],
                "Class-Skipping Errors": m["class_skipping"]["total_skipping"],
            })

        for k, m in eval_metrics.get("classifiers", {}).items():
            label = "Selected " + k.replace("_test", "").replace("_", " ").title()
            f1_ci = f"{m['macro_f1']:.3f} [{m['macro_f1_ci95'][0]:.2f}, {m['macro_f1_ci95'][1]:.2f}]"
            rec_ci = f"{m['per_class']['High']['recall']:.3f} [{m['high_recall_ci95'][0]:.2f}, {m['high_recall_ci95'][1]:.2f}]"
            bench_rows.append({
                "Model / Baseline": label,
                "Accuracy": f"{m['accuracy']:.3f}",
                "Macro-F1 (95% CI)": f1_ci,
                "High Recall (95% CI)": rec_ci,
                "False High Alerts": m["false_high_alerts"],
                "Class-Skipping Errors": m["class_skipping"]["total_skipping"],
            })

        st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)

        st.info(
            "💡 **Baseline Comparison Insight:** "
            "The heuristic Vehicle Count Rule is already solid (Accuracy: 0.828), showing that vehicle volume is the primary factor. "
            "However, the ML models provide a **+73% relative gain in High Recall** (0.960 vs. 0.554), preventing missed traffic bottlenecks."
        )

        st.markdown("#### 2. Operational Alert Threshold Calibration")
        calib = eval_metrics.get("alert_calibration", {})
        test_calib = eval_metrics.get("operational_alert_test", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Validation-Chosen Cutoff", f"p >= {calib.get('chosen_threshold', 0.5):.3f}")
        c2.metric("Validation High Recall", f"{calib.get('val_high_recall', 0):.1%}")
        c3.metric("TEST High Recall", f"{test_calib.get('test_high_recall', 0):.1%}")
        c4.metric("TEST False Alarm Rate", f"{test_calib.get('test_false_alarm_rate', 0):.1%}")

        st.markdown("#### 3. Condition Slices (Weather & Events on TEST Set)")
        slice_rows = []
        for w, s in eval_metrics.get("slices", {}).get("weather", {}).items():
            slice_rows.append({
                "Condition": f"Weather: {w}",
                "Mornings": s["mornings"],
                "Intervals": s["intervals"],
                "Sample Status": s["status"],
                "Accuracy": f"{s['accuracy']:.3f}",
                "High Recall": f"{s['high_recall']:.3f}",
            })
        for ev, s in eval_metrics.get("slices", {}).get("event", {}).items():
            slice_rows.append({
                "Condition": f"Event: {ev}",
                "Mornings": s["mornings"],
                "Intervals": s["intervals"],
                "Sample Status": s["status"],
                "Accuracy": f"{s['accuracy']:.3f}",
                "High Recall": f"{s['high_recall']:.3f}",
            })
        st.dataframe(pd.DataFrame(slice_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 4. 15-Minute Advance Alert Task (Early Warning)")
        adv_res = eval_metrics.get("advance_15min_alert", {})
        adv_rows = []
        for k, m in adv_res.items():
            if isinstance(m, dict):
                adv_rows.append({
                    "Model / Strategy": k.replace("_", " ").title(),
                    "Accuracy": f"{m['accuracy']:.3f}",
                    "Precision": f"{m['precision']:.3f}",
                    "Recall": f"{m['recall']:.3f}",
                    "F1-Score": f"{m['f1']:.3f}",
                    "False Alerts": m["false_alerts"],
                })
        st.dataframe(pd.DataFrame(adv_rows), use_container_width=True, hide_index=True)

        # Embedded Figures Gallery
        st.markdown("#### 5. Publication Figures Gallery")
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            cm_path = cfg.DOCS_FIGURES_DIR / "confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix: Baseline vs Selected ML Model")
            alert_curve_path = cfg.DOCS_FIGURES_DIR / "alert_threshold_curve.png"
            if alert_curve_path.exists():
                st.image(str(alert_curve_path), caption="Operational Threshold Calibration Curve (Validation)")
        with fig_col2:
            shap_path = cfg.DOCS_FIGURES_DIR / "shap_summary.png"
            if shap_path.exists():
                st.image(str(shap_path), caption="Global SHAP Summary Plot (Regressor Score Points)")
            cf_path = cfg.DOCS_FIGURES_DIR / "core_vs_full_comparison.png"
            if cf_path.exists():
                st.image(str(cf_path), caption="Value of Lag Features: CORE vs FULL Comparison")

    # -------------------------------------------------------------------------
    # TAB 4: System Methodology & Architecture
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("### System Methodology & Physics Formulation")
        
        st.markdown("""
        #### Physics-Informed Congestion Modeling
        Congestion is formulated as a volume-to-effective-capacity ($V/C$) function with upstream queue accumulation:
        
        $$\\text{Effective Capacity } C_t = \\max\\left(20, \\; \\text{BASE\\_CAPACITY} \\times (1 - f_{\\text{weather}}) \\times (1 - f_{\\text{gate\\_t}})\\right)$$
        
        $$\\text{Latent Queue } Q_t = \\max(0, \\; Q_{t-1} + V_t - C_t)$$
        
        $$\\text{Congestion Score } P_t = 65.0 \\cdot \\left(\\frac{V_t}{C_t}\\right) + 25.0 \\cdot \\left(\\frac{Q_t}{\\text{BASE\\_CAPACITY}}\\right) + \\epsilon_t$$
        
        Where:
        - **$f_{\\text{weather}}$:** Weather capacity friction (Clear: 0%, Cloudy: 2%, Rain: 15%, Heavy Rain: 35%).
        - **$f_{\\text{gate\\_t}}$:** Gate security inspection throttling (10–20% on event mornings).
        - **$\\epsilon_t$:** Auto-regressive $\\text{AR}(1)$ observational variability ($\\phi = 0.50, \\sigma = 4.5$).
        
        #### Classification Thresholds
        - **Low:** $\\text{Score} < 40.0$ (Free flow, minimal queuing)
        - **Moderate:** $40.0 \\le \\text{Score} < 70.0$ (Transient queues, slower approach)
        - **High:** $\\text{Score} \\ge 70.0$ (Critical bottleneck, triggers operational alert)
        
        *Note: These are prototype-defined operational boundaries, not statutory highway safety standards.*
        """)

        st.markdown("#### Zero-Budget Tech Stack")
        stack_data = [
            {"Role": "Language & Environment", "Tool": "Python 3.14 + virtualenv", "Cost Tier": "Open-Source Free"},
            {"Role": "Simulation & ML", "Tool": "pandas, scikit-learn, XGBoost", "Cost Tier": "Open-Source Free"},
            {"Role": "Model Explainability", "Tool": "SHAP (TreeExplainer)", "Cost Tier": "Open-Source Free"},
            {"Role": "Interactive UI", "Tool": "Streamlit", "Cost Tier": "Open-Source Free"},
            {"Role": "Geospatial Mock-up", "Tool": "Folium + OpenStreetMap", "Cost Tier": "Open-Source Free"},
            {"Role": "Live Traffic (Phase 4)", "Tool": "TomTom Flow API", "Cost Tier": "Free Tier (2,500 req/day)"},
            {"Role": "Live Weather (Phase 4)", "Tool": "Open-Meteo API", "Cost Tier": "Free (No Key Required)"},
        ]
        st.dataframe(pd.DataFrame(stack_data), use_container_width=True, hide_index=True)

        st.markdown("For full column descriptions and mathematical derivations, see [docs/data_dictionary.md](file:///d:/disk%20d/project/Smart%20Traffic%20Congestion%20Alert/docs/data_dictionary.md).")

    # 5. Global Footer
    st.markdown("---")
    f_c1, f_c2, f_c3 = st.columns(3)
    with f_c1:
        st.caption(f"Random Seed: `{run_metadata.get('random_seed', 42)}`")
    with f_c2:
        st.caption(f"Processed Dataset Hash: `{run_metadata.get('data_hash_processed', 'unknown')}`")
    with f_c3:
        st.caption(f"Model Training Timestamp: `{run_metadata.get('timestamp', 'unknown')[:19]}`")


if __name__ == "__main__":
    main()
