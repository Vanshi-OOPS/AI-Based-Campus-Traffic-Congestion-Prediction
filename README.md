# 🚦 AI-Based Campus Traffic Congestion Prediction & Alert System

An end-to-end AI prototype designed to estimate and predict road segment traffic congestion near a college campus gate during morning arrival hours (**8:00 AM – 10:00 AM**). 

The system classifies congestion levels (**Low**, **Moderate**, **High**), predicts numerical congestion scores (0–100), surfaces the underlying factors driving predictions using explainable AI (SHAP), triggers real-time visual alerts for high congestion, and integrates map-based visualization with zero-budget live traffic feeds.

---

## 📌 Project Overview & Problem Statement
College campuses experience severe congestion bottlenecks around arrival gates during morning peak hours (8:00–10:00 AM). Congestion is governed by dynamic factors including vehicle volume, adverse weather conditions, and scheduled campus/local events.

### Required Deliverables:
- **AI Prediction Engine:** Multi-class classification (Low / Moderate / High) and numeric score regression (0–100).
- **Explainability:** Factor contribution charts (SHAP / Feature Importance) explaining why congestion is expected.
- **Alert Mechanism:** Automatic high-congestion warning banner when the risk threshold is exceeded.
- **Model Evaluation:** Held-out test set comparison against simple baselines (majority class, heuristic threshold rules).
- **Interactive Dashboard:** Streamlit web app with interactive input controls, congestion timeline, and Folium road map.
- **Live Integration Track:** Real-time TomTom Traffic Flow and Open-Meteo weather integration with transparent status labeling.

---

## ⚠️ Data Nature & Explicit Simulation Disclosure

> [!IMPORTANT]
> **All data used in Phase 1 and Phase 2 is SIMULATED.**
> This dataset is synthetically generated using a domain-informed queueing and volume-to-capacity formula. It is created for prototyping, benchmarking, and pedagogical purposes. Real live feeds (TomTom Traffic Flow and Open-Meteo) are layered on top in Phase 4 and are explicitly labeled as "Live" only when active.

### Simulation Assumptions (Assumptions, Not Empirical Ground Truth):
1. **Weather Distribution:** Modeled as 60% Clear, 25% Cloudy, 10% Rain, 5% Heavy Rain. Weather condition persists throughout each morning window.
2. **Campus Events:** Occur on ~10% of mornings, creating a 25%–40% arrival volume surge and a 10%–20% gate throughput restriction due to security checkpoints.
3. **Road Capacity:** Base physical capacity is modeled at 100 vehicles per 5-minute interval, reduced by adverse weather friction (Rain: 15%, Heavy Rain: 35%) and gate security inspections.
4. **Queue Dynamics:** Unserved vehicles accumulate in a latent queue ($Q_t = \max(0, Q_{t-1} + V_t - C_t)$), carrying congestion memory across consecutive 5-minute intervals.
5. **Observational Variability:** Modeled as an AR(1) auto-regressive noise process ($\phi = 0.50$, marginal $\sigma = 4.5$) with stationary initialization.

---

## ⚠️ Known Limitations
- **Synthetic Origin & Metric Interpretation:** High model accuracy or low prediction errors in later phases may partly reflect the machine learning model recovering the known simulator formula rather than real-world traffic behavior.
- **Single Road Segment:** The prototype models only the primary access segment approaching the main gate; grid network spillover to adjoining arterial roads is not modeled.
- **No Real Vehicle Counts from Free Public APIs:** Free traffic APIs (e.g. TomTom) provide speeds and travel times, not physical vehicle counts. Live vehicle counts would require dedicated camera sensors or edge vision.
- **Manual Event Inputs:** Event presence is modeled as a manual administrative toggle or schedule flag rather than automated campus calendar synchronization.

---

## 🏗️ Repository Architecture

```text
smart-traffic-congestion-alert/
├── data/
│   ├── raw/
│   │   ├── simulated_traffic_data.csv        # Public observable features & ground truth
│   │   └── simulated_traffic_diagnostics.csv # Latent state logs (capacity, queue, noise)
│   ├── processed/
│   │   └── featured_traffic_data.csv         # Processed features & chronological splits
│   └── live/                                 # Live collector logs (Phase 4)
├── docs/
│   ├── data_dictionary.md                    # Complete data schema and formulas
│   └── figures/                              # Sanity-check verification plots
├── models/                                   # Serialized model binaries (Phase 2)
├── src/
│   ├── __init__.py
│   ├── config.py                             # Single source of truth (seeds, paths, thresholds)
│   ├── simulate_data.py                      # 180-day realistic morning traffic simulator
│   ├── features.py                           # Per-morning feature engineering pipeline
│   ├── validate_data.py                      # Integrity verification & sanity visualizer
│   └── live_collector.py                     # Zero-budget TomTom & Open-Meteo collector (Phase 4)
├── tests/
│   ├── __init__.py
│   ├── test_simulate_data.py                 # Tests for data generation & physics
│   └── test_features.py                      # Tests for lag isolation & leakage prevention
├── app/                                      # Streamlit dashboard application (Phase 3)
├── requirements.txt                          # Pinned production dependencies
├── README.md                                 # Project documentation
└── train.py                                  # Model training & baseline evaluation (Phase 2)
```

---

## 🛠️ Zero-Budget Tech Stack

| Domain | Technology / Service | Pricing Tier | Purpose |
|---|---|---|---|
| **Language & Environment** | Python 3.10+ | Open-source | Core programming environment |
| **Data Processing & ML** | `pandas`, `numpy`, `scikit-learn`, `xgboost` | Open-source | Simulation, feature engineering, modeling |
| **Model Explainability** | `shap` | Open-source | Per-prediction factor contribution analysis |
| **Interactive Dashboard** | `streamlit` | Open-source | Web UI, alerts, prediction playground |
| **Geospatial Mapping** | `folium`, `streamlit-folium`, OpenStreetMap | Open-source | Road segment congestion coloring |
| **Live Traffic Feed** | TomTom Traffic Flow API | Free Tier (2,500 req/day) | Real-time speed & flow ratio |
| **Live Weather Feed** | Open-Meteo API | Free (No API key needed) | Live precipitation & cloud cover |

---

## 🚀 One-Command Data Pipeline Reproduction

To generate the simulated dataset, engineer features, and run full validation in a single command:

```bash
python src/simulate_data.py && python src/features.py && python src/validate_data.py
```

### Running Automated Tests:
```bash
python -m pytest tests/
```

---

## 📊 Phase 2 Benchmark Results (Held-Out Test Set Only)

> [!NOTE]
> All models were fitted strictly on the 126-day training set, selected on the 27-day validation set, and evaluated **once** on the final 27-day chronological test set (675 unseen observations). Confidence intervals are 95% whole-morning cluster bootstrap CIs (1,000 resamples).

### Classification Benchmark (Nowcast: Low / Moderate / High)
| Model / Baseline | Accuracy | Macro-F1 (95% CI) | High Recall (95% CI) | False High Alerts | Class-Skipping Errors (Low $\leftrightarrow$ High) |
|---|---|---|---|---|---|
| **Majority Class Baseline** | 0.643 | 0.261 [0.24, 0.28] | 0.000 [0.00, 0.00] | 0 | 74 |
| **Historical Time-Slot Baseline** | 0.710 | 0.448 [0.39, 0.51] | 0.000 [0.00, 0.00] | 0 | 16 |
| **Vehicle Count Rule Baseline** | 0.828 | 0.719 [0.65, 0.78] | 0.486 [0.37, 0.63] | 12 | 6 |
| **Count + Weather Rule Baseline** | 0.868 | 0.796 [0.73, 0.87] | 0.554 [0.39, 0.74] | 2 | 3 |
| **Selected ML Model (CORE Features)** | **0.896** | **0.855 [0.82, 0.90]** | **0.960 [0.88, 1.00]** | 30 | 6 |
| **Selected ML Model (FULL Features)** | **0.889** | **0.849 [0.81, 0.89]** | **0.960 [0.88, 1.00]** | 28 | 4 |

### Regression Benchmark (Continuous Congestion Score: 0 – 100)
- **Random Forest (CORE):** $\text{MAE} = 5.14\text{ points} \mid \text{RMSE} = 7.73 \mid R^2 = 0.8916$
- **XGBoost (FULL):** $\text{MAE} = 4.91\text{ points} \mid \text{RMSE} = 6.79 \mid R^2 = 0.9163$

### Key Empirical Findings:
1. **Honest ML Value-Add:** Simple rule baselines (Vehicle Count Rule) already achieve solid accuracy ($82.8\%$), showing that arrival volume is the dominant primary driver. However, Machine Learning provides a **+73% relative boost in High Recall** ($0.960$ vs $0.554$), eliminating safety-critical missed congestion bottlenecks.
2. **CORE vs. FULL Feature Sets:** The CORE feature set (without lag history) performs almost identically to FULL (Macro-F1 $0.855$ vs $0.849$), validating that the manual input form in the Phase 3 dashboard is reliable even when rolling history is unavailable.
3. **Operational High-Alert Calibration:** On the validation set, the decision cutoff was tuned to $p(\text{High}) \ge 0.732$ to target 90% recall. On the held-out test set, this fixed cutoff achieved **91.9% High Recall** (catching 67 of 74 congestion spikes) while keeping the **False Alarm Rate at only 2.2%** (13 false alarms out of 601 non-high intervals).
4. **15-Minute Advance Alerting:** An early warning XGBoost classifier predicted whether High congestion would occur 15 minutes ahead, achieving **$F_1 = 0.671$** ($\text{Precision}=0.681, \text{Recall}=0.662$), heavily outperforming naive persistence ($F_1 = 0.540$) and static count rules ($F_1 = 0.393$).

---

## 🧭 Project Roadmap & Phases
- [x] **Phase 0: Setup & Repository Architecture** (Git repo, folder structure, config, requirements)
- [x] **Phase 1: Realistic Data Simulator & Features** (180 days, peak curves, capacity friction, queue memory, feature isolation, tests & docs)
- [x] **Phase 2: Baseline & Model Training Pipeline** (Time-based split, baseline vs. ML models, metrics, SHAP explanations)
- [ ] **Phase 3: Interactive Dashboard** (Controls, prediction card, alert banner, factor contributions, Folium map)
- [ ] **Phase 4: Zero-Budget Live Integration** (TomTom API + Open-Meteo collectors, live vs simulated badge)
- [ ] **Phase 5: Polish & Documentation** (Demo scripts, benchmark table, presentation materials)

---

## 📄 License
This project is developed for educational and academic research purposes under the MIT License.
