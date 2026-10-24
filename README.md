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

> **Data Disclosure:** The core prototype uses a realistically modeled, physics-informed simulated dataset (accounting for arrival curves, weather friction, event surges, and stochastic noise). Real live feeds are layered on top via zero-cost public APIs.

---

## 🏗️ Repository Architecture

```text
AI-Based-Campus-Traffic-Congestion-Prediction/
├── .gitignore               # Ignored binaries, env keys, caches, datasets
├── requirements.txt         # Pinned production dependencies
├── README.md                # Project documentation & runbook
├── data/
│   ├── raw/                 # Raw simulated & historical data
│   ├── processed/           # Feature-engineered training/validation/test sets
│   └── live/                # Live collector logs (CSV/SQLite)
├── models/                  # Serialized trained model binaries & metrics
├── src/
│   ├── __init__.py
│   ├── config.py            # Central configurations, seeds, paths & thresholds
│   ├── simulate_data.py     # Realistic 2-3 month morning traffic data generator
│   ├── features.py          # Time-based feature engineering & transformations
│   ├── train.py             # Reproducible training pipeline & baseline comparisons
│   ├── live_collector.py    # Zero-budget TomTom & Open-Meteo live data collector
│   └── utils.py             # Evaluation metrics and helper routines
└── app/
    ├── __init__.py
    ├── main.py              # Streamlit dashboard application
    └── components/
        ├── map_view.py      # Road segment Folium map (Green/Amber/Red)
        └── factor_chart.py  # SHAP factor contribution breakdown
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

## 🚀 Getting Started

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Vanshi-OOPS/AI-Based-Campus-Traffic-Congestion-Prediction.git
cd AI-Based-Campus-Traffic-Congestion-Prediction

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Simulated Dataset
```bash
python src/simulate_data.py
```

### 3. Train Models & Evaluate Against Baselines
```bash
python src/train.py
```

### 4. Launch the Dashboard
```bash
streamlit run app/main.py
```

---

## 🧭 Project Roadmap & Phases
- [x] **Phase 0: Setup & Repository Architecture** (Git repo, folder structure, config, requirements)
- [ ] **Phase 1: Realistic Data Simulator** (3 months of 5-min intervals, peak curves, weather friction, event surges)
- [ ] **Phase 2: Baseline & Model Training Pipeline** (Time-based split, baseline vs. ML models, metrics, SHAP explanations)
- [ ] **Phase 3: Interactive Dashboard** (Controls, prediction card, alert banner, factor contributions, Folium map)
- [ ] **Phase 4: Zero-Budget Live Integration** (TomTom API + Open-Meteo collectors, live vs simulated badge)
- [ ] **Phase 5: Polish & Documentation** (Demo scripts, benchmark table, limitations & future work)

---

## 📄 License
This project is developed for educational and academic research purposes under the MIT License.
