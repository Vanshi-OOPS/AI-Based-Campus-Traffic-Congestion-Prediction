# 📖 Data Dictionary & Simulation Physics Specification

This document details the data schemas, physical traffic formulation, ground-truth congestion labeling rules, and feature engineering policies for the **Smart Traffic Congestion Alert** prototype.

---

## 1. Raw Dataset Schema (`data/raw/simulated_traffic_data.csv`)

| Column Name | Type | Unit / Allowed Values | Description |
|---|---|---|---|
| `date` | String | `YYYY-MM-DD` | Calendar date of observation. Grouping identifier. |
| `timestamp` | String | `YYYY-MM-DD HH:MM:SS` | 5-minute interval timestamp (08:00:00 to 10:00:00 inclusive). |
| `day_index` | Integer | `0` to `179` | Sequential day counter since simulation start. |
| `day_of_week` | Integer | `0` to `6` | Day of week index (`0` = Monday, `6` = Sunday). |
| `minute_of_window` | Integer | `0, 5, 10, ..., 120` | Minutes elapsed since window start (0 = 8:00 AM, 120 = 10:00 AM). |
| `vehicle_count` | Integer | Vehicles / 5-min | Realized vehicle arrivals drawn from Poisson distribution. |
| `weather` | String | `Clear`, `Cloudy`, `Rain`, `Heavy Rain` | Morning weather condition (constant throughout each morning). |
| `event_flag` | Integer | `0` or `1` | Campus event indicator (1 = Event morning, 0 = Normal morning). |
| `congestion_score` | Float | `0.0` to `100.0` | Ground-truth congestion score derived from physics formula. |
| `congestion_class` | String | `Low`, `Moderate`, `High` | Categorical congestion level binned from `congestion_score`. |
| `speed_ratio` | Float | `0.15` to `1.00` | Monotone decreasing function of congestion score (schema placeholder for Phase 4 live API). **LEAKAGE COLUMN.** |
| `data_source` | String | `"simulated"` | Source provenance indicator. Declares simulated nature. |

---

## 2. Latent Diagnostics Schema (`data/raw/simulated_traffic_diagnostics.csv`)

> [!WARNING]
> **Data Leakage Notice:** All columns in this file represent internal simulator states. They are recorded exclusively for auditing and model validation and must **NEVER** be provided as features to any predictive model.

| Column Name | Type | Description |
|---|---|---|
| `date` | String | Calendar date. |
| `timestamp` | String | Interval timestamp. |
| `day_index` | Integer | Sequential day index. |
| `effective_capacity` | Float | $C_t$: Maximum vehicles the road segment and gate can process per 5-min interval under current weather and gate friction. |
| `queue_length` | Float | $Q_t$: Latent queue of unserved vehicles waiting upstream at interval $t$. |
| `ar1_noise` | Float | $\epsilon_t$: Realized auto-regressive AR(1) observational variability. |
| `raw_score_pre_clip` | Float | $P_t$: Unclipped continuous congestion score prior to applying $[0, 100]$ bounds. |

---

## 3. Processed Feature Dataset Schema (`data/processed/featured_traffic_data.csv`)

| Column Name | Category | Type | Description |
|---|---|---|---|
| `date` | Metadata | String | Grouping key. |
| `timestamp` | Metadata | String | Timestamp. |
| `day_index` | Metadata | Integer | Sequential day index. |
| `split` | Metadata | String | Chronological assignment: `train` (first 70%), `val` (next 15%), `test` (final 15%). |
| `data_source` | Metadata | String | Always `"simulated"`. |
| `minute_of_window` | Feature | Integer | `0` to `120`. Temporal location in morning rush. |
| `day_of_week` | Feature | Integer | `0` to `6` (Monday=0, Sunday=6). |
| `is_peak_window` | Feature | Binary (`0`/`1`) | `1` if time is within the core arrival rush (8:30–9:15 AM, min 30–75). |
| `vehicle_count` | Feature | Integer | Current 5-minute vehicle arrival count. |
| `rolling_vehicle_avg_15min` | Feature | Float | Mean vehicle count of previous 3 intervals (excluding current row). Initialized via first-interval rule. |
| `vehicle_surge_rate` | Feature | Float | Arrival acceleration: $\text{count}_t - \text{count}_{t-1}$. At $t=0$, set to `0.0`. |
| `weather_ordinal` | Feature | Integer | `Clear` = 0, `Cloudy` = 1, `Rain` = 2, `Heavy Rain` = 3. |
| `weather_Cloudy` | Feature | Binary (`0`/`1`) | One-hot indicator (`1` if Cloudy). |
| `weather_Rain` | Feature | Binary (`0`/`1`) | One-hot indicator (`1` if Rain). |
| `weather_Heavy Rain` | Feature | Binary (`0`/`1`) | One-hot indicator (`1` if Heavy Rain). Baseline = Clear. |
| `event_flag` | Feature | Binary (`0`/`1`) | Scheduled campus event active this morning. |
| `congestion_score` | Target (Nowcast) | Float | Ground-truth score (0 to 100) for regression tasks. |
| `congestion_class` | Target (Nowcast) | String | Ground-truth class (`Low`, `Moderate`, `High`) for classification. |
| `congestion_class_t_plus_15min` | Target (Forecast) | String | Ground-truth class 3 intervals (15 min) ahead. Last 3 rows per day are `NaN`. |
| `speed_ratio` | Audit / Placeholder | Float | Deterministic downstream score function. **LEAKAGE COLUMN.** |
| `weather` | Audit / Display | String | Raw weather string retained for dashboard filters and sanity auditing. |

---

## 4. Physics Formulation & Mathematical Definitions

### A. Demand vs. Capacity vs. Queue
- **Arrival Demand ($V_t$):** Number of vehicles attempting to enter or pass through the campus gate road segment in interval $t$. Realized as $V_t \sim \text{Poisson}(\lambda_t)$.
- **Effective Capacity ($C_t$):** Maximum volume of vehicles that can physically traverse the gate segment in interval $t$.
  $$C_t = \max\left(20.0, \; \text{BASE\_CAPACITY} \times (1 - f_{\text{weather}}) \times (1 - f_{\text{gate\_t}})\right)$$
  - $\text{BASE\_CAPACITY} = 100.0\text{ vehicles / 5-min}$
  - $f_{\text{weather}}$: Capacity reduction friction (`Clear`: 0.00, `Cloudy`: 0.02, `Rain`: 0.15, `Heavy Rain`: 0.35).
  - $f_{\text{gate\_t}}$: Gate security checkpoint throttling. For normal days, $f_{\text{gate\_t}} \sim \mathcal{N}(0, 0.03^2)$ (micro-jitter). On event days, security bag/ID inspections add a $10\%\text{ to }20\%$ bottleneck reduction.
- **Latent Queue ($Q_t$):** Vehicles that arrived but could not be processed during interval $t$ due to demand exceeding capacity:
  $$Q_t = \max(0, \; Q_{t-1} + V_t - C_t)$$
  $Q_t$ resets strictly to $0$ at the start of each morning (8:00 AM).

### B. Congestion Score Ground-Truth Labeling Rule
$$P_t = 65.0 \times \left( \frac{V_t}{C_t} \right) + 25.0 \times \left( \frac{Q_t}{\text{BASE\_CAPACITY}} \right) + \epsilon_t$$
$$\text{congestion\_score}_t = \text{clip}(P_t, \; 0.0, \; 100.0)$$

- **No Double Counting:** Adverse weather and events operate exclusively through physical mechanisms (reducing capacity $C_t$, slowing throughput $f_{\text{gate}}$, and surging demand $V_t$). No artificial additive score penalties are included.
- **Queue Weight ($25.0$):** Sized so that persistent bottleneck queues carry memory across intervals without overpowering instantaneous arrivals.
- **Simulated Observational Variability ($\epsilon_t$):** Auto-regressive AR(1) process with $\phi = 0.50$ and marginal standard deviation $\sigma = 4.5$:
  - Stationary initialization at 8:00 AM ($t=0$): $\epsilon_0 \sim \mathcal{N}(0, 4.5^2)$
  - Innovation for $t > 0$: $\eta_t \sim \mathcal{N}(0, \sigma_\eta^2)$ where $\sigma_\eta = 4.5 \times \sqrt{1 - 0.5^2} \approx 3.8971$
  - $\epsilon_t = 0.50 \, \epsilon_{t-1} + \eta_t$

---

## 5. Congestion Classification Boundaries

> [!NOTE]
> **Prototype-Defined Thresholds:** The boundaries below are prototype-defined engineering cutoffs established for campus transit alerts, rather than validated municipal or highway safety standards.

- **Low ($\text{Score} < 40.0$):** Smooth traffic flow. Volume well below capacity, zero or negligible queuing. Free travel speeds.
- **Moderate ($40.0 \le \text{Score} < 70.0$):** Noticeable slowdown. Flow approaches segment capacity; transient queues form but dissipate within 1–2 cycles.
- **High ($\text{Score} \ge 70.0$):** Critical congestion. Segment is at or beyond capacity; persistent queuing occurs. **Triggers real-time alert banner.**

---

## 6. Speed Ratio Formulation (Phase 4 Schema Alignment)

To maintain schema compatibility with TomTom's `currentSpeed / freeFlowSpeed` ratio:
$$\text{speed\_ratio}_t = \text{clip}\left(1.0 - \left(\frac{\text{congestion\_score}_t}{100.0}\right) \times 0.85, \; 0.15, \; 1.00\right)$$
- At score 0 (empty road): $\text{speed\_ratio} = 1.00$.
- At score 40 (Low/Moderate boundary): $\text{speed\_ratio} = 0.660$.
- At score 70 (High Alert boundary): $\text{speed\_ratio} = 1.0 - 0.70 \times 0.85 = 0.405$.
- At score 100 (complete gridlock): $\text{speed\_ratio} = 0.150$ (minimum crawl speed floor).

---

## 7. First-Interval Feature Initialization Rule (8:00 AM)

At the start of each morning ($t=0$, `minute_of_window = 0`), no prior window telemetry exists:
- `vehicle_surge_rate`: Explicitly initialized to `0.0`.
- `rolling_vehicle_avg_15min`: Falls back to the current observation `vehicle_count`. This is the single documented case where the rolling feature references the current row due to boundary initialization.
- At $t=1$ (minute 5): Uses 1 prior interval (`counts[0]`).
- At $t=2$ (minute 10): Uses 2 prior intervals (`mean(counts[0:2])`).
- At $t \ge 3$ (minute 15+): Strictly averages the preceding 3 intervals (`mean(counts[t-3:t])`), strictly excluding the current row $t$.
