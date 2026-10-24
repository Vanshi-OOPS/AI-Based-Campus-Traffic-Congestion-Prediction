# Phase 4 Live Track: Real-Time Traffic & Weather Data Integration

## 1. Executive Summary & Design Principles

Phase 4 introduces live external data ingestion to the Smart Campus Traffic Congestion Alert project, augmenting the Phase 2/3 simulated ML prototype with real-world sensor and meteorological telemetry under a strict zero-budget constraint (free tiers only).

### Core Principles
1. **Verifiable Grounding & No Mock Infiltration:** Real API feeds and mock feeds are permanently segregated. Real readings write strictly to `data/live/traffic_weather_live.csv`; mock demo readings write to `data/live/mock_feed_demo.csv` with `is_illustrative=True` and `traffic_source="mock"`.
2. **Speed vs. Count ML Separation:** Live traffic APIs provide corridor speeds, travel times, and delays—not vehicle count headcounts. Because the Phase 2 ML classifier was trained on gate-level vehicle counts, **the live speed signal does not feed into or retrain the Phase 2 ML model**. In the UI, the live speed rule and ML inference remain strictly independent displays and never share a meter.
3. **Conservative Terminology:** 
   - Live traffic metrics are explicitly labeled **`corridor speed signal (13.5 km segment)`**, never "gate measurement".
   - Speeds and travel times are attributed as **`reported by TomTom`**.
   - Weather observations are attributed as **`estimate from Open-Meteo`**.
   - Neither feed is described as "measured at the gate".
4. **Credential Security:** Real API keys reside exclusively in the local, gitignored `.env` file. Keys are systematically scrubbed from logs, URLs, error messages, terminal output, and committed documentation.
5. **Best-Effort Audit Trail:** Data logging implements an atomic, schema-validated append with audit metrics for missing polling intervals, provider outages, and carriageway fingerprint stability.

---

## 2. API Contracts & Documentation

### 2.1 TomTom Traffic Flow API (Flow Segment Data)
- **Official Documentation:** [TomTom Traffic Flow API - Flow Segment Data](https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data)
- **Endpoint Pattern:**
  ```http
  GET https://api.tomtom.com/traffic/services/4/flowSegmentData/{style}/{zoom}/json?point={lat}%2C{lon}&unit=kmph&key={API_KEY}
  ```
- **Selected Parameters:**
  - `style`: `relative` (speed relative to free flow).
  - `zoom`: `18` (highest street-level detail supported by the endpoint).
  - `point`: Query latitude and longitude (`12.9683232,79.1559336`).
  - `unit`: `kmph`.
- **Returned Schema Elements (`flowSegmentData`):**
  - `frc`: Functional Road Class string (e.g., `FRC2` for major regional arterial).
  - `currentSpeed`: Current harmonic average speed along the segment (integer km/h).
  - `freeFlowSpeed`: Speed expected under unimpeded conditions (integer km/h).
  - `currentTravelTime`: Current travel time across the full segment length (seconds).
  - `freeFlowTravelTime`: Travel time under free-flow conditions (seconds).
  - `confidence`: Floating point value in $[0.0, 1.0]$ representing road coverage and data source reliability.
  - `roadClosure`: Boolean flag indicating whether the segment is closed to general traffic.
  - `coordinates.coordinate`: Array of `{latitude, longitude}` vertices tracing the mapped polyline.

### 2.2 Open-Meteo Public Weather API
- **Official Documentation:** [Open-Meteo Weather Forecast & Current APIs](https://open-meteo.com/en/docs)
- **Endpoint Pattern:**
  ```http
  GET https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,rain,weather_code&timezone=auto
  ```
- **Selected Parameters:**
  - `latitude`, `longitude`: Target coordinates (`12.9683232, 79.1559336`).
  - `current`: Comma-separated variables: `precipitation`, `rain`, `weather_code`.
  - `timezone`: `auto` (resolves to `Asia/Kolkata`).
- **Precipitation Window Citation:**
  - As defined in the official Open-Meteo API specifications (`current.interval = 900`), the `precipitation` and `rain` variables report the **15-minute rolling accumulation (in mm)** preceding the query timestamp, not an instantaneous rate.

---

## 3. Free-Tier Terms, Limits, and Attribution

| Provider | Free Tier Allocation | Terms of Use & Data Storage | Required Attribution |
| :--- | :--- | :--- | :--- |
| **TomTom** | 2,500 non-tile requests / day (evaluation account, subject to standard QPS caps) | Evaluation and non-commercial educational prototyping. Persistent caching of live dynamic speeds is restricted; historical audit logs stored for non-commercial thesis/research benchmarking. | *"Traffic data reported by TomTom"* |
| **Open-Meteo** | Free for non-commercial use up to 10,000 requests / day (no API key required) | Open Database License / Creative Commons Attribution 4.0 International (CC BY 4.0). Permitted for research and non-commercial storage. | *"Weather estimate from Open-Meteo (CC BY 4.0)"* |

### 3.1 Obtaining a Free TomTom API Key
1. Navigate to the [TomTom Developer Portal](https://developer.tomtom.com/).
2. Create a free Developer Account.
3. In the Developer Dashboard, navigate to **My Apps** > **Add a new app**.
4. Enable the **Traffic API** product.
5. Copy the generated Consumer API Key.
6. Create or edit `.env` in the repository root (never commit this file):
   ```bash
   TOMTOM_API_KEY=your_actual_tomtom_api_key_here
   ```

---

## 4. Geospatial Road Snapping & Inbound Direction Analysis

### 4.1 Target Campus Gate vs. Regional Arterial
- **Campus Main Gate Coordinates:** `12.9692232° N, 79.1559336° E` (field-verified).
- **Road Class Coverage Reality:**
  - Querying at the gate or along adjacent internal roads (FRC4–FRC6) returns either HTTP 400 (`Point too far from nearest existing segment`) or snaps to the major parallel regional arterial (`FRC2`).
  - **Careful Claim:** *No finer segment returned near the gate for points tested*. The TomTom Flow Segment Data API models this corridor as a continuous $13.52\text{ km}$ regional trunk segment passing within $121.1\text{ m}$ of the gate.
  - Alternative endpoints investigated:
    - *TomTom Routing API (`calculateRoute`):* Returns `HTTP 403 Forbidden` on free evaluation tier keys lacking Routing product provisioning.
    - *TomTom Vector Flow Tiles:* Returns raw vector protobuf tiles with color codes, but does not provide numeric speed or travel time metrics in km/h.

### 4.2 Carriageway Direction & Inbound Determination
The regional arterial consists of divided dual carriageways:
1. **Query at Gate (`12.9692232, 79.1559336`):**
   - Snapped start $\rightarrow$ end bearing: $253.7^\circ$ (local tangent $270.3^\circ$, Westbound toward railway station/city center).
   - Gate offset along polyline: $11.37\text{ km}$ from start, $2.14\text{ km}$ from end.
2. **Query 100 m South (`12.9683232, 79.1559336`):**
   - Snapped start $\rightarrow$ end bearing: $73.7^\circ$ (local tangent $89.3^\circ$, Eastbound toward campus arrival).
   - Gate offset along polyline: $2.14\text{ km}$ from start, $11.37\text{ km}$ from end.
   - Nearest polyline vertex: $121.1\text{ m}$ from campus main gate.

**Selected Pinned Query Point:** `(12.9683232, 79.1559336)` pins the Eastbound inbound carriageway, measuring traffic entering the campus corridor during morning arrival hours.

### 4.3 Segment Fingerprinting
To detect upstream TomTom map graph updates or silent carriageway flipping, each poll calculates an invariant fingerprint:
$$\text{Fingerprint} = \text{StartLat,StartLon}\rightarrow\text{EndLat,EndLon}:\text{LengthMeters}$$
- **Expected Baseline Fingerprint:** `12.9660,79.1373->12.9968,79.2455:13519`
- Any change triggers a logged collector warning (`[WARN] Warning: segment fingerprint changed`).

---

## 5. Non-Overlapping WMO Weather Code Mapping

The Open-Meteo WMO 4677 weather code is mapped to the project's categorical weather schema without overlap:

| WMO Code | Description | Project Category | Rationale & Documentation Notes |
| :---: | :--- | :---: | :--- |
| **0** | Clear sky | **Clear** | Cloud cover $< 1/8$ oktas. |
| **1** | Mainly clear | **Clear** | **Project Assumption:** Cloud cover is $< 2/8$ oktas; road surface is dry with negligible solar radiation reduction or friction change. |
| **2** | Partly cloudy | **Cloudy** | Cloud cover $3/8$ to $5/8$ oktas. |
| **3** | Overcast | **Cloudy** | Cloud cover $\ge 6/8$ oktas. |
| **51** | Light drizzle | **Rain** | 15-min rolling precipitation $< 5.0\text{ mm}$. |
| **53** | Moderate drizzle | **Rain** | 15-min rolling precipitation $< 5.0\text{ mm}$. |
| **55** | Dense drizzle | **Rain** | 15-min rolling precipitation $< 5.0\text{ mm}$. |
| **61** | Slight rain | **Rain** | 15-min rolling precipitation $< 5.0\text{ mm}$. |
| **63** | Moderate rain | **Rain** | 15-min rolling precipitation $< 5.0\text{ mm}$. |
| **65** | Heavy rain | **Heavy Rain** | 15-min rolling precipitation $\ge 5.0\text{ mm}$ or intense convective deluge. |
| **80** | Slight rain showers | **Rain** | 15-min rolling precipitation $< 5.0\text{ mm}$. |
| **81** | Moderate rain showers | **Rain** | 15-min rolling precipitation $< 5.0\text{ mm}$. |
| **82** | Violent rain showers | **Heavy Rain** | 15-min rolling precipitation $\ge 5.0\text{ mm}$. |
| **95** | Thunderstorm: Slight/mod | **Heavy Rain** | Convective storm activity with road hazards. |
| **45, 48** | Fog, depositing rime fog | **Other/Unknown** | Outside target campus transit categories. |
| **56, 57** | Freezing drizzle | **Other/Unknown** | Outside target campus transit categories. |
| **66, 67** | Freezing rain | **Other/Unknown** | Outside target campus transit categories. |
| **71-77** | Snow fall, snow grains | **Other/Unknown** | Outside target campus transit categories. |
| **85, 86** | Snow showers | **Other/Unknown** | Outside target campus transit categories. |
| **96, 99** | Thunderstorm with hail | **Other/Unknown** | Outside target campus transit categories. |

> [!IMPORTANT]
> When `Other/Unknown` is returned, the dashboard live mode **must never auto-fill** the weather dropdown; it retains manual operator selection while displaying the raw WMO code and description.

---

## 6. Live Speed-Based Congestion Index

Because TomTom reports speed rather than vehicle count, a provisional, transparent speed-reduction index is computed:

### 6.1 Mathematical Formulation
1. **Speed Ratio ($r$):**
   $$r = \frac{v_{\text{current}}}{v_{\text{free\_flow}}}$$
2. **Provisional Congestion Index ($I_{\text{speed}}$):**
   $$I_{\text{speed}} = \min\left(100.0, \max\left(0.0, \frac{1.0 - r}{0.85} \times 100.0\right)\right)$$
   *(Calibrated such that $r = 1.0 \implies I = 0$, $r = 0.5 \implies I \approx 58.8$, and $r \le 0.15 \implies I = 100.0$).*

### 6.2 Classification Thresholds
- **Low:** $I_{\text{speed}} < 35.0$ ($r > 0.70$)
- **Moderate:** $35.0 \le I_{\text{speed}} \le 70.0$ ($0.405 \le r \le 0.70$)
- **High:** $I_{\text{speed}} > 70.0$ ($r < 0.405$)

### 6.3 Special State Handling
- **Road Closure (`roadClosure == True`):**
  - `live_speed_congestion_index`: blank (`None`)
  - `live_index_class`: blank (`None`)
  - `live_state`: `"road_closure"`
  - `note`: `"Corridor road closure reported by TomTom; index not applicable"`
- **Low Confidence (`confidence < 0.50`):**
  - `low_confidence_flag`: `"True"`
  - `live_state`: `"low_confidence"`
  - `note`: `"Low confidence traffic reading (confidence < 0.5)"`
- **Provider Outage:**
  - `live_state`: `"traffic_unavailable"`

---

## 7. CSV Schema Specification (Version 1.2)

Every row in `data/live/traffic_weather_live.csv` and `data/live/mock_feed_demo.csv` strictly contains **35 columns**:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `poll_id` | String | Unique identifier: `poll_YYYYMMDD_HHMMSS_mmm`. |
| `schema_version` | String | Fixed version: `1.2`. |
| `is_illustrative` | Boolean | `False` for real live polls; `True` for demo/mock records. |
| `utc_timestamp` | ISO-8601 | UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`). |
| `local_timestamp` | ISO-8601 | Asia/Kolkata local timestamp with timezone offset (`+05:30`). |
| `traffic_source` | String | Data source: `tomtom`, `mock`, or `unavailable`. |
| `weather_source` | String | Data source: `open-meteo`, `mock`, or `unavailable`. |
| `query_lat` | Float | Query latitude (rounded to 5 decimals). |
| `query_lon` | Float | Query longitude (rounded to 5 decimals). |
| `snapped_lat` | Float / Blank | First vertex latitude returned by TomTom. |
| `snapped_lon` | Float / Blank | First vertex longitude returned by TomTom. |
| `segment_fingerprint` | String / Blank | Invariant polyline fingerprint. |
| `gate_to_segment_m` | Float / Blank | Distance in meters from campus gate to nearest polyline vertex. |
| `segment_length_m` | Float / Blank | Total length of the snapped segment polyline in meters. |
| `road_class` | String / Blank | Functional road class (e.g. `FRC2`). |
| `closure_flag` | String / Blank | `True` or `False` if observed; blank if poll failed. |
| `current_speed` | Float / Blank | Current harmonic speed (km/h) reported by TomTom. |
| `free_flow_speed` | Float / Blank | Free flow speed (km/h) reported by TomTom. |
| `speed_ratio` | Float / Blank | Ratio $v_{\text{current}} / v_{\text{free\_flow}}$. |
| `current_travel_time` | Float / Blank | Current travel time (seconds) reported by TomTom. |
| `free_flow_travel_time`| Float / Blank | Free flow travel time (seconds) reported by TomTom. |
| `confidence` | Float / Blank | Provider confidence value in $[0.0, 1.0]$. |
| `low_confidence_flag` | String / Blank | `True` if confidence $< 0.5$; `False` otherwise; blank if failed. |
| `live_speed_congestion_index` | Float / Blank | Computed speed index in $[0, 100]$; blank on road closure or failure. |
| `live_index_class` | String / Blank | `Low`, `Moderate`, or `High`; blank on road closure or failure. |
| `live_state` | String | Overarching status: `normal`, `road_closure`, `low_confidence`, `traffic_unavailable`. |
| `precip_mm` | Float / Blank | 15-minute rolling precipitation accumulation (mm) from Open-Meteo. |
| `rain_mm` | Float / Blank | 15-minute rolling rain accumulation (mm) from Open-Meteo. |
| `wmo_code` | Integer / Blank | WMO 4677 weather interpretation code. |
| `weather_category` | String / Blank | Mapped category: `Clear`, `Cloudy`, `Rain`, `Heavy Rain`, or `Other/Unknown`. |
| `traffic_status` | String | `OK` or `ERROR`. |
| `traffic_http_status`| Integer / Blank | HTTP status code (e.g. `200`, `400`, `429`). |
| `weather_status` | String | `OK` or `ERROR`. |
| `note` | String / Blank | Informational messages (corridor attribution, closures, warnings). |
| `error` | String / Blank | Scrubbed error message if an exception occurred. |

---

## 8. Operational Scheduling & Data Readiness

### 8.1 Automated Polling Schedule
The monitoring window is **08:00 AM – 10:00 AM IST**, Monday through Friday. At 5-minute intervals, this produces **25 polling slots per morning session**.

#### Timezone Offsets (IST vs UTC)
- **IST (Indian Standard Time):** UTC + 05:30.
- **Monitoring Window:** 08:00–10:00 AM IST $\longleftrightarrow$ **02:30–04:30 AM UTC**.

#### Execution Options
1. **Local Background Collector:**
   ```bash
   python src/live_collector.py --interval 5 --window 08:00-10:00
   ```
2. **Windows Task Scheduler:**
   - Action: Start a program `python.exe`
   - Arguments: `src/live_collector.py --interval 5 --window 08:00-10:00`
   - Trigger: Daily at 07:55 AM IST.
3. **GitHub Actions Workflow (Cron in UTC):**
   - Scheduled cron: `30 2 * * 1-5` (Triggers at 02:30 UTC on weekdays).

### 8.2 Data Readiness & Retraining Feasibility
The data readiness script audits the real log volume and determines whether retraining the ML model is scientifically feasible:
```bash
python src/data_readiness.py
```
- **Auditor Checks:**
  - Real row count vs. mock row separation.
  - Distinct morning sessions captured.
  - Missing 5-minute poll slots per session (out of 25 expected).
  - API provider failure rates (TomTom and Open-Meteo).
  - Speed-based class distribution (Low / Moderate / High).
  - Overall live state breakdown (`normal`, `road_closure`, `low_confidence`, `traffic_unavailable`).
- **Retraining Verdict Policy:**
  - Retraining is prohibited until at least 2–3 weeks of continuous weekday morning data ($\ge 500$ observations) have been gathered under stable conditions with verified gate ground truth.
