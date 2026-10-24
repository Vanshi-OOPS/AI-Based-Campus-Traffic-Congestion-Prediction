"""Central configuration file for Smart Traffic Congestion Alert.

Single source of truth for all parameters, paths, seeds, thresholds, and distributions.
No magic numbers outside this module.
"""

from pathlib import Path
import numpy as np

# -----------------------------------------------------------------------------
# Base Directories & Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_LIVE_DIR = DATA_DIR / "live"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
DOCS_FIGURES_DIR = DOCS_DIR / "figures"

# Ensure runtime directories exist
for p in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_LIVE_DIR, MODELS_DIR, DOCS_DIR, DOCS_FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = DATA_RAW_DIR / "simulated_traffic_data.csv"
DIAGNOSTICS_DATA_PATH = DATA_RAW_DIR / "simulated_traffic_diagnostics.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "featured_traffic_data.csv"
DATA_DICTIONARY_PATH = DOCS_DIR / "data_dictionary.md"

# -----------------------------------------------------------------------------
# Random Seed & Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

def get_rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Return a reproducible NumPy Generator."""
    return np.random.default_rng(seed)

# -----------------------------------------------------------------------------
# Simulation Grid & Calendar Settings
# -----------------------------------------------------------------------------
START_DATE = "2026-09-01"
NUM_DAYS = 180  # 180 days * 25 intervals = 4,500 rows
INTERVAL_MINUTES = 5

WINDOW_START_HOUR = 8
WINDOW_START_MINUTE = 0
WINDOW_END_HOUR = 10
WINDOW_END_MINUTE = 0
INTERVALS_PER_MORNING = 25  # 08:00 to 10:00 inclusive at 5-min steps: 120 / 5 + 1 = 25

# Rush Hour Peak Surge Window (Arrival peak distribution)
PEAK_START_MINUTE_OF_WINDOW = 30  # 08:30 (minute 30)
PEAK_END_MINUTE_OF_WINDOW = 75    # 09:15 (minute 75)
PEAK_CENTER_MINUTE = 50           # ~08:50 (minute 50)
PEAK_SIGMA_MINUTES = 16.0         # Width of Gaussian arrival rush

# Weekend demand multiplier (applied to baseline volume)
WEEKEND_DEMAND_MULTIPLIER = 0.40

# -----------------------------------------------------------------------------
# Road, Gate, and Weather Physical Parameters
# -----------------------------------------------------------------------------
BASE_CAPACITY = 100.0  # Vehicles processed per 5-min interval under ideal conditions

WEATHER_CATEGORIES = ["Clear", "Cloudy", "Rain", "Heavy Rain"]
WEATHER_PROBABILITIES = {
    "Clear": 0.60,
    "Cloudy": 0.25,
    "Rain": 0.10,
    "Heavy Rain": 0.05,
}

# Capacity reduction fractions due to weather friction
WEATHER_CAPACITY_REDUCTION = {
    "Clear": 0.00,
    "Cloudy": 0.02,
    "Rain": 0.15,
    "Heavy Rain": 0.35,
}

# Campus Event Parameters
EVENT_PROBABILITY = 0.10  # ~10% of mornings
EVENT_DEMAND_SURGE_RANGE = (0.25, 0.40)      # +25% to +40% arrivals
EVENT_GATE_REDUCTION_RANGE = (0.10, 0.20)    # 10% to 20% throughput reduction from security

# Gate Jitter (independent physical mechanism noise)
GATE_JITTER_SIGMA = 0.03  # ~3% standard deviation in gate processing flow

# -----------------------------------------------------------------------------
# Congestion Scoring & Noise Parameters
# -----------------------------------------------------------------------------
# Formula: P_t = 65.0 * (V_t / C_t) + QUEUE_WEIGHT * (Q_t / BASE_CAPACITY) + eps_t
VOLUME_CAPACITY_WEIGHT = 65.0
QUEUE_WEIGHT = 25.0  # Meaningful queue accumulation memory without saturating score

# AR(1) Observational Noise Parameters
AR1_PHI = 0.50
AR1_MARGINAL_SIGMA = 4.50
AR1_INNOVATION_SIGMA = float(AR1_MARGINAL_SIGMA * np.sqrt(1.0 - AR1_PHI**2))  # ~3.8971

# Score to Speed Ratio parameters: speed_ratio = clip(1.0 - (score / 100) * 0.85, 0.15, 1.0)
SPEED_RATIO_SLOPE = 0.85
SPEED_RATIO_MIN = 0.15
SPEED_RATIO_MAX = 1.00

# Congestion Classification Thresholds (Prototype-defined thresholds)
SCORE_LOW_MAX = 40.0
SCORE_MODERATE_MAX = 70.0
CONGESTION_CLASSES = ["Low", "Moderate", "High"]

def score_to_class(score: float) -> str:
    """Map numeric congestion score (0-100) to categorical class with strict boundaries."""
    if score < SCORE_LOW_MAX:
        return "Low"
    elif score < SCORE_MODERATE_MAX:
        return "Moderate"
    else:
        return "High"

def get_class_index(model: Any, class_name: str = "High") -> int:
    """Safely get the column index for a given class in model.predict_proba."""
    classes = None
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    elif hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf_step = model.named_steps["clf"]
        if hasattr(clf_step, "classes_"):
            classes = list(clf_step.classes_)
        elif hasattr(clf_step, "estimator") and hasattr(clf_step.estimator, "classes_"):
            classes = list(clf_step.estimator.classes_)
    if classes and class_name in classes:
        return classes.index(class_name)
    return CONGESTION_CLASSES.index(class_name)

def score_to_speed_ratio(score: float) -> float:
    """Monotone decreasing function of congestion score to simulate TomTom speed ratio."""
    raw = 1.0 - (score / 100.0) * SPEED_RATIO_SLOPE
    return float(np.clip(raw, SPEED_RATIO_MIN, SPEED_RATIO_MAX))

# -----------------------------------------------------------------------------
# Chronological Split Fractions (Assigned by whole morning dates)
# -----------------------------------------------------------------------------
TRAIN_SPLIT_RATIO = 0.70  # First 70% of days
VAL_SPLIT_RATIO = 0.15    # Next 15% of days
TEST_SPLIT_RATIO = 0.15   # Final 15% of days

# -----------------------------------------------------------------------------
# Feature Engineering Encodings & Column Lists
# -----------------------------------------------------------------------------
WEATHER_ORDINAL_MAP = {
    "Clear": 0,
    "Cloudy": 1,
    "Rain": 2,
    "Heavy Rain": 3,
}

# Two feature sets: CORE (manual form input) and FULL (timeline with lag features)
CORE_FEATURE_COLS = [
    "minute_of_window",
    "day_of_week",
    "vehicle_count",
    "weather_ordinal",
    "weather_Cloudy",
    "weather_Rain",
    "weather_Heavy Rain",
    "event_flag",
]

FULL_FEATURE_COLS = CORE_FEATURE_COLS + [
    "rolling_vehicle_avg_15min",
    "vehicle_surge_rate",
]

# Legacy alias for backward compatibility
MODEL_FEATURE_COLS = FULL_FEATURE_COLS

# High-alert recall tuning target on validation (default: 90%)
ALERT_TARGET_HIGH_RECALL = 0.90

# Columns that represent latent state or target leakage (MUST NEVER BE MODEL FEATURES)
LEAKAGE_COLS = [
    "effective_capacity",
    "queue_length",
    "ar1_noise",
    "raw_score_pre_clip",
    "congestion_score",
    "congestion_class",
    "congestion_class_t_plus_15min",
    "speed_ratio",  # Deterministic mapping of congestion_score
]

# Identifier / Metadata columns
METADATA_COLS = [
    "date",
    "timestamp",
    "day_index",
    "split",
    "data_source",
]
