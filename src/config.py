"""Central configuration file for Smart Traffic Congestion Alert."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_LIVE_DIR = DATA_DIR / "live"
MODELS_DIR = BASE_DIR / "models"

# Ensure runtime directories exist
for p in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_LIVE_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Campus & Segment Geography
# Default coordinate: College Main Gate (example: 12.9698, 79.1559)
CAMPUS_NAME = "College Main Gate Segment"
GATE_COORDINATES = (12.9698, 79.1559)
ROAD_SEGMENT_POINTS = [
    (12.9675, 79.1530),  # Approach road start
    (12.9698, 79.1559),  # Main Campus Gate
    (12.9715, 79.1585),  # Inner perimeter road
]

# Time Monitoring Window
MONITOR_START_HOUR = 8
MONITOR_START_MINUTE = 0
MONITOR_END_HOUR = 10
MONITOR_END_MINUTE = 0
INTERVAL_MINUTES = 5

# Rush Hour Peak Surge Window (Peak arrival distribution)
PEAK_START_HOUR = 8
PEAK_START_MINUTE = 30
PEAK_END_HOUR = 9
PEAK_END_MINUTE = 15

# Weather Categories
WEATHER_CATEGORIES = ["Clear", "Cloudy", "Rain", "Heavy Rain"]

# Congestion Thresholds & Classes
CONGESTION_CLASSES = ["Low", "Moderate", "High"]
SCORE_LOW_MAX = 40.0
SCORE_MODERATE_MAX = 70.0

def score_to_class(score: float) -> str:
    """Map numeric congestion score (0-100) to categorical class."""
    if score < SCORE_LOW_MAX:
        return "Low"
    elif score < SCORE_MODERATE_MAX:
        return "Moderate"
    else:
        return "High"

# Model Artifact Paths
CLASSIFIER_MODEL_PATH = MODELS_DIR / "congestion_classifier.joblib"
REGRESSOR_MODEL_PATH = MODELS_DIR / "congestion_regressor.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = MODELS_DIR / "eval_metrics.json"

# API Endpoints & Free-tier settings
TOMTOM_FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
