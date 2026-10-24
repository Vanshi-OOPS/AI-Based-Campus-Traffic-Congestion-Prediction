"""Live Speed-Based Congestion Index Calculation.

Converts real-time road speeds (currentSpeed, freeFlowSpeed) from TomTom
into a standardized relative congestion index (0 to 100) and provisional class.

IMPORTANT METHODOLOGICAL NOTE:
The formula `index = clip((1 - r) / 0.85 * 100, 0, 100)` is a PROTOTYPE assumption
inherited from the Phase 1 synthetic data generation schema (where speed ratio was
modeled as r = 1.0 - 0.85 * score / 100). It is NOT an empirically calibrated
traffic physics equation, and this live index is NOT equivalent to the Phase 2
supervised ML congestion score.
"""

from typing import Dict, Any, Optional
import numpy as np

from src import config as cfg


def compute_live_speed_index(
    current_speed: Optional[float],
    free_flow_speed: Optional[float],
    road_closure: bool = False,
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate the live speed-based congestion index and provisional class.

    Parameters
    ----------
    current_speed : Optional[float]
        Current observed speed in km/h.
    free_flow_speed : Optional[float]
        Free-flow unconstrained speed in km/h.
    road_closure : bool
        Flag indicating whether a road closure is reported on the segment.
    confidence : Optional[float]
        TomTom data confidence score between 0.0 and 1.0.

    Returns
    -------
    Dict[str, Any]
        speed_ratio: raw unconstrained ratio (current / free_flow) or None
        speed_ratio_capped: ratio capped at [0.0, 1.0] for index computation or None
        live_speed_congestion_index: float 0.0 to 100.0 or None
        live_index_class: 'Low' | 'Moderate' | 'High' | None
        live_state: 'normal' | 'road_closure' | 'low_confidence' | 'traffic_unavailable'
        low_confidence_flag: Optional[bool]
        note: str
    """
    # 1. Road Closure: index and class are blank, reported under live_state
    if road_closure:
        return {
            "speed_ratio": 0.0,
            "speed_ratio_capped": 0.0,
            "live_speed_congestion_index": None,
            "live_index_class": None,
            "live_state": "road_closure",
            "low_confidence_flag": False if confidence is not None else None,
            "note": "Road closure reported by provider",
        }

    # 2. Unobserved or Invalid Speed Data: leave blank, state = traffic_unavailable
    if (
        current_speed is None
        or free_flow_speed is None
        or free_flow_speed <= 0.0
        or current_speed < 0.0
    ):
        return {
            "speed_ratio": None,
            "speed_ratio_capped": None,
            "live_speed_congestion_index": None,
            "live_index_class": None,
            "live_state": "traffic_unavailable",
            "low_confidence_flag": None,
            "note": "Traffic speed data unavailable",
        }

    # 3. Compute Speed Ratios
    raw_ratio = float(current_speed) / float(free_flow_speed)
    capped_ratio = float(np.clip(raw_ratio, 0.0, 1.0))

    # 4. Map to Congestion Index (0 to 100)
    # Inherited prototype relation: r = 1.0 - (score / 100) * 0.85
    # Inverted: index = (1.0 - r) / 0.85 * 100
    slope = getattr(cfg, "SPEED_RATIO_SLOPE", 0.85)
    raw_index = ((1.0 - capped_ratio) / slope) * 100.0
    live_index = float(np.clip(raw_index, 0.0, 100.0))

    # 5. Provisional Classification Thresholds
    low_cutoff = getattr(cfg, "LIVE_INDEX_LOW_MAX", 40.0)
    mod_cutoff = getattr(cfg, "LIVE_INDEX_MODERATE_MAX", 70.0)

    if live_index < low_cutoff:
        live_class = "Low"
    elif live_index < mod_cutoff:
        live_class = "Moderate"
    else:
        live_class = "High"

    # 6. Confidence Evaluation
    conf_threshold = getattr(cfg, "TOMTOM_MIN_CONFIDENCE", 0.50)
    if confidence is not None:
        is_low_conf = bool(confidence < conf_threshold)
        live_state = "low_confidence" if is_low_conf else "normal"
    else:
        is_low_conf = None
        live_state = "normal"

    note = "Provisional speed-based rule; normal flow"
    if is_low_conf:
        note = f"Warning: provider confidence below threshold ({confidence:.2f})"

    return {
        "speed_ratio": round(raw_ratio, 4),
        "speed_ratio_capped": round(capped_ratio, 4),
        "live_speed_congestion_index": round(live_index, 2),
        "live_index_class": live_class,
        "live_state": live_state,
        "low_confidence_flag": is_low_conf,
        "note": note,
    }
