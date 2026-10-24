"""Central Operational Alert Logic for Campus Traffic Congestion.

A single, unit-tested source of truth for:
- Mapping predicted probabilities, classes, and continuous scores to operational alert banners.
- Rule: High alert is triggered if and only if p(High) >= alert_threshold (validation-chosen cutoff).
- If p(High) is not directly available, triggers if predicted_class == 'High'.
- Formats editable diversion routes with mandatory college-confirmation disclaimers.
"""

from typing import Dict, Any, Optional
from src import config as cfg


def evaluate_operational_alert(
    predicted_class: str,
    congestion_score: float,
    high_probability: Optional[float] = None,
    alert_threshold: float = 0.50,
    custom_diversion_route: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate alert status, badge styling, banner messaging, and boundary notices.

    Parameters
    ----------
    predicted_class : str
        Predicted class from classifier ('Low', 'Moderate', 'High').
    congestion_score : float
        Predicted continuous score from regressor (0.0 to 100.0).
    high_probability : Optional[float]
        Predicted probability of High class, p(High).
    alert_threshold : float
        Validation-selected decision threshold for High alerts (e.g. 0.732).
    custom_diversion_route : Optional[str]
        Optional campus-provided diversion route text.

    Returns
    -------
    Dict[str, Any]
        alert_level: 'High' | 'Moderate' | 'Low'
        badge_color: 'red' | 'orange' | 'green'
        banner_type: 'error' | 'warning' | 'success'
        banner_message: str
        diversion_route_note: Optional[str]
        disagreement_note: Optional[str]
    """
    # 1. Operational Alert Rule: High if p(High) >= threshold, else falls back to predicted class
    if high_probability is not None:
        is_high_alert = bool(high_probability >= alert_threshold)
    else:
        is_high_alert = bool(predicted_class == "High")

    # 2. Assign operational level
    if is_high_alert:
        alert_level = "High"
        badge_color = "#e74c3c"  # Red
        banner_type = "error"
        banner_message = cfg.ALERT_MSG_HIGH
        route_text = custom_diversion_route or cfg.DEFAULT_DIVERSION_ROUTE_EXAMPLE
        diversion_route_note = (
            f"Suggested diversion route: {route_text}\n"
            f"(Example procedure, confirm with the college administration before activating)."
        )
    elif predicted_class == "Moderate" or (congestion_score >= cfg.SCORE_LOW_MAX and congestion_score < cfg.SCORE_MODERATE_MAX):
        alert_level = "Moderate"
        badge_color = "#f39c12"  # Amber/Orange
        banner_type = "warning"
        banner_message = cfg.ALERT_MSG_MODERATE
        diversion_route_note = None
    else:
        alert_level = "Low"
        badge_color = "#27ae60"  # Green
        banner_type = "success"
        banner_message = cfg.ALERT_MSG_LOW
        diversion_route_note = None

    # 3. Disagreement Detection near boundaries
    # E.g. score >= 70 but class is Moderate, or score < 70 but class is High
    disagreement_note = None
    score_implied_class = cfg.score_to_class(congestion_score)
    if score_implied_class != predicted_class:
        disagreement_note = cfg.DISAGREEMENT_NOTE.format(
            score=congestion_score,
            cls=predicted_class
        )

    return {
        "alert_level": alert_level,
        "badge_color": badge_color,
        "banner_type": banner_type,
        "banner_message": banner_message,
        "diversion_route_note": diversion_route_note,
        "disagreement_note": disagreement_note,
        "alert_threshold_used": alert_threshold,
    }
