"""SHAP Factor Contribution Chart Component.

Renders factor attribution charts:
1. Regressor Waterfall / Horizontal Bar Chart (points of congestion score).
   Labeled: 'Model attribution in points of congestion score, not a physical cause'.
2. Classifier Probability Contribution Chart (effect on probability of High).
"""

from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg


def render_regressor_factor_chart(explanation: Dict[str, Any]) -> plt.Figure:
    """Generate a horizontal bar chart of regressor SHAP factor contributions.

    Parameters
    ----------
    explanation : Dict[str, Any]
        Output of TrafficExplainer.explain_prediction(mode='regressor').
        Contains 'base_value', 'predicted_output', 'contributions'.
    """
    base_val = explanation["base_value"]
    pred_val = explanation["predicted_output"]
    contribs = explanation["contributions"]

    # Sort factors by absolute impact
    sorted_factors = sorted(contribs.items(), key=lambda item: abs(item[1]), reverse=False)
    factors = [f[0].replace("_", " ").title() for f in sorted_factors]
    values = [f[1] for f in sorted_factors]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, len(factors) * 0.45)))

    y_pos = np.arange(len(factors))
    bars = ax.barh(y_pos, values, color=colors, height=0.6, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors, fontsize=10, fontweight="bold")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Impact (Points of Congestion Score)", fontweight="bold", fontsize=10)
    ax.set_title(
        f"Factor Contributions to Predicted Score: {pred_val:.1f} pts (Base: {base_val:.1f} pts)\n"
        f"[{cfg.SHAP_CHART_DISCLAIMER}]",
        fontsize=10,
        fontweight="bold",
        pad=10
    )

    # Annotate bar values
    for bar, val in zip(bars, values):
        sign = "+" if val > 0 else ""
        x_pos = val + (0.5 if val >= 0 else -0.5)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{sign}{val:.1f} pts",
                va="center", ha=ha, fontsize=9, fontweight="bold")

    # Set x-limits with margin
    limit = max(5.0, max(abs(min(values, default=0)), abs(max(values, default=0))) + 3.0)
    ax.set_xlim(-limit, limit)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig


def render_classifier_probability_chart(explanation: Dict[str, Any]) -> plt.Figure:
    """Generate a horizontal bar chart of classifier effect on probability of High."""
    base_val = explanation["base_value"]
    pred_val = explanation["predicted_output"]
    contribs = explanation["contributions"]

    sorted_factors = sorted(contribs.items(), key=lambda item: abs(item[1]), reverse=False)
    factors = [f[0].replace("_", " ").title() for f in sorted_factors]
    values = [f[1] for f in sorted_factors]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, len(factors) * 0.45)))
    y_pos = np.arange(len(factors))
    bars = ax.barh(y_pos, values, color=colors, height=0.6, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(factors, fontsize=10, fontweight="bold")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Impact on P(High Congestion)", fontweight="bold", fontsize=10)
    ax.set_title(
        f"Factor Effect on P(High): {pred_val:.1%} (Base: {base_val:.1%})\n"
        "[Probability Units: Model Attribution on High Risk]",
        fontsize=10,
        fontweight="bold",
        pad=10
    )

    for bar, val in zip(bars, values):
        sign = "+" if val > 0 else ""
        x_pos = val + (0.01 if val >= 0 else -0.01)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{sign}{val:.3f}",
                va="center", ha=ha, fontsize=9, fontweight="bold")

    limit = max(0.10, max(abs(min(values, default=0)), abs(max(values, default=0))) + 0.05)
    ax.set_xlim(-limit, limit)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig
