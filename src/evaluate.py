"""Comprehensive Evaluation Pipeline for Smart Traffic Congestion Alert.

Evaluates on the held-out TEST set (scored once):
- Multi-class metrics: Accuracy, Macro-F1, per-class Precision/Recall/F1, Confusion Matrix.
- Operational alert metrics: False High alerts, Missed High intervals, Class-skipping errors (Low <-> High).
- Whole-morning cluster bootstrap 95% Confidence Intervals (1,000 resamples).
- Condition slices by weather and event_flag (flags slices with < 5 mornings as anecdotal).
- High-alert probability threshold tuning on VALIDATION to hit ~90% High recall, evaluated on TEST.
- 15-Minute Advance Alert model evaluation vs baselines (Current Count Rule & Current Class Persists).
- Visualization exports: confusion_matrix.png, test_score_predictions.png, core_vs_full_comparison.png, alert_threshold_curve.png.
"""

from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

from src import config as cfg


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute detailed multi-class classification and operational alert metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_class = {}
    for c in cfg.CONGESTION_CLASSES:
        prec = float(precision_score(y_true == c, y_pred == c, zero_division=0))
        rec = float(recall_score(y_true == c, y_pred == c, zero_division=0))
        f1 = float(f1_score(y_true == c, y_pred == c, zero_division=0))
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    # Operational metrics
    false_high_alerts = int(np.sum((y_pred == "High") & (y_true != "High")))
    missed_high_intervals = int(np.sum((y_pred != "High") & (y_true == "High")))

    # Class-skipping errors: critical misclassifications
    low_pred_high_true = int(np.sum((y_pred == "Low") & (y_true == "High")))
    high_pred_low_true = int(np.sum((y_pred == "High") & (y_true == "Low")))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=cfg.CONGESTION_CLASSES)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "false_high_alerts": false_high_alerts,
        "missed_high_intervals": missed_high_intervals,
        "class_skipping": {
            "low_pred_when_high_true": low_pred_high_true,
            "high_pred_when_low_true": high_pred_low_true,
            "total_skipping": low_pred_high_true + high_pred_low_true,
        },
        "confusion_matrix": cm.tolist(),
    }


def compute_whole_morning_bootstrap_ci(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    seed: int = cfg.RANDOM_SEED,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% Confidence Intervals by resampling WHOLE MORNINGS with replacement."""
    rng = cfg.get_rng(seed)
    df = test_df.copy()
    df["pred"] = y_pred

    dates = df["date"].unique()
    n_dates = len(dates)

    morning_dict = {d: sub for d, sub in df.groupby("date")}

    macro_f1s = np.zeros(n_resamples)
    high_recalls = np.zeros(n_resamples)

    for i in range(n_resamples):
        sample_dates = rng.choice(dates, size=n_dates, replace=True)
        sample_df = pd.concat([morning_dict[d] for d in sample_dates], ignore_index=True)

        y_t = sample_df["congestion_class"].values
        y_p = sample_df["pred"].values

        macro_f1s[i] = f1_score(y_t, y_p, average="macro", zero_division=0)
        high_recalls[i] = recall_score(y_t == "High", y_p == "High", zero_division=0)

    return {
        "macro_f1_ci95": (
            round(float(np.percentile(macro_f1s, 2.5)), 4),
            round(float(np.percentile(macro_f1s, 97.5)), 4),
        ),
        "high_recall_ci95": (
            round(float(np.percentile(high_recalls, 2.5)), 4),
            round(float(np.percentile(high_recalls, 97.5)), 4),
        ),
    }


def compute_condition_slices(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Slice classification metrics by weather condition and campus event presence.

    Flags any condition with < 5 mornings as 'anecdotal, not a reliable claim'.
    """
    df = test_df.copy()
    df["pred"] = y_pred

    slices = {"weather": {}, "event": {}}

    # Weather slices
    for w in cfg.WEATHER_CATEGORIES:
        sub = df[df["weather"] == w]
        mornings = sub["date"].nunique()
        is_anecdotal = bool(mornings < 5)
        note = "anecdotal, not a reliable claim" if is_anecdotal else "reliable sample"

        if len(sub) == 0:
            continue

        acc = accuracy_score(sub["congestion_class"], sub["pred"])
        rec_high = recall_score(sub["congestion_class"] == "High", sub["pred"] == "High", zero_division=0)
        prec_high = precision_score(sub["congestion_class"] == "High", sub["pred"] == "High", zero_division=0)

        slices["weather"][w] = {
            "mornings": mornings,
            "intervals": len(sub),
            "is_anecdotal": is_anecdotal,
            "status": note,
            "accuracy": round(float(acc), 4),
            "high_recall": round(float(rec_high), 4),
            "high_precision": round(float(prec_high), 4),
        }

    # Event slices
    for ev in [0, 1]:
        label = "Event (1)" if ev == 1 else "No Event (0)"
        sub = df[df["event_flag"] == ev]
        mornings = sub["date"].nunique()
        is_anecdotal = bool(mornings < 5)
        note = "anecdotal, not a reliable claim" if is_anecdotal else "reliable sample"

        acc = accuracy_score(sub["congestion_class"], sub["pred"])
        rec_high = recall_score(sub["congestion_class"] == "High", sub["pred"] == "High", zero_division=0)
        prec_high = precision_score(sub["congestion_class"] == "High", sub["pred"] == "High", zero_division=0)

        slices["event"][label] = {
            "mornings": mornings,
            "intervals": len(sub),
            "is_anecdotal": is_anecdotal,
            "status": note,
            "accuracy": round(float(acc), 4),
            "high_recall": round(float(rec_high), 4),
            "high_precision": round(float(prec_high), 4),
        }

    return slices


def tune_alert_threshold_on_val(
    model: Any,
    val_df: pd.DataFrame,
    target_high_recall: float = cfg.ALERT_TARGET_HIGH_RECALL,
) -> Dict[str, Any]:
    """Find the High-probability threshold on VALIDATION that reaches ~target_high_recall.

    Returns the chosen threshold and validation curve data.
    """
    probs = model.predict_proba(val_df)
    high_idx = cfg.get_class_index(model, "High")
    high_probs = probs[:, high_idx]
    y_true_high = (val_df["congestion_class"] == "High").values

    thresholds = np.linspace(0.05, 0.95, 100)
    recalls = []
    precisions = []
    false_alarm_rates = []

    best_thresh = 0.50
    best_diff = 999.0

    for t in thresholds:
        pred_high = (high_probs >= t)
        rec = recall_score(y_true_high, pred_high, zero_division=0)
        prec = precision_score(y_true_high, pred_high, zero_division=0)
        far = np.sum(pred_high & ~y_true_high) / max(1, np.sum(~y_true_high))

        recalls.append(float(rec))
        precisions.append(float(prec))
        false_alarm_rates.append(float(far))

        # Check closest recall >= target (or closest overall)
        if rec >= target_high_recall:
            diff = rec - target_high_recall
            if diff < best_diff:
                best_diff = diff
                best_thresh = float(t)

    # In case no threshold gave >= target, pick the one closest to target
    if best_diff == 999.0:
        idx = int(np.argmin(np.abs(np.array(recalls) - target_high_recall)))
        best_thresh = float(thresholds[idx])

    # Compute validation metrics at chosen threshold
    val_pred_high = (high_probs >= best_thresh)
    val_rec = float(recall_score(y_true_high, val_pred_high, zero_division=0))
    val_prec = float(precision_score(y_true_high, val_pred_high, zero_division=0))
    val_far = float(np.sum(val_pred_high & ~y_true_high) / max(1, np.sum(~y_true_high)))

    return {
        "chosen_threshold": round(best_thresh, 3),
        "target_recall": target_high_recall,
        "val_high_recall": round(val_rec, 4),
        "val_high_precision": round(val_prec, 4),
        "val_false_alarm_rate": round(val_far, 4),
        "curve_thresholds": thresholds.tolist(),
        "curve_recalls": recalls,
        "curve_far": false_alarm_rates,
    }


def evaluate_alert_threshold_on_test(
    model: Any,
    test_df: pd.DataFrame,
    chosen_threshold: float,
) -> Dict[str, Any]:
    """Apply the fixed validation-selected High threshold to TEST."""
    probs = model.predict_proba(test_df)
    high_idx = cfg.get_class_index(model, "High")
    high_probs = probs[:, high_idx]
    y_true_high = (test_df["congestion_class"] == "High").values

    pred_high = (high_probs >= chosen_threshold)
    test_rec = float(recall_score(y_true_high, pred_high, zero_division=0))
    test_prec = float(precision_score(y_true_high, pred_high, zero_division=0))
    false_alarms = int(np.sum(pred_high & ~y_true_high))
    far = float(false_alarms / max(1, np.sum(~y_true_high)))

    return {
        "applied_threshold": chosen_threshold,
        "test_high_recall": round(test_rec, 4),
        "test_high_precision": round(test_prec, 4),
        "test_false_alarms": false_alarms,
        "test_false_alarm_rate": round(far, 4),
    }


def evaluate_15min_advance_alert(
    adv_model: Any,
    count_baseline: Any,
    test_df: pd.DataFrame,
    seed: int = cfg.RANDOM_SEED,
) -> Dict[str, Any]:
    """Evaluate 15-minute advance alert task on test set vs baselines."""
    # Exclude tail rows where forecast target is NaN
    clean_test = test_df.dropna(subset=["congestion_class_t_plus_15min"]).copy()
    y_true = (clean_test["congestion_class_t_plus_15min"] == "High").astype(int).values

    # 1. Advanced ML Model Predictions
    p_adv = adv_model.predict_proba(clean_test)[:, 1]
    pred_adv = (p_adv >= 0.35).astype(int)  # Balanced threshold for early warning

    # 2. Baseline A: Current Count Rule (Predict High-soon if Count Rule says High now)
    base_class_preds = count_baseline.predict(clean_test)
    pred_count_rule = (base_class_preds == "High").astype(int)

    # 3. Baseline B: Current Class Persists (Predict High-soon if currently High)
    pred_persists = (clean_test["congestion_class"] == "High").astype(int).values

    def get_adv_metrics(y_p: np.ndarray) -> Dict[str, Any]:
        return {
            "accuracy": round(float(accuracy_score(y_true, y_p)), 4),
            "precision": round(float(precision_score(y_true, y_p, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_p, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_p, zero_division=0)), 4),
            "false_alerts": int(np.sum((y_p == 1) & (y_true == 0))),
        }

    return {
        "xgboost_advance_model": get_adv_metrics(pred_adv),
        "current_count_rule_baseline": get_adv_metrics(pred_count_rule),
        "current_class_persists_baseline": get_adv_metrics(pred_persists),
        "test_rows_evaluated": len(clean_test),
        "true_high_soon_count": int(np.sum(y_true == 1)),
    }


def generate_evaluation_figures(
    best_baseline_cm: np.ndarray,
    best_model_cm: np.ndarray,
    best_baseline_name: str,
    best_model_name: str,
    test_df: pd.DataFrame,
    reg_preds: np.ndarray,
    core_vs_full_metrics: Dict[str, Dict[str, float]],
    alert_curve_data: Dict[str, Any],
) -> None:
    """Generate and save all required Phase 2 evaluation charts to docs/figures/."""
    cfg.DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix: Best Baseline vs Best Model Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, cm, title in zip(
        axes,
        [best_baseline_cm, best_model_cm],
        [f"Baseline: {best_baseline_name}", f"Model: {best_model_name}"]
    ):
        cax = ax.matshow(cm, cmap="Blues")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(cfg.CONGESTION_CLASSES)))
        ax.set_yticks(range(len(cfg.CONGESTION_CLASSES)))
        ax.set_xticklabels(cfg.CONGESTION_CLASSES)
        ax.set_yticklabels(cfg.CONGESTION_CLASSES)
        ax.set_xlabel("Predicted Class", fontweight="bold")
        ax.set_ylabel("True Class (Ground Truth)", fontweight="bold")
        ax.set_title(title, pad=15, fontweight="bold")
        for i in range(len(cfg.CONGESTION_CLASSES)):
            for j in range(len(cfg.CONGESTION_CLASSES)):
                val = cm[i, j]
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val > cm.max() / 2 else "black", fontweight="bold")

    fig.suptitle("Test Set Confusion Matrix Comparison (Simulated Data)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(cfg.DOCS_FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Predicted vs Actual Score Timeline on Sample Test Mornings
    sample_dates = test_df["date"].unique()[:3]
    fig, axes = plt.subplots(len(sample_dates), 1, figsize=(10, 2.8 * len(sample_dates)), sharex=True)
    if len(sample_dates) == 1:
        axes = [axes]

    test_copy = test_df.copy()
    test_copy["pred_score"] = reg_preds

    for ax, d in zip(axes, sample_dates):
        sub = test_copy[test_copy["date"] == d].sort_values("minute_of_window")
        ax.plot(sub["minute_of_window"], sub["congestion_score"], "b-o", label="Actual Ground Truth", lw=2)
        ax.plot(sub["minute_of_window"], sub["pred_score"], "r--s", label="Predicted Score (ML)", lw=2)
        ax.axhline(40, color="green", linestyle=":", alpha=0.6, label="Low Threshold (40)")
        ax.axhline(70, color="darkred", linestyle=":", alpha=0.6, label="High Alert (70)")
        weather_label = sub["weather"].iloc[0]
        event_label = "Event" if sub["event_flag"].iloc[0] == 1 else "Normal"
        ax.set_title(f"Test Morning: {d} (Weather: {weather_label}, Status: {event_label})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Congestion Score (0-100)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(-5, 105)

    axes[-1].set_xlabel("Minute of Window (0 = 8:00 AM, 60 = 9:00 AM, 120 = 10:00 AM)")
    axes[0].legend(loc="upper left", ncol=4, fontsize=8)
    fig.suptitle("Actual vs. Predicted Congestion Score Over Time (Simulated Test Set)", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(cfg.DOCS_FIGURES_DIR / "test_score_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. CORE vs FULL Feature Set Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics = ["Accuracy", "Macro-F1", "High Recall"]
    core_vals = [
        core_vs_full_metrics["CORE"]["accuracy"],
        core_vs_full_metrics["CORE"]["macro_f1"],
        core_vs_full_metrics["CORE"]["high_recall"],
    ]
    full_vals = [
        core_vs_full_metrics["FULL"]["accuracy"],
        core_vs_full_metrics["FULL"]["macro_f1"],
        core_vs_full_metrics["FULL"]["high_recall"],
    ]

    x = np.arange(len(metrics))
    width = 0.35
    rects1 = ax.bar(x - width/2, core_vals, width, label="CORE Features (No Lags)", color="#4a90e2")
    rects2 = ax.bar(x + width/2, full_vals, width, label="FULL Features (+ Rolling Avg & Surge)", color="#50e3c2")

    ax.set_ylabel("Score (0.0 to 1.0)", fontweight="bold")
    ax.set_title("Value of Temporal Lag Features: CORE vs. FULL Feature Set (Test Set)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(cfg.DOCS_FIGURES_DIR / "core_vs_full_comparison.png", dpi=150)
    plt.close(fig)

    # 4. High-Recall vs False-Alarm Curve with Chosen Cutoff
    fig, ax = plt.subplots(figsize=(8, 4.5))
    far = alert_curve_data["curve_far"]
    rec = alert_curve_data["curve_recalls"]
    chosen_t = alert_curve_data["chosen_threshold"]
    v_rec = alert_curve_data["val_high_recall"]
    v_far = alert_curve_data["val_false_alarm_rate"]

    ax.plot(far, rec, color="#e74c3c", lw=2, label="Validation Trade-off Curve")
    ax.scatter([v_far], [v_rec], color="black", s=100, zorder=5,
               label=f"Selected Cutoff (p={chosen_t:.2f}): Recall={v_rec:.1%}, FAR={v_far:.1%}")
    ax.axhline(alert_curve_data["target_recall"], color="gray", linestyle="--", label=f"Target Recall ({alert_curve_data['target_recall']:.0%})")
    ax.set_xlabel("False Alarm Rate (False High Alerts / Non-High Intervals)", fontweight="bold")
    ax.set_ylabel("High Congestion Recall (Safety Coverage)", fontweight="bold")
    ax.set_title("Operational Alert Calibration: High Recall vs. False Alarms (Validation Set)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.DOCS_FIGURES_DIR / "alert_threshold_curve.png", dpi=150)
    plt.close(fig)
