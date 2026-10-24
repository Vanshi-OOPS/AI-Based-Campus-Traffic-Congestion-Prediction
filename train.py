"""Phase 2 Orchestrator: Baseline & Model Training, Evaluation, and Serialization.

Single-command entry point:
- Fits baselines and ML pipelines strictly on TRAIN.
- Selects winning models strictly on VALIDATION.
- Evaluates selected models and baselines on TEST exactly once.
- Calibrates operational High-alert probability threshold.
- Computes cluster bootstrap 95% CIs (resampling whole mornings).
- Slices test metrics by weather and event (anecdotal flags for < 5 mornings).
- Generates SHAP explainability artifacts and publication figures.
- Serializes complete model artifacts and run metadata into models/.
"""

import sys
from pathlib import Path
import json
import hashlib
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
import sklearn
import xgboost
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config as cfg
from src.baselines import (
    MajorityClassBaseline,
    VehicleCountRuleBaseline,
    CountAndWeatherRuleBaseline,
    HistoricalTimeSlotBaseline,
)
from src.train_models import run_model_training_and_selection
from src.evaluate import (
    compute_classification_metrics,
    compute_whole_morning_bootstrap_ci,
    compute_condition_slices,
    tune_alert_threshold_on_val,
    evaluate_alert_threshold_on_test,
    evaluate_15min_advance_alert,
    generate_evaluation_figures,
)
from src.explain import TrafficExplainer


def compute_file_hash(filepath: Path) -> str:
    """Compute sha256 hash of a file for run provenance tracking."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def run_pipeline() -> None:
    print("=" * 80)
    print("STARTING PHASE 2: BASELINES, MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 80)

    # 1. Load Processed Data
    if not cfg.PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data not found at {cfg.PROCESSED_DATA_PATH}. Run Phase 1 first.")

    df = pd.read_csv(cfg.PROCESSED_DATA_PATH)
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    print(f"Loaded dataset: {len(df)} total rows across 180 days.")
    print(f"  TRAIN : {train_df['date'].nunique()} days ({len(train_df)} rows)")
    print(f"  VAL   : {val_df['date'].nunique()} days ({len(val_df)} rows)")
    print(f"  TEST  : {test_df['date'].nunique()} days ({len(test_df)} rows)")

    y_train_class = train_df["congestion_class"]
    y_val_class = val_df["congestion_class"]
    y_test_class = test_df["congestion_class"]

    y_train_score = train_df["congestion_score"]
    y_val_score = val_df["congestion_score"]
    y_test_score = test_df["congestion_score"]

    # -------------------------------------------------------------------------
    # 2. Fit Baselines on TRAIN
    # -------------------------------------------------------------------------
    print("\n--- Fitting Benchmark Baselines on TRAIN ---")
    baselines = {
        "Majority Class": MajorityClassBaseline().fit(train_df, y_train_class),
        "Vehicle Count Rule": VehicleCountRuleBaseline().fit(train_df, y_train_class),
        "Count + Weather Rule": CountAndWeatherRuleBaseline().fit(train_df, y_train_class),
        "Historical Time-Slot": HistoricalTimeSlotBaseline().fit(train_df, y_train_class),
    }
    for b_name, b_mod in baselines.items():
        if hasattr(b_mod, "t_low"):
            print(f"  [OK] {b_name} tuned cutoffs: Low < {b_mod.t_low:.1f}, High >= {b_mod.t_high:.1f}")
        else:
            print(f"  [OK] {b_name} fitted on train.")

    # -------------------------------------------------------------------------
    # 3. Train Models & Perform Model Selection on VALIDATION
    # -------------------------------------------------------------------------
    print("\n--- Training Supervised ML Models (Fit on TRAIN, Validated on VAL) ---")
    training_results = run_model_training_and_selection(train_df, val_df)

    print("\n" + "=" * 80)
    print("VALIDATION COMPARISON TABLE (Primary Metric: Macro-F1)")
    print("=" * 80)
    print(f"{'Model & Feature Set':<30} | {'Macro-F1':<9} | {'High Recall':<11} | {'High Prec':<9} | {'False Highs':<11}")
    print("-" * 80)

    # Baselines on Val
    for b_name, b_mod in baselines.items():
        b_preds = b_mod.predict(val_df)
        b_m = compute_classification_metrics(y_val_class, b_preds)
        print(f"{b_name + ' (Baseline)':<30} | {b_m['macro_f1']:<9.4f} | {b_m['per_class']['High']['recall']:<11.4f} | "
              f"{b_m['per_class']['High']['precision']:<9.4f} | {b_m['false_high_alerts']:<11}")

    # ML Classifiers on Val
    for m_key, m_val in training_results["classification_val"].items():
        print(f"{m_key:<30} | {m_val['macro_f1']:<9.4f} | {m_val['high_recall']:<11.4f} | "
              f"{m_val['high_precision']:<9.4f} | {m_val['false_high_alerts']:<11}")

    selected_clf_core_key = training_results["selected_classifier_core"]
    selected_clf_full_key = training_results["selected_classifier_full"]
    selected_reg_core_key = training_results["selected_regressor_core"]
    selected_reg_full_key = training_results["selected_regressor_full"]

    print("\n--- Selection Outcome (Governed by Validation Macro-F1) ---")
    print(f"  Winner Classifier (CORE): {selected_clf_core_key} "
          f"(Macro-F1: {training_results['classification_val'][selected_clf_core_key]['macro_f1']:.4f})")
    print(f"  Winner Classifier (FULL): {selected_clf_full_key} "
          f"(Macro-F1: {training_results['classification_val'][selected_clf_full_key]['macro_f1']:.4f})")
    print(f"  Winner Regressor (CORE): {selected_reg_core_key} "
          f"(MAE: {training_results['regression_val'][selected_reg_core_key]['mae']:.3f})")
    print(f"  Winner Regressor (FULL): {selected_reg_full_key} "
          f"(MAE: {training_results['regression_val'][selected_reg_full_key]['mae']:.3f})")

    best_clf_core = training_results["trained_classifiers"][selected_clf_core_key]
    best_clf_full = training_results["trained_classifiers"][selected_clf_full_key]
    best_reg_core = training_results["trained_regressors"][selected_reg_core_key]
    best_reg_full = training_results["trained_regressors"][selected_reg_full_key]
    adv_model = training_results["advance_alert_model"]

    # -------------------------------------------------------------------------
    # 4. Calibrate High-Alert Probability Threshold on VALIDATION
    # -------------------------------------------------------------------------
    print("\n--- Calibrating Operational High-Alert Threshold on VALIDATION (~90% Recall) ---")
    alert_calibration = tune_alert_threshold_on_val(best_clf_full, val_df, target_high_recall=cfg.ALERT_TARGET_HIGH_RECALL)
    chosen_threshold = alert_calibration["chosen_threshold"]
    print(f"  Selected Cutoff: p(High) >= {chosen_threshold:.3f}")
    print(f"  Validation High Recall: {alert_calibration['val_high_recall']:.1%} | False Alarm Rate: {alert_calibration['val_false_alarm_rate']:.1%}")

    # -------------------------------------------------------------------------
    # 5. TEST SET EVALUATION (Scored exactly once)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION (Unseen Chronological Split: 27 Days, 675 Rows)")
    print("=" * 80)

    test_eval_results = {
        "baselines": {},
        "classifiers": {},
        "regressors": {},
        "alert_calibration": alert_calibration,
        "slices": {},
    }

    # Evaluate Baselines on Test
    best_baseline_name = "Count + Weather Rule"
    best_baseline_cm = None

    for b_name, b_mod in baselines.items():
        preds = b_mod.predict(test_df)
        m = compute_classification_metrics(y_test_class, preds)
        ci = compute_whole_morning_bootstrap_ci(test_df, preds, seed=cfg.RANDOM_SEED)
        m.update(ci)
        test_eval_results["baselines"][b_name] = m
        if b_name == best_baseline_name:
            best_baseline_cm = np.array(m["confusion_matrix"])

    # Evaluate Selected Classifiers on Test
    clf_test_preds = {}
    for f_set, model, key in [("CORE", best_clf_core, selected_clf_core_key), ("FULL", best_clf_full, selected_clf_full_key)]:
        preds = model.predict(test_df)
        clf_test_preds[f_set] = preds
        m = compute_classification_metrics(y_test_class, preds)
        ci = compute_whole_morning_bootstrap_ci(test_df, preds, seed=cfg.RANDOM_SEED)
        m.update(ci)
        test_eval_results["classifiers"][f"{key}_test"] = m

    # Evaluate Regressors on Test
    reg_test_preds = {}
    for f_set, model, key in [("CORE", best_reg_core, selected_reg_core_key), ("FULL", best_reg_full, selected_reg_full_key)]:
        preds = model.predict(test_df)
        reg_test_preds[f_set] = preds
        mae = float(mean_absolute_error(y_test_score, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test_score, preds)))
        r2 = float(r2_score(y_test_score, preds))
        test_eval_results["regressors"][f"{key}_test"] = {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 4),
        }

    # Operational Alert Threshold Evaluation on Test
    test_alert_eval = evaluate_alert_threshold_on_test(best_clf_full, test_df, chosen_threshold)
    test_eval_results["operational_alert_test"] = test_alert_eval

    # Condition Slices (Weather and Event) on Selected FULL Model
    best_full_preds = clf_test_preds["FULL"]
    slices = compute_condition_slices(test_df, best_full_preds)
    test_eval_results["slices"] = slices

    # 15-Minute Advance Alert Task Evaluation
    adv_eval = evaluate_15min_advance_alert(adv_model, baselines["Vehicle Count Rule"], test_df)
    test_eval_results["advance_15min_alert"] = adv_eval

    # Print Plain-Language Results Table
    print(f"{'Model / Baseline':<32} | {'Acc':<7} | {'Macro-F1 (95% CI)':<22} | {'High Recall (95% CI)':<22} | {'False Highs':<11} | {'Skip Err':<9}")
    print("-" * 115)

    for b_name, m in test_eval_results["baselines"].items():
        f1_ci = f"{m['macro_f1']:.3f} [{m['macro_f1_ci95'][0]:.2f}-{m['macro_f1_ci95'][1]:.2f}]"
        rec_ci = f"{m['per_class']['High']['recall']:.3f} [{m['high_recall_ci95'][0]:.2f}-{m['high_recall_ci95'][1]:.2f}]"
        print(f"{b_name + ' (Base)':<32} | {m['accuracy']:<7.3f} | {f1_ci:<22} | {rec_ci:<22} | {m['false_high_alerts']:<11} | {m['class_skipping']['total_skipping']:<9}")

    for k, m in test_eval_results["classifiers"].items():
        label = f"Selected {k.replace('_test', '')}"
        f1_ci = f"{m['macro_f1']:.3f} [{m['macro_f1_ci95'][0]:.2f}-{m['macro_f1_ci95'][1]:.2f}]"
        rec_ci = f"{m['per_class']['High']['recall']:.3f} [{m['high_recall_ci95'][0]:.2f}-{m['high_recall_ci95'][1]:.2f}]"
        print(f"{label:<32} | {m['accuracy']:<7.3f} | {f1_ci:<22} | {rec_ci:<22} | {m['false_high_alerts']:<11} | {m['class_skipping']['total_skipping']:<9}")

    print("\n--- Regression Performance (Test Set: Congestion Score 0 - 100) ---")
    for k, m in test_eval_results["regressors"].items():
        print(f"  {k:<32}: MAE = {m['mae']:.2f} points | RMSE = {m['rmse']:.2f} | R² = {m['r2']:.4f}")

    print("\n--- Operational High-Alert Threshold Result on TEST ---")
    print(f"  Applied Probability Cutoff : p(High) >= {test_alert_eval['applied_threshold']:.3f}")
    print(f"  High Recall Achieved       : {test_alert_eval['test_high_recall']:.1%} (caught {int(test_alert_eval['test_high_recall'] * 74)} of 74 High intervals)")
    print(f"  High Precision             : {test_alert_eval['test_high_precision']:.1%}")
    print(f"  False Alarms Generated     : {test_alert_eval['test_false_alarms']} (FAR: {test_alert_eval['test_false_alarm_rate']:.1%})")

    print("\n--- Sliced Performance (Weather & Events on TEST) ---")
    print(f"{'Condition':<18} | {'Mornings':<9} | {'Intervals':<10} | {'Status Note':<32} | {'Accuracy':<9} | {'High Rec':<9}")
    print("-" * 95)
    for w, s in slices["weather"].items():
        print(f"{w:<18} | {s['mornings']:<9} | {s['intervals']:<10} | {s['status']:<32} | {s['accuracy']:<9.3f} | {s['high_recall']:<9.3f}")
    for ev, s in slices["event"].items():
        print(f"{ev:<18} | {s['mornings']:<9} | {s['intervals']:<10} | {s['status']:<32} | {s['accuracy']:<9.3f} | {s['high_recall']:<9.3f}")

    print("\n--- 15-Minute Advance Alert Task Results (Predicting High 15-min Ahead) ---")
    print(f"{'Strategy / Model':<35} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'False Alerts':<12}")
    print("-" * 80)
    for k, m in adv_eval.items():
        if isinstance(m, dict):
            print(f"{k:<35} | {m['precision']:<10.3f} | {m['recall']:<10.3f} | {m['f1']:<10.3f} | {m['false_alerts']:<12}")

    # -------------------------------------------------------------------------
    # 6. Explainability & Visualization Artifacts
    # -------------------------------------------------------------------------
    print("\n--- Generating Explainability (SHAP) and Figures ---")
    explainer = TrafficExplainer(
        regressor_pipeline=best_reg_full,
        classifier_pipeline=best_clf_full,
        background_df=train_df,
        feature_set="FULL",
    )
    explainer.generate_shap_summary_plot(val_df)

    core_vs_full_chart_data = {
        "CORE": {
            "accuracy": test_eval_results["classifiers"][f"{selected_clf_core_key}_test"]["accuracy"],
            "macro_f1": test_eval_results["classifiers"][f"{selected_clf_core_key}_test"]["macro_f1"],
            "high_recall": test_eval_results["classifiers"][f"{selected_clf_core_key}_test"]["per_class"]["High"]["recall"],
        },
        "FULL": {
            "accuracy": test_eval_results["classifiers"][f"{selected_clf_full_key}_test"]["accuracy"],
            "macro_f1": test_eval_results["classifiers"][f"{selected_clf_full_key}_test"]["macro_f1"],
            "high_recall": test_eval_results["classifiers"][f"{selected_clf_full_key}_test"]["per_class"]["High"]["recall"],
        },
    }

    best_model_cm = np.array(test_eval_results["classifiers"][f"{selected_clf_full_key}_test"]["confusion_matrix"])

    generate_evaluation_figures(
        best_baseline_cm=best_baseline_cm,
        best_model_cm=best_model_cm,
        best_baseline_name=best_baseline_name,
        best_model_name=selected_clf_full_key,
        test_df=test_df,
        reg_preds=reg_test_preds["FULL"],
        core_vs_full_metrics=core_vs_full_chart_data,
        alert_curve_data=alert_calibration,
    )
    print(f"  [OK] Saved all publication figures to: {cfg.DOCS_FIGURES_DIR}")

    # -------------------------------------------------------------------------
    # 7. Serialize Artifacts to models/
    # -------------------------------------------------------------------------
    print("\n--- Serializing Models & Pipelines to models/ ---")
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_clf_core, cfg.MODELS_DIR / "congestion_classifier.joblib")
    joblib.dump(best_clf_full, cfg.MODELS_DIR / "congestion_classifier_full.joblib")
    joblib.dump(best_reg_core, cfg.MODELS_DIR / "congestion_regressor.joblib")
    joblib.dump(best_reg_full, cfg.MODELS_DIR / "congestion_regressor_full.joblib")
    joblib.dump(adv_model, cfg.MODELS_DIR / "advance_alert_model.joblib")
    joblib.dump(explainer, cfg.MODELS_DIR / "traffic_explainer.joblib")

    # feature_columns.json
    feature_meta = {
        "core_features": cfg.CORE_FEATURE_COLS,
        "full_features": cfg.FULL_FEATURE_COLS,
        "class_order": cfg.CONGESTION_CLASSES,
        "weather_ordinal_map": cfg.WEATHER_ORDINAL_MAP,
        "selected_classifier_core": selected_clf_core_key,
        "selected_classifier_full": selected_clf_full_key,
        "selected_regressor_core": selected_reg_core_key,
        "selected_regressor_full": selected_reg_full_key,
        "alert_threshold": chosen_threshold,
        "split_counts": {
            "train_days": int(train_df["date"].nunique()),
            "val_days": int(val_df["date"].nunique()),
            "test_days": int(test_df["date"].nunique()),
        },
    }
    with open(cfg.MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # eval_metrics.json
    with open(cfg.MODELS_DIR / "eval_metrics.json", "w") as f:
        json.dump(test_eval_results, f, indent=2)

    # run_metadata.json
    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": cfg.RANDOM_SEED,
        "data_hash_processed": compute_file_hash(cfg.PROCESSED_DATA_PATH),
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "config_snapshot": {
            "num_days": cfg.NUM_DAYS,
            "intervals_per_morning": cfg.INTERVALS_PER_MORNING,
            "base_capacity": cfg.BASE_CAPACITY,
            "queue_weight": cfg.QUEUE_WEIGHT,
            "ar1_phi": cfg.AR1_PHI,
            "score_low_max": cfg.SCORE_LOW_MAX,
            "score_moderate_max": cfg.SCORE_MODERATE_MAX,
        },
    }
    with open(cfg.MODELS_DIR / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"  [OK] Saved all model binaries and metadata to: {cfg.MODELS_DIR}")
    print("=" * 80)
    print("PHASE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
