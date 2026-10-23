import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "nhanes_processed.csv"
MODEL_DIR = ROOT / "saved_models"
TARGETS = ["diabetes_label", "cvd_label", "hypertension_label", "metabolic_label", "stress_label"]
MODEL_NAMES = ["diabetes", "cvd", "hypertension", "metabolic", "stress"]
LEAKAGE_PRONE_FEATURES = {
    "diabetes": ["hba1c", "glucose_mgdl", "has_diabetes"],
    "cvd": ["age", "sbp", "cholesterol_mgdl"],
    "hypertension": ["sbp", "dbp", "has_hypertension"],
    "metabolic": ["bmi", "triglycerides", "hdl", "sbp", "glucose_mgdl"],
    "stress": ["stress_level"],
}


def train_model(X, y, target_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, aps, briers, losses, thresholds, f1s = [], [], [], [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipeline = _make_pipeline(_scale_pos_weight(y_tr))
        pipeline.fit(X_tr, y_tr)
        proba = pipeline.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, proba))
        aps.append(average_precision_score(y_val, proba))
        briers.append(brier_score_loss(y_val, proba))
        losses.append(log_loss(y_val, np.clip(proba, 1e-6, 1 - 1e-6)))
        threshold, f1 = _best_threshold(y_val, proba)
        thresholds.append(threshold)
        f1s.append(f1)

    pipeline = _make_pipeline(_scale_pos_weight(y))
    pipeline.fit(X, y)
    return pipeline, {
        "target": target_name,
        "mean_auc_roc": float(np.mean(aucs)),
        "mean_auc_pr": float(np.mean(aps)),
        "std_auc_roc": float(np.std(aucs)),
        "mean_brier": float(np.mean(briers)),
        "mean_log_loss": float(np.mean(losses)),
        "recommended_threshold": float(np.mean(thresholds)),
        "mean_f1_at_threshold": float(np.mean(f1s)),
        "positive_prevalence": float(y.mean()),
        "feature_count": int(X.shape[1]),
        "sample_count": int(len(y)),
        "training_method": "regularized_xgboost_scale_pos_weight_cv",
    }


def _make_pipeline(scale_pos_weight):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=220,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2.0,
            reg_alpha=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _scale_pos_weight(y):
    counts = y.value_counts()
    if len(counts) < 2 or counts.get(1, 0) == 0:
        return 1.0
    return float(counts.get(0, 0) / counts.get(1, 1))


def _best_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5, f1_score(y_true, proba >= 0.5)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-9)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def leakage_aware_columns(feature_cols, model_name):
    blocked = set(LEAKAGE_PRONE_FEATURES.get(model_name, []))
    return [col for col in feature_cols if col not in blocked]


def main():
    parser = argparse.ArgumentParser(description="Train HealthTwin NHANES risk models.")
    parser.add_argument(
        "--leakage-aware",
        action="store_true",
        help="Drop target-defining biomarker columns for each target and save separate ablation metrics.",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in TARGETS + ["SEQN"]]
    X = df[feature_cols]
    metrics = []
    ablation_metrics = []
    model_card = {
        "dataset": "NHANES prototype feature table",
        "validation": "5-fold stratified internal cross-validation",
        "primary_warning": (
            "Some labels are threshold-derived from biomarkers. Report leakage-aware "
            "ablation metrics instead of internal CV metrics for credibility."
        ),
        "runtime_prediction_policy": (
            "The app blends model output with clinical heuristics until external "
            "calibration/validation data is available."
        ),
        "targets": {},
    }
    for target, name in zip(TARGETS, MODEL_NAMES):
        model, report = train_model(X, df[target], name)
        joblib.dump(model, MODEL_DIR / f"{name}_model.pkl")
        metrics.append(report)
        model_card["targets"][name] = {"internal_cv": report}
        print(f"[{name}] AUC-ROC {report['mean_auc_roc']:.4f} | AUC-PR {report['mean_auc_pr']:.4f}")

        if args.leakage_aware:
            ablation_cols = leakage_aware_columns(feature_cols, name)
            ablation_model, ablation_report = train_model(df[ablation_cols], df[target], name)
            ablation_report["removed_features"] = LEAKAGE_PRONE_FEATURES.get(name, [])
            ablation_report["features_used"] = ablation_cols
            joblib.dump(ablation_model, MODEL_DIR / f"{name}_leakage_aware_model.pkl")
            ablation_metrics.append(ablation_report)
            model_card["targets"][name]["leakage_aware_cv"] = ablation_report
            print(
                f"[{name} leakage-aware] AUC-ROC {ablation_report['mean_auc_roc']:.4f} "
                f"| AUC-PR {ablation_report['mean_auc_pr']:.4f}"
            )

    (MODEL_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODEL_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    if args.leakage_aware:
        (MODEL_DIR / "training_metrics_leakage_aware.json").write_text(json.dumps(ablation_metrics, indent=2), encoding="utf-8")
    print("All five NHANES models trained.")


if __name__ == "__main__":
    main()
