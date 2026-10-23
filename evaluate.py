import json
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, classification_report, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "saved_models"
DATA_PATH = ROOT / "data" / "nhanes_processed.csv"
TARGETS = {"diabetes": "diabetes_label", "cvd": "cvd_label", "hypertension": "hypertension_label", "metabolic": "metabolic_label", "stress": "stress_label"}
LEAKAGE_PRONE_FEATURES = {
    "diabetes": ["hba1c", "glucose_mgdl", "has_diabetes"],
    "cvd": ["age", "sbp", "cholesterol_mgdl"],
    "hypertension": ["sbp", "dbp", "has_hypertension"],
    "metabolic": ["bmi", "triglycerides", "hdl", "sbp", "glucose_mgdl"],
    "stress": ["stress_level"],
}


def evaluate(data_path=DATA_PATH, leakage_aware=False):
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in set(TARGETS.values()) | {"SEQN"}]
    reports = {}
    for name, target in TARGETS.items():
        if target not in df:
            reports[name] = {"error": f"Missing target column: {target}"}
            continue
        suffix = "_leakage_aware_model.pkl" if leakage_aware else "_model.pkl"
        model = joblib.load(MODEL_DIR / f"{name}{suffix}")
        cols = feature_cols
        removed = []
        if leakage_aware:
            removed = [col for col in LEAKAGE_PRONE_FEATURES.get(name, []) if col in cols]
            cols = [col for col in cols if col not in removed]
        expected = list(getattr(model, "feature_names_in_", cols))
        missing = [col for col in expected if col not in df]
        if missing:
            reports[name] = {"error": f"Missing feature columns: {', '.join(missing)}"}
            continue
        proba = model.predict_proba(df[expected])[:, 1]
        pred = (proba >= 0.5).astype(int)
        reports[name] = {
            "auc_roc": roc_auc_score(df[target], proba),
            "auc_pr": average_precision_score(df[target], proba),
            "brier": brier_score_loss(df[target], proba),
            "log_loss": log_loss(df[target], proba.clip(1e-6, 1 - 1e-6)),
            "classification_report": classification_report(df[target], pred, output_dict=True),
            "removed_features": removed,
            "sample_count": int(len(df)),
        }
    print(json.dumps(reports, indent=2))
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved HealthTwin models on an internal or external CSV.")
    parser.add_argument("--data", default=str(DATA_PATH), help="CSV path containing feature and target columns.")
    parser.add_argument("--leakage-aware", action="store_true", help="Report target-defining features removed where possible.")
    args = parser.parse_args()
    evaluate(Path(args.data), args.leakage_aware)
