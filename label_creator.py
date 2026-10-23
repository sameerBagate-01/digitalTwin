from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def create_labels(input_path=DATA_DIR / "nhanes_features.csv"):
    df = pd.read_csv(input_path)
    df["diabetes_label"] = ((df["has_diabetes"] == 1) | (df["hba1c"] >= 6.5)).astype(int)
    df["hypertension_label"] = ((df["has_hypertension"] == 1) | (df["sbp"] >= 140)).astype(int)
    df["cvd_label"] = ((df.get("age", 0) >= 55) & ((df["sbp"] >= 140) | (df["cholesterol_mgdl"] >= 240))).astype(int)
    df["metabolic_label"] = (
        ((df["bmi"] >= 30).astype(int) + (df["triglycerides"] >= 150).astype(int) +
         (df["hdl"] < 40).astype(int) + (df["sbp"] >= 130).astype(int) +
         (df["glucose_mgdl"] >= 100).astype(int)) >= 3
    ).astype(int)
    df["stress_label"] = (df["stress_level"] >= 10).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include="number").columns:
        fill_value = df[col].median()
        df[col] = df[col].fillna(0 if pd.isna(fill_value) else fill_value)
    output = DATA_DIR / "nhanes_processed.csv"
    df.to_csv(output, index=False)
    feature_cols = [c for c in df.columns if c not in {"SEQN", "diabetes_label", "cvd_label", "hypertension_label", "metabolic_label", "stress_label"}]
    (DATA_DIR / "feature_columns.txt").write_text("\n".join(feature_cols), encoding="utf-8")
    print(f"Labelled dataset saved to {output} with shape {df.shape}")
    return output


if __name__ == "__main__":
    create_labels()
