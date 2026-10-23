from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def engineer_features(input_path=DATA_DIR / "nhanes_merged.csv"):
    df = pd.read_csv(input_path)
    out = pd.DataFrame()
    out["SEQN"] = df["SEQN"]
    out["age"] = _col(df, "RIDAGEYR")
    out["gender_code"] = _col(df, "RIAGENDR").map({1: 1, 2: 0})
    out["height_cm"] = _col(df, "BMXHT")
    out["weight_kg"] = _col(df, "BMXWT")
    out["bmi"] = _col(df, "BMXBMI")
    out["sleep_hrs"] = _first_available(df, ["SLD012", "SLD010H", "SLQ300"])
    out["exercise_days"] = _first_available(df, ["PAD680", "PAQ650", "PAQ665"]).apply(_activity_to_days)
    out["steps_daily"] = np.nan
    out["water_litres"] = np.nan
    out["smoker"] = _col(df, "SMQ020").map({1: 1, 2: 0, 7: np.nan, 9: np.nan})
    out["alcohol_units"] = _first_available(df, ["ALQ130", "ALQ121"])
    out["stress_level"] = _phq9_score(df)
    out["sbp"] = _mean_available(df, ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"])
    out["dbp"] = _mean_available(df, ["BPXODI1", "BPXODI2", "BPXODI3", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])
    out["heart_rate"] = _first_available(df, ["BPXOPLS1", "BPXOPLS2", "BPXOPLS3", "BPXPLS"])
    out["spo2"] = np.nan
    out["temp_c"] = np.nan
    out["glucose_mgdl"] = _col(df, "LBXGLU")
    out["cholesterol_mgdl"] = _col(df, "LBXTC")
    out["hdl"] = _first_available(df, ["LBDHDD", "LBXHDD"])
    out["ldl"] = _first_available(df, ["LBDLDL", "LBDLDLM"])
    out["triglycerides"] = _col(df, "LBXTR")
    out["hba1c"] = _col(df, "LBXGH")
    out["hemoglobin"] = _col(df, "LBXHGB")
    out["has_diabetes"] = _col(df, "DIQ010").map({1: 1, 2: 0, 3: 0, 7: np.nan, 9: np.nan})
    out["has_hypertension"] = _col(df, "BPQ020").map({1: 1, 2: 0, 7: np.nan, 9: np.nan})
    out["has_asthma"] = _col(df, "MCQ010").map({1: 1, 2: 0, 7: np.nan, 9: np.nan})
    out = out.dropna(thresh=int(len(out.columns) * 0.55))
    output = DATA_DIR / "nhanes_features.csv"
    out.to_csv(output, index=False)
    print(f"Engineered features saved to {output} with shape {out.shape}")
    return output


def _activity_to_days(value):
    if pd.isna(value):
        return np.nan
    return max(0, min(7, round((600 - float(value)) / 90)))


def _col(df, name):
    if name in df:
        return df[name]
    return pd.Series(np.nan, index=df.index)


def _first_available(df, names):
    for name in names:
        if name in df:
            return df[name]
    return pd.Series(np.nan, index=df.index)


def _mean_available(df, names):
    present = [name for name in names if name in df]
    if not present:
        return pd.Series(np.nan, index=df.index)
    return df[present].mean(axis=1)


def _phq9_score(df):
    cols = [c for c in ["DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090"] if c in df]
    if not cols:
        return np.nan
    return df[cols].replace({7: np.nan, 9: np.nan}).sum(axis=1)


if __name__ == "__main__":
    engineer_features()
