from pathlib import Path

import pandas as pd
import requests


# NHANES 2017-March 2020 prepandemic files use the "_P" suffix and are
# published under the CDC public 2017 data-file path.
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COMPONENTS = {
    "P_DEMO": "Demographics",
    "P_BMX": "Body measures",
    "P_BPXO": "Blood pressure - oscillometric",
    "P_GHB": "HbA1c",
    "P_GLU": "Fasting glucose",
    "P_TCHOL": "Total cholesterol",
    "P_HDL": "HDL cholesterol",
    "P_TRIGLY": "Triglycerides and LDL",
    "P_CBC": "Complete blood count",
    "P_SMQ": "Smoking",
    "P_ALQ": "Alcohol",
    "P_PAQ": "Physical activity",
    "P_SLQ": "Sleep",
    "P_DIQ": "Diabetes",
    "P_BPQ": "Hypertension",
    "P_MCQ": "Medical conditions",
    "P_DPQ": "Depression",
    "P_HSQ": "General health",
}


def download_components():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for code, label in COMPONENTS.items():
        url = f"{BASE_URL}/{code}.xpt"
        path = DATA_DIR / f"{code}.xpt"
        if not path.exists():
            print(f"Downloading {label}: {code}.xpt")
            response = requests.get(url, timeout=60)
            if response.status_code == 404:
                raise RuntimeError(
                    f"CDC file not found: {url}. Check whether the component code changed on the NHANES portal."
                )
            response.raise_for_status()
            path.write_bytes(response.content)
        paths.append(path)
    return paths


def merge_components(paths=None):
    paths = paths or list(DATA_DIR.glob("P_*.xpt"))
    merged = None
    for path in paths:
        frame = pd.read_sas(path, format="xport")
        frame.columns = [col.upper() for col in frame.columns]
        merged = frame if merged is None else merged.merge(frame, on="SEQN", how="outer")
    output = DATA_DIR / "nhanes_merged.csv"
    merged.to_csv(output, index=False)
    print(f"Merged NHANES table saved to {output} with shape {merged.shape}")
    return output


if __name__ == "__main__":
    merge_components(download_components())
