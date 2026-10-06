"""
Data Acquisition, Cleaning & Provenance Pipeline
=================================================
Automated pipeline for:
1. Downloading public environmental datasets from public GitHub mirrors:
   - Central Pollution Control Board (CPCB), India River Monitoring Data
   - Kaggle Global Water Potability Laboratory Dataset
2. Performing domain-aware environmental data cleaning & validation:
   - Fixing column-swap corruption in CPCB dataset (pH vs Conductivity)
   - Removing/flagging physically impossible sensor anomalies
   - Ensuring zero data leakage in preprocessing
3. Generating physics-informed synthetic benchmark dataset for IoT sensor suite
4. Exporting comprehensive dataset summary statistics and cleaning audits

Usage:
    python src/data_pipeline.py
"""

import os
import sys
import json
import urllib.request
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from src.calculate_wqi import FEATURES, TARGET

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset Sources & Mirror URLs (Accurate Provenance)
# ---------------------------------------------------------------------------
DATA_SOURCES = {
    "cpcb_india": {
        "dataset_name": "Indian Water Quality Data (CPCB)",
        "original_source": "Central Pollution Control Board (CPCB), Ministry of Environment, Forest & Climate Change, Government of India",
        "retrieval_mirror_url": "https://raw.githubusercontent.com/aditikhatri/-Indian-water-quality-analysis-and-prediction/master/water_dataX.csv",
        "raw_file": os.path.join(DATA_DIR, "indian_cpcb_water_quality.csv"),
        "clean_file": os.path.join(DATA_DIR, "cleaned_indian_cpcb_water_quality.csv"),
        "license_or_access": "Open Government Data / Public Educational Research Mirror",
    },
    "kaggle_potability": {
        "dataset_name": "Water Potability Dataset (Kaggle)",
        "original_source": "Drinking Water Quality Studies published on Kaggle (Water Potability by Aditya Kadiwal)",
        "retrieval_mirror_url": "https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/main/water_potability.csv",
        "raw_file": os.path.join(DATA_DIR, "kaggle_water_potability.csv"),
        "clean_file": os.path.join(DATA_DIR, "cleaned_kaggle_water_potability.csv"),
        "license_or_access": "CC0 Public Domain (Kaggle Community Dataset)",
    },
    "synthetic_benchmark": {
        "dataset_name": "Physics-Calibrated Sensor Benchmark Dataset",
        "original_source": "Simulated multiparameter environmental sensor suite with latent environmental condition modeling",
        "retrieval_mirror_url": "Locally synthesized via src/generate_data.py",
        "raw_file": os.path.join(DATA_DIR, "water_quality.csv"),
        "clean_file": os.path.join(DATA_DIR, "water_quality.csv"),
        "license_or_access": "Project-Generated MIT License",
    },
}


def download_public_datasets() -> Dict[str, str]:
    """
    Download public datasets from verified GitHub repository mirrors.
    Preserves raw files locally for reproducible audit trails.
    """
    results = {}
    for key in ["cpcb_india", "kaggle_potability"]:
        cfg = DATA_SOURCES[key]
        raw_path = cfg["raw_file"]
        url = cfg["retrieval_mirror_url"]

        if os.path.exists(raw_path):
            print(f"[Provenance Check] {cfg['dataset_name']} raw file already exists at {raw_path}")
            results[key] = raw_path
            continue

        print(f"[Download] Fetching {cfg['dataset_name']} from mirror: {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EVS-Semester-3-Academic-Agent)"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                content = resp.read()
                with open(raw_path, "wb") as f:
                    f.write(content)
            print(f"  -> Saved raw copy to {raw_path}")
            results[key] = raw_path
        except Exception as e:
            print(f"  [!] Failed to download from mirror {url}: {e}")
            results[key] = None
    return results


def clean_indian_cpcb_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform rigorous, domain-aware environmental data cleaning on CPCB data.

    INVESTIGATION OF KNOWN SOURCE DATA ANOMALY:
    In the raw mirror file, rows 1901 to 1990 (89 monitoring records from 2003-2004)
    contain an inversion where the 'PH' column contains electrical conductivity values
    (e.g., 239, 442, up to 67,115 µmhos/cm for station 1435 in Vapi), while the
    'CONDUCTIVITY' column contains the valid pH values (e.g., 6.2, 6.7, 5.0).

    CLEANING PROCEDURE:
    1. Parse numeric columns and strip whitespace/special characters.
    2. Swap columns back for rows matching (pH > 14.0 and Conductivity in [0, 14.0]).
    3. Apply environmental surface water boundaries:
       - pH in [2.0, 12.0] (natural surface waters; flag pH < 2.0 or > 12.0 as NaN).
       - Temperature in [0.0, 50.0] °C.
       - Dissolved Oxygen in [0.0, 20.0] mg/L.
       - Conductivity >= 0 µS/cm.
       - BOD in [0.0, 1000.0] mg/L.
       - Nitrate in [0.0, 500.0] mg/L.
    """
    raw_path = DATA_SOURCES["cpcb_india"]["raw_file"]
    clean_path = DATA_SOURCES["cpcb_india"]["clean_file"]

    if not os.path.exists(raw_path):
        download_public_datasets()

    df_raw = pd.read_csv(raw_path, encoding="latin1")

    clean_df = pd.DataFrame()
    clean_df["Station_Code"] = df_raw["STATION CODE"].astype(str).str.strip()
    clean_df["Location"] = df_raw["LOCATIONS"].astype(str).str.strip()
    clean_df["State"] = df_raw["STATE"].astype(str).str.strip()
    clean_df["Temperature"] = pd.to_numeric(df_raw["Temp"], errors="coerce")
    clean_df["Dissolved_Oxygen"] = pd.to_numeric(df_raw["D.O. (mg/l)"], errors="coerce")
    clean_df["pH"] = pd.to_numeric(df_raw["PH"], errors="coerce")

    # Locate conductivity column regardless of encoding variances
    cond_col = [c for c in df_raw.columns if "CONDUCTIVITY" in c][0]
    clean_df["Conductivity"] = pd.to_numeric(df_raw[cond_col], errors="coerce")
    clean_df["BOD"] = pd.to_numeric(df_raw["B.O.D. (mg/l)"], errors="coerce")
    clean_df["Nitrate"] = pd.to_numeric(df_raw["NITRATENAN N+ NITRITENANN (mg/l)"], errors="coerce")
    clean_df["Fecal_Coliform"] = pd.to_numeric(df_raw["FECAL COLIFORM (MPN/100ml)"], errors="coerce")
    clean_df["Total_Coliform"] = pd.to_numeric(df_raw["TOTAL COLIFORM (MPN/100ml)Mean"], errors="coerce")
    clean_df["Year"] = pd.to_numeric(df_raw["year"], errors="coerce")

    audit = {
        "raw_rows": int(len(clean_df)),
        "raw_ph_max_before_clean": float(clean_df["pH"].max()),
        "raw_ph_mean_before_clean": float(clean_df["pH"].mean()),
    }

    # Step 1: Fix swapped columns
    swap_mask = (clean_df["pH"] > 14.0) & (clean_df["Conductivity"] >= 0.0) & (clean_df["Conductivity"] <= 14.0)
    swapped_count = int(swap_mask.sum())

    temp_ph = clean_df.loc[swap_mask, "pH"].copy()
    clean_df.loc[swap_mask, "pH"] = clean_df.loc[swap_mask, "Conductivity"]
    clean_df.loc[swap_mask, "Conductivity"] = temp_ph

    audit["column_swapped_rows_repaired"] = swapped_count

    # Step 2: Validate environmental physical domains
    invalid_ph = (clean_df["pH"] < 2.0) | (clean_df["pH"] > 12.0)
    audit["invalid_ph_count"] = int(invalid_ph.sum())
    clean_df.loc[invalid_ph, "pH"] = np.nan

    invalid_temp = (clean_df["Temperature"] < 0.0) | (clean_df["Temperature"] > 50.0)
    audit["invalid_temp_count"] = int(invalid_temp.sum())
    clean_df.loc[invalid_temp, "Temperature"] = np.nan

    invalid_do = (clean_df["Dissolved_Oxygen"] < 0.0) | (clean_df["Dissolved_Oxygen"] > 20.0)
    audit["invalid_do_count"] = int(invalid_do.sum())
    clean_df.loc[invalid_do, "Dissolved_Oxygen"] = np.nan

    invalid_cond = clean_df["Conductivity"] < 0.0
    audit["invalid_conductivity_count"] = int(invalid_cond.sum())
    clean_df.loc[invalid_cond, "Conductivity"] = np.nan

    invalid_bod = (clean_df["BOD"] < 0.0) | (clean_df["BOD"] > 1000.0)
    audit["invalid_bod_count"] = int(invalid_bod.sum())
    clean_df.loc[invalid_bod, "BOD"] = np.nan

    invalid_nitrate = (clean_df["Nitrate"] < 0.0) | (clean_df["Nitrate"] > 500.0)
    audit["invalid_nitrate_count"] = int(invalid_nitrate.sum())
    clean_df.loc[invalid_nitrate, "Nitrate"] = np.nan

    audit["clean_ph_min"] = float(clean_df["pH"].min())
    audit["clean_ph_mean"] = float(clean_df["pH"].mean())
    audit["clean_ph_max"] = float(clean_df["pH"].max())

    # Save cleaned CPCB dataset
    clean_df.to_csv(clean_path, index=False)
    print(f"[CPCB Cleaned] Corrected {swapped_count} swapped rows and {audit['invalid_ph_count']} invalid pH values.")
    print(f"  -> Cleaned pH range: {audit['clean_ph_min']:.2f} to {audit['clean_ph_max']:.2f} (mean: {audit['clean_ph_mean']:.2f})")
    print(f"  -> Saved cleaned file to {clean_path}")

    return clean_df, audit


def clean_kaggle_potability_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardize Kaggle Water Potability dataset WITHOUT premature imputation.

    CRITICAL LEAKAGE PREVENTION:
    Missing values (in ph, Sulfate, Trihalomethanes) are preserved in the cleaned
    dataset so that training/test splits happen FIRST. Imputation will be fitted
    strictly on training splits inside an sklearn Pipeline.
    """
    raw_path = DATA_SOURCES["kaggle_potability"]["raw_file"]
    clean_path = DATA_SOURCES["kaggle_potability"]["clean_file"]

    if not os.path.exists(raw_path):
        download_public_datasets()

    df = pd.read_csv(raw_path)

    # Cast numeric columns cleanly
    numeric_cols = [
        "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity", "Potability"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure integer binary target
    df["Potability"] = df["Potability"].astype(int)

    null_summary = df.isnull().sum().to_dict()

    audit = {
        "total_rows": int(len(df)),
        "features_count": int(df.shape[1] - 1),
        "missing_values_by_column": null_summary,
        "potable_count": int((df["Potability"] == 1).sum()),
        "non_potable_count": int((df["Potability"] == 0).sum()),
        "potability_ratio": float(df["Potability"].mean()),
    }

    # Save without leakage (nulls preserved for pipeline imputer)
    df.to_csv(clean_path, index=False)
    print(f"[Kaggle Cleaned] Standardized {len(df)} records without premature imputation (Zero data leakage).")
    print(f"  -> Missing counts: ph={null_summary['ph']}, Sulfate={null_summary['Sulfate']}, Trihalomethanes={null_summary['Trihalomethanes']}")
    print(f"  -> Saved to {clean_path}")

    return df, audit


def ensure_benchmark_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate or verify physics-calibrated synthetic benchmark dataset."""
    from src.generate_data import generate_synthetic_data
    benchmark_path = DATA_SOURCES["synthetic_benchmark"]["clean_file"]

    if not os.path.exists(benchmark_path):
        print(f"[Benchmark] Generating {n_samples} physics-calibrated samples...")
        df = generate_synthetic_data(n_samples=n_samples, seed=seed)
        df.to_csv(benchmark_path, index=False)
    else:
        df = pd.read_csv(benchmark_path)
        print(f"[Benchmark] Loaded existing benchmark dataset ({len(df)} samples)")
    return df


def generate_provenance_and_summary() -> Dict[str, Any]:
    """
    Generate complete dataset summary and provenance report JSON.
    Used by Streamlit dashboard and project documentation.
    """
    download_public_datasets()
    cpcb_df, cpcb_audit = clean_indian_cpcb_data()
    kaggle_df, kaggle_audit = clean_kaggle_potability_data()
    syn_df = ensure_benchmark_dataset()

    summary = {
        "datasets": {
            "cpcb_india": {
                "name": DATA_SOURCES["cpcb_india"]["dataset_name"],
                "origin": DATA_SOURCES["cpcb_india"]["original_source"],
                "mirror_url": DATA_SOURCES["cpcb_india"]["retrieval_mirror_url"],
                "access": DATA_SOURCES["cpcb_india"]["license_or_access"],
                "rows": int(len(cpcb_df)),
                "columns": cpcb_df.columns.tolist(),
                "cleaning_audit": cpcb_audit,
                "statistics": cpcb_df[["Temperature", "Dissolved_Oxygen", "pH", "Conductivity", "BOD", "Nitrate"]].describe().round(3).to_dict(),
            },
            "kaggle_potability": {
                "name": DATA_SOURCES["kaggle_potability"]["dataset_name"],
                "origin": DATA_SOURCES["kaggle_potability"]["original_source"],
                "mirror_url": DATA_SOURCES["kaggle_potability"]["retrieval_mirror_url"],
                "access": DATA_SOURCES["kaggle_potability"]["license_or_access"],
                "rows": int(len(kaggle_df)),
                "columns": kaggle_df.columns.tolist(),
                "audit": kaggle_audit,
                "statistics": kaggle_df.describe().round(3).to_dict(),
            },
            "synthetic_benchmark": {
                "name": DATA_SOURCES["synthetic_benchmark"]["dataset_name"],
                "origin": DATA_SOURCES["synthetic_benchmark"]["original_source"],
                "mirror_url": DATA_SOURCES["synthetic_benchmark"]["retrieval_mirror_url"],
                "access": DATA_SOURCES["synthetic_benchmark"]["license_or_access"],
                "rows": int(len(syn_df)),
                "columns": syn_df.columns.tolist(),
                "description": "Correlated multiparameter simulation matching IoT field sensors (pH, Turbidity, DO, Temp, Conductivity, TDS).",
                "statistics": syn_df.describe().round(3).to_dict(),
            },
        }
    }

    summary_file = os.path.join(DATA_DIR, "dataset_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Generated and saved comprehensive summary to {summary_file}")
    return summary


if __name__ == "__main__":
    generate_provenance_and_summary()
