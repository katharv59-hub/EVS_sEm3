"""
Data Sourcing and Preprocessing Pipeline
=========================================
Handles acquiring real-world water quality datasets from the internet,
cleaning and standardizing them, generating calibrated synthetic benchmark
data, and producing comprehensive exploratory data analysis (EDA) summaries.

Datasets Handled:
1. Real Internet Dataset A: CPCB (Central Pollution Control Board, India) Water Quality
2. Real Internet Dataset B: Kaggle Global Water Potability (3,276 water bodies)
3. Synthetic Benchmark Dataset: Physical-chemical latent simulation (3,000 samples)
"""

import os
import sys
import io
import json
import urllib.request
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.calculate_wqi import calculate_wqi_row, FEATURES, TARGET

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def download_real_datasets() -> dict:
    """Download public water quality datasets from open GitHub mirrors."""
    urls = {
        "kaggle_potability": (
            "https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/main/water_potability.csv",
            os.path.join(DATA_DIR, "kaggle_water_potability.csv")
        ),
        "indian_cpcb": (
            "https://raw.githubusercontent.com/aditikhatri/-Indian-water-quality-analysis-and-prediction/master/water_dataX.csv",
            os.path.join(DATA_DIR, "indian_cpcb_water_quality.csv")
        )
    }

    results = {}
    for name, (url, filepath) in urls.items():
        if os.path.exists(filepath):
            print(f"[{name}] Already exists at {filepath}")
            results[name] = filepath
            continue

        print(f"[{name}] Downloading from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read()
                with open(filepath, "wb") as f:
                    f.write(content)
            print(f"[{name}] Downloaded successfully to {filepath}")
            results[name] = filepath
        except Exception as e:
            print(f"[{name}] Download failed: {e}")
            results[name] = None
    return results


def clean_indian_cpcb_data() -> pd.DataFrame:
    """Clean and standardize the Indian CPCB Water Quality monitoring dataset."""
    filepath = os.path.join(DATA_DIR, "indian_cpcb_water_quality.csv")
    if not os.path.exists(filepath):
        download_real_datasets()

    df_raw = pd.read_csv(filepath, encoding="latin1")

    clean_df = pd.DataFrame()
    clean_df["Station_Code"] = df_raw["STATION CODE"].astype(str)
    clean_df["Location"] = df_raw["LOCATIONS"].astype(str)
    clean_df["State"] = df_raw["STATE"].astype(str)
    clean_df["Temperature"] = pd.to_numeric(df_raw["Temp"], errors="coerce")
    clean_df["Dissolved_Oxygen"] = pd.to_numeric(df_raw["D.O. (mg/l)"], errors="coerce")
    clean_df["pH"] = pd.to_numeric(df_raw["PH"], errors="coerce")

    cond_col = [c for c in df_raw.columns if "CONDUCTIVITY" in c][0]
    clean_df["Conductivity"] = pd.to_numeric(df_raw[cond_col], errors="coerce")
    clean_df["BOD"] = pd.to_numeric(df_raw["B.O.D. (mg/l)"], errors="coerce")
    clean_df["Nitrate"] = pd.to_numeric(df_raw["NITRATENAN N+ NITRITENANN (mg/l)"], errors="coerce")
    clean_df["Fecal_Coliform"] = pd.to_numeric(df_raw["FECAL COLIFORM (MPN/100ml)"], errors="coerce")
    clean_df["Total_Coliform"] = pd.to_numeric(df_raw["TOTAL COLIFORM (MPN/100ml)Mean"], errors="coerce")
    clean_df["Year"] = pd.to_numeric(df_raw["year"], errors="coerce")

    out_file = os.path.join(DATA_DIR, "cleaned_indian_cpcb_water_quality.csv")
    clean_df.to_csv(out_file, index=False)
    print(f"Saved cleaned Indian CPCB dataset to {out_file} ({clean_df.shape[0]} rows)")
    return clean_df


def clean_kaggle_potability_data() -> pd.DataFrame:
    """Clean and inspect Kaggle Water Potability dataset."""
    filepath = os.path.join(DATA_DIR, "kaggle_water_potability.csv")
    if not os.path.exists(filepath):
        download_real_datasets()

    df = pd.read_csv(filepath)
    # Impute missing values with column medians for modeling
    df_imputed = df.copy()
    for col in df_imputed.columns:
        if df_imputed[col].isnull().sum() > 0:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

    out_file = os.path.join(DATA_DIR, "cleaned_kaggle_water_potability.csv")
    df_imputed.to_csv(out_file, index=False)
    print(f"Saved cleaned Kaggle Potability dataset to {out_file} ({df_imputed.shape[0]} rows)")
    return df_imputed


def generate_benchmark_synthetic_data(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate calibrated synthetic water quality dataset with physics correlations."""
    from src.generate_data import generate_synthetic_data
    df = generate_synthetic_data(n_samples=n_samples, seed=seed)
    out_file = os.path.join(DATA_DIR, "water_quality.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved synthetic benchmark dataset to {out_file} ({df.shape[0]} rows)")
    return df


def generate_dataset_summary() -> dict:
    """Compute rich summary statistics and comparison for all datasets."""
    # Ensure datasets exist
    download_real_datasets()
    cpcb_df = clean_indian_cpcb_data()
    pot_df = clean_kaggle_potability_data()
    syn_df = pd.read_csv(os.path.join(DATA_DIR, "water_quality.csv"))

    summary = {
        "datasets": {
            "synthetic_benchmark": {
                "name": "Synthetic Sensor Benchmark Dataset",
                "source": "Physics-informed environmental simulation with latent condition modeling",
                "rows": int(syn_df.shape[0]),
                "columns": syn_df.columns.tolist(),
                "description": "Custom calibrated sensor dataset covering pH, Turbidity, DO, Temp, Conductivity, and TDS with computed WQI.",
                "statistics": syn_df.describe().round(3).to_dict()
            },
            "indian_cpcb": {
                "name": "Indian CPCB Water Quality Dataset",
                "source": "Central Pollution Control Board (CPCB), Ministry of Environment, Forest & Climate Change, India",
                "rows": int(cpcb_df.shape[0]),
                "columns": cpcb_df.columns.tolist(),
                "description": "Historical field station monitoring across rivers and water bodies in Indian states.",
                "statistics": cpcb_df[['Temperature', 'Dissolved_Oxygen', 'pH', 'Conductivity', 'BOD', 'Nitrate']].describe().round(3).to_dict()
            },
            "kaggle_potability": {
                "name": "Kaggle Water Potability Dataset",
                "source": "Global water quality and potability study (mirrored on Kaggle & GitHub)",
                "rows": int(pot_df.shape[0]),
                "columns": pot_df.columns.tolist(),
                "description": "Laboratory measurements of water chemistry metrics for predicting human drinkability (Potability: 0/1).",
                "statistics": pot_df.describe().round(3).to_dict()
            }
        }
    }

    summary_file = os.path.join(DATA_DIR, "dataset_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved dataset summary to {summary_file}")
    return summary


if __name__ == "__main__":
    download_real_datasets()
    clean_indian_cpcb_data()
    clean_kaggle_potability_data()
    generate_dataset_summary()
