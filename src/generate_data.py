"""
Synthetic Water Quality Data Generator

Generates ~3,000 rows of realistic, correlated water-quality data
using a latent quality_condition variable to drive parameter relationships.

Usage:
    python src/generate_data.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path so we can import calculate_wqi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.calculate_wqi import calculate_wqi_row, FEATURES, TARGET


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SAMPLES = 3000
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "water_quality.csv")


def generate_synthetic_data(n_samples: int = NUM_SAMPLES,
                            seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic water-quality data with realistic correlations.

    A latent `quality_condition` variable (0=poor, 1=good) drives all
    six parameters, creating natural inter-parameter relationships.
    Controlled noise ensures the ML problem is not perfectly deterministic.

    Parameters
    ----------
    n_samples : int - Number of rows to generate
    seed : int - Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns: pH, Turbidity, Dissolved Oxygen,
                                Temperature, Conductivity, TDS, WQI
    """
    rng = np.random.default_rng(seed)

    # Latent quality condition: 0 = very poor water, 1 = excellent water
    quality = rng.uniform(0, 1, n_samples)

    # --- pH ---
    # Good water: pH near 7 (neutral)
    # Poor water: pH drifts toward acidic (4.5) or alkaline (10)
    base_ph = 7.0 + (1 - quality) * rng.choice([-1, 1], n_samples) * rng.uniform(1.0, 3.0, n_samples)
    ph = base_ph + rng.normal(0, 0.3, n_samples)
    ph = np.clip(ph, 4.5, 10.0)

    # --- Turbidity (NTU) ---
    # Good water: low turbidity (0.5–10)
    # Poor water: high turbidity (30–100)
    turbidity = (1 - quality) * rng.uniform(30, 100, n_samples) + quality * rng.uniform(0.5, 10, n_samples)
    turbidity += rng.normal(0, 3, n_samples)
    turbidity = np.clip(turbidity, 0.5, 100.0)

    # --- Dissolved Oxygen (mg/L) ---
    # Good water: high DO (7–12)
    # Poor water: low DO (1–5)
    do = quality * rng.uniform(7, 12, n_samples) + (1 - quality) * rng.uniform(1, 5, n_samples)
    do += rng.normal(0, 0.5, n_samples)
    do = np.clip(do, 1.0, 12.0)

    # --- Temperature (°C) ---
    # Good water: moderate temperature (18–28)
    # Poor water: extreme temperature (10–15 or 33–40)
    moderate_temp = rng.uniform(18, 28, n_samples)
    # Randomly assign each sample to low-extreme or high-extreme
    use_low = rng.random(n_samples) < 0.5
    extreme_temp = np.where(use_low,
                            rng.uniform(10, 15, n_samples),
                            rng.uniform(33, 40, n_samples))
    temperature = quality * moderate_temp + (1 - quality) * extreme_temp
    temperature += rng.normal(0, 1.5, n_samples)
    temperature = np.clip(temperature, 10.0, 40.0)

    # --- Conductivity (µS/cm) and TDS (ppm) ---
    # These two are physically correlated (TDS ≈ 0.5–0.7 × Conductivity)
    # Good water: low conductivity (50–400)
    # Poor water: high conductivity (800–2000)
    conductivity = (1 - quality) * rng.uniform(800, 2000, n_samples) + quality * rng.uniform(50, 400, n_samples)
    conductivity += rng.normal(0, 50, n_samples)
    conductivity = np.clip(conductivity, 50.0, 2000.0)

    # TDS derived from conductivity with realistic conversion factor + noise
    tds_factor = rng.uniform(0.5, 0.7, n_samples)
    tds = conductivity * tds_factor + rng.normal(0, 20, n_samples)
    tds = np.clip(tds, 30.0, 1500.0)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "pH": np.round(ph, 2),
        "Turbidity": np.round(turbidity, 2),
        "Dissolved Oxygen": np.round(do, 2),
        "Temperature": np.round(temperature, 2),
        "Conductivity": np.round(conductivity, 2),
        "TDS": np.round(tds, 2),
    })

    # --- Calculate WQI target ---
    df[TARGET] = df.apply(calculate_wqi_row, axis=1).round(2)

    return df


def validate_data(df: pd.DataFrame) -> None:
    """Run quality checks on the generated dataset."""
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # Shape
    print(f"\nDataset shape: {df.shape}")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values")
    else:
        print(f"✗ Missing values found:\n{missing[missing > 0]}")

    # Duplicates
    n_dupes = df.duplicated().sum()
    print(f"{'✓' if n_dupes == 0 else '✗'} Duplicate rows: {n_dupes}")

    # Range checks
    print("\nParameter ranges:")
    for col in FEATURES:
        print(f"  {col:20s}  min={df[col].min():8.2f}  max={df[col].max():8.2f}")

    # WQI range
    wqi_min, wqi_max = df[TARGET].min(), df[TARGET].max()
    in_range = 0 <= wqi_min and wqi_max <= 100
    print(f"\n{'✓' if in_range else '✗'} WQI range: [{wqi_min:.2f}, {wqi_max:.2f}]")

    # Descriptive statistics
    print("\nDescriptive statistics:")
    print(df.describe().round(2).to_string())

    # Correlation check: TDS vs Conductivity
    corr = df["TDS"].corr(df["Conductivity"])
    print(f"\n✓ TDS–Conductivity correlation: {corr:.3f} (expected: high positive)")

    print("\n" + "=" * 60)


def main():
    """Generate, validate, and save the synthetic dataset."""
    print("Generating synthetic water-quality data...")
    df = generate_synthetic_data()

    validate_data(df)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Dataset saved to {OUTPUT_FILE}")
    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
