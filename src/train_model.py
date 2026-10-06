"""
ML Model Training Pipeline

Trains a RandomForestRegressor to predict WQI from six water-quality
parameters. Saves the trained model and evaluation metrics.

Usage:
    python src/train_model.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.calculate_wqi import FEATURES, TARGET


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "water_quality.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_FILE = os.path.join(MODEL_DIR, "wqi_model.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_and_validate(filepath: str) -> pd.DataFrame:
    """Load the dataset and perform basic validation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            "Run 'python src/generate_data.py' first."
        )

    df = pd.read_csv(filepath)

    # Validate columns
    required_cols = FEATURES + [TARGET]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # Check for missing values
    null_count = df[required_cols].isnull().sum().sum()
    if null_count > 0:
        print(f"Warning: {null_count} missing values found. Dropping rows with nulls.")
        df = df.dropna(subset=required_cols)

    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def train_model(df: pd.DataFrame) -> dict:
    """
    Train a RandomForestRegressor and return results.

    Returns a dict with keys: model, metrics, feature_importances,
    X_test, y_test, y_pred
    """
    X = df[FEATURES]
    y = df[TARGET]

    # Verify no leakage — WQI must NOT be in features
    assert TARGET not in FEATURES, "Data leakage: WQI found in feature list!"

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"✓ Split: {len(X_train)} train / {len(X_test)} test")

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("Training RandomForestRegressor (200 trees)...")
    model.fit(X_train, y_train)
    print("✓ Model trained")

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Feature importances
    importances = dict(zip(FEATURES, model.feature_importances_.tolist()))

    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "feature_importances": importances,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return {
        "model": model,
        "metrics": metrics,
    }


def save_results(model, metrics: dict) -> None:
    """Save the trained model and metrics to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"✓ Model saved to {MODEL_FILE}")

    # Save metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {METRICS_FILE}")


def print_results(metrics: dict) -> None:
    """Print training results to console."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n  MAE  : {metrics['MAE']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f}")
    print(f"  R²   : {metrics['R2']:.4f}")

    print("\n  Feature Importances:")
    importances = metrics["feature_importances"]
    # Sort by importance descending
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 50)
        print(f"    {feat:20s}  {imp:.4f}  {bar}")

    total = sum(importances.values())
    print(f"\n  Importance sum: {total:.4f} (should be ≈ 1.0)")
    print("=" * 60)


def main():
    """Full training pipeline: load → train → evaluate → save."""
    print("=" * 60)
    print("WATER QUALITY ML TRAINING PIPELINE")
    print("=" * 60 + "\n")

    df = load_and_validate(DATA_FILE)
    results = train_model(df)
    save_results(results["model"], results["metrics"])
    print_results(results["metrics"])

    print("\n✓ Training pipeline complete!")


if __name__ == "__main__":
    main()
