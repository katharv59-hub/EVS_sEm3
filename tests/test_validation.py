"""
Automated System & Scientific Integrity Validation Suite
=========================================================
Runs automated sanity checks on:
1. Mathematical integrity of WQI scoring functions and weights
2. Gapless quality category mapping
3. Environmental data cleaning boundaries in CPCB dataset
4. Model artifacts, pipeline loading, and inference safety
5. Metric files and documentation contracts

Usage:
    python tests/test_validation.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.calculate_wqi import (
    FEATURES,
    TARGET,
    WEIGHTS,
    calculate_wqi,
    get_quality_category,
    evaluate_threshold_diagnostics,
)


def test_wqi_weights_sum_to_one():
    total_weight = sum(WEIGHTS.values())
    assert abs(total_weight - 1.0) < 1e-6, f"WQI weights do not sum to 1.0: {total_weight}"
    print("  [PASS] WQI weights sum exactly to 1.0")


def test_wqi_bounds_and_extremes():
    # Test boundary extremes
    min_wqi = calculate_wqi(ph=0.0, turbidity=500.0, do=0.0, temp=50.0, conductivity=5000.0, tds=4000.0)
    max_wqi = calculate_wqi(ph=7.0, turbidity=0.0, do=10.0, temp=25.0, conductivity=50.0, tds=30.0)

    assert 0.0 <= min_wqi <= 100.0, f"Min WQI out of range: {min_wqi}"
    assert 0.0 <= max_wqi <= 100.0, f"Max WQI out of range: {max_wqi}"
    assert not np.isnan(min_wqi) and not np.isnan(max_wqi), "WQI returned NaN for edge inputs"
    print(f"  [PASS] WQI output is strictly bounded in [0, 100] (Min: {min_wqi:.2f}, Max: {max_wqi:.2f})")


def test_gapless_category_mapping():
    test_scores = [0.0, 24.9, 25.0, 49.8, 50.0, 74.5, 75.0, 89.2, 90.0, 100.0]
    expected_categories = [
        "Very Poor", "Very Poor", "Poor", "Poor", "Moderate",
        "Moderate", "Good", "Good", "Excellent", "Excellent"
    ]
    for score, expected in zip(test_scores, expected_categories):
        cat = get_quality_category(score)
        assert cat == expected, f"Score {score} mapped to {cat}, expected {expected}"
        assert cat != "Unknown", f"Category gap detected for score {score}"
    print("  [PASS] Quality category mapping is continuous and gapless")


def test_cpcb_cleaning_integrity():
    cpcb_path = os.path.join(BASE_DIR, "data", "cleaned_indian_cpcb_water_quality.csv")
    assert os.path.exists(cpcb_path), f"Cleaned CPCB file missing: {cpcb_path}"
    df = pd.read_csv(cpcb_path)

    # Check pH bounds
    valid_ph = df["pH"].dropna()
    assert (valid_ph >= 2.0).all(), f"Found pH < 2.0 in cleaned CPCB: min={valid_ph.min()}"
    assert (valid_ph <= 12.0).all(), f"Found pH > 12.0 in cleaned CPCB: max={valid_ph.max()}"
    assert valid_ph.max() < 14.0, f"Impossible pH above 14 found: {valid_ph.max()}"
    assert not (valid_ph == 67115.0).any(), "Corrupt pH 67,115 is still present!"
    print(f"  [PASS] CPCB pH verified in realistic physical range ({valid_ph.min():.2f} to {valid_ph.max():.2f})")


def test_model_loading_and_inference():
    wqi_model_path = os.path.join(BASE_DIR, "models", "wqi_model.pkl")
    pot_model_path = os.path.join(BASE_DIR, "models", "potability_model.pkl")

    assert os.path.exists(wqi_model_path), f"WQI model missing at {wqi_model_path}"
    assert os.path.exists(pot_model_path), f"Potability model missing at {pot_model_path}"

    wqi_model = joblib.load(wqi_model_path)
    pot_model = joblib.load(pot_model_path)

    # Test WQI inference
    sample_wqi_input = pd.DataFrame([{
        "pH": 7.2, "Turbidity": 4.5, "Dissolved Oxygen": 8.1,
        "Temperature": 24.5, "Conductivity": 280.0, "TDS": 170.0
    }], columns=FEATURES)
    wqi_pred = float(wqi_model.predict(sample_wqi_input)[0])
    assert 0.0 <= wqi_pred <= 100.0, f"WQI model returned out-of-range prediction: {wqi_pred}"

    # Test Potability inference with missing values to confirm pipeline imputer works
    sample_pot_input = pd.DataFrame([{
        "ph": np.nan,  # Simulating missing value
        "Hardness": 204.8, "Solids": 20791.0, "Chloramines": 7.3,
        "Sulfate": np.nan, "Conductivity": 564.0, "Organic_carbon": 10.3,
        "Trihalomethanes": 86.9, "Turbidity": 2.9
    }])
    pot_pred = int(pot_model.predict(sample_pot_input)[0])
    pot_prob = float(pot_model.predict_proba(sample_pot_input)[0, 1])
    assert pot_pred in [0, 1], f"Invalid potability prediction: {pot_pred}"
    assert 0.0 <= pot_prob <= 1.0, f"Invalid potability probability: {pot_prob}"

    print(f"  [PASS] WQI model loads and predicts safely (sample prediction: {wqi_pred:.2f})")
    print(f"  [PASS] Potability pipeline handles raw NaNs and predicts safely (prob: {pot_prob:.2f})")


def test_metrics_file_integrity():
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    assert os.path.exists(metrics_path), f"Metrics file missing: {metrics_path}"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "best_wqi_model" in metrics, "Missing 'best_wqi_model' in metrics"
    assert "wqi_regression_models" in metrics, "Missing regression leaderboard in metrics"
    assert "potability_classification" in metrics, "Missing potability classification in metrics"
    assert "confusion_matrix" in metrics["potability_classification"], "Missing confusion matrix in potability metrics"
    print(f"  [PASS] Metrics file contains complete evaluation records (Best model: {metrics['best_wqi_model']})")


def test_diagnostics_function():
    sample_inputs = {
        "pH": 7.1, "Turbidity": 3.0, "Dissolved Oxygen": 7.5,
        "Temperature": 24.0, "Conductivity": 300.0, "TDS": 200.0
    }
    diag = evaluate_threshold_diagnostics(sample_inputs)
    assert len(diag) == 6, f"Expected 6 parameter diagnostics, got {len(diag)}"
    for item in diag:
        assert "parameter" in item and "value" in item and "status" in item and "standard" in item
    print("  [PASS] Parameter diagnostics function returns complete structured records")


def run_all_validation_tests():
    print("=" * 60)
    print("RUNNING SYSTEM & SCIENTIFIC INTEGRITY VALIDATION SUITE")
    print("=" * 60)
    test_wqi_weights_sum_to_one()
    test_wqi_bounds_and_extremes()
    test_gapless_category_mapping()
    test_cpcb_cleaning_integrity()
    test_model_loading_and_inference()
    test_metrics_file_integrity()
    test_diagnostics_function()
    print("=" * 60)
    print("[SUCCESS] ALL 7 INTEGRITY VALIDATION TESTS PASSED CLEANLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_validation_tests()
