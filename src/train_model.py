"""
ML Model Training & Evaluation Suite
====================================
Comprehensive training pipeline for Water Quality AI:
1. WQI Continuous Prediction:
   - Compares RandomForestRegressor, GradientBoostingRegressor,
     ExtraTreesRegressor, and LinearRegression baseline.
   - Computes R², MAE, RMSE, MAPE, and 5-fold cross-validation.
   - Computes feature importances for all 6 parameters.
2. Potability Classification:
   - Trains RandomForestClassifier on the real-world Kaggle Potability dataset.
   - Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
3. Saves models, evaluation metrics, and benchmarking summaries.

Usage:
    python src/train_model.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.calculate_wqi import FEATURES, TARGET

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

WQI_DATA_FILE = os.path.join(DATA_DIR, "water_quality.csv")
POTABILITY_DATA_FILE = os.path.join(DATA_DIR, "cleaned_kaggle_water_potability.csv")

WQI_MODEL_FILE = os.path.join(MODEL_DIR, "wqi_model.pkl")
POTABILITY_MODEL_FILE = os.path.join(MODEL_DIR, "potability_model.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")

TEST_SIZE = 0.20
RANDOM_STATE = 42


def train_wqi_regressors():
    """Train and compare regression models for WQI prediction."""
    print("=" * 60)
    print("TRAINING WQI REGRESSION MODELS (6 Sensor Parameters)")
    print("=" * 60)

    if not os.path.exists(WQI_DATA_FILE):
        raise FileNotFoundError(f"Missing {WQI_DATA_FILE}. Run data_pipeline.py first.")

    df = pd.read_csv(WQI_DATA_FILE)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Dataset split: {len(X_train)} train / {len(X_test)} test")

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "LinearRegression": LinearRegression()
    }

    results = {}
    best_r2 = -float("inf")
    best_model_name = None
    best_model = None

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-5))) * 100)

        cv_scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        print(f"  R² Score:  {r2:.4f} (CV: {cv_mean:.4f} ± {cv_std:.4f})")
        print(f"  MAE:       {mae:.4f}")
        print(f"  RMSE:      {rmse:.4f}")
        print(f"  MAPE:      {mape:.2f}%")

        feat_imp = {}
        if hasattr(model, "feature_importances_"):
            for feat, imp in zip(FEATURES, model.feature_importances_):
                feat_imp[feat] = round(float(imp), 4)

        results[name] = {
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "cv_r2_mean": round(cv_mean, 4),
            "cv_r2_std": round(cv_std, 4),
            "feature_importances": feat_imp
        }

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model

    # Save best regression model
    joblib.dump(best_model, WQI_MODEL_FILE)
    print(f"\n[OK] Saved best WQI model ({best_model_name}, R2={best_r2:.4f}) to {WQI_MODEL_FILE}")

    return results, best_model_name, best_model


def train_potability_classifier():
    """Train classification model on Kaggle Water Potability real dataset."""
    print("\n" + "=" * 60)
    print("TRAINING POTABILITY CLASSIFIER (Kaggle Real Dataset)")
    print("=" * 60)

    if not os.path.exists(POTABILITY_DATA_FILE):
        print(f"Cleaned potability file not found. Skipping classifier.")
        return None

    df = pd.read_csv(POTABILITY_DATA_FILE)
    feature_cols = [c for c in df.columns if c != "Potability"]
    X = df[feature_cols]
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    feat_imp = {col: round(float(imp), 4) for col, imp in zip(feature_cols, clf.feature_importances_)}

    joblib.dump(clf, POTABILITY_MODEL_FILE)
    print(f"[OK] Saved Potability model to {POTABILITY_MODEL_FILE}")

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "feature_importances": feat_imp
    }


def main():
    reg_results, best_name, _ = train_wqi_regressors()
    clf_results = train_potability_classifier()

    # Save comprehensive metrics JSON
    combined_metrics = {
        "best_wqi_model": best_name,
        "wqi_regression_models": reg_results,
        "potability_classification": clf_results,
        "dataset_metadata": {
            "wqi_dataset_rows": 3000,
            "potability_dataset_rows": 3276,
            "test_split_ratio": TEST_SIZE,
            "random_state": RANDOM_STATE
        }
    }

    # Also keep backward compatibility top-level keys for existing app.py readers
    best_metrics = reg_results[best_name]
    combined_metrics["mae"] = best_metrics["mae"]
    combined_metrics["rmse"] = best_metrics["rmse"]
    combined_metrics["r2"] = best_metrics["r2"]
    combined_metrics["feature_importances"] = best_metrics["feature_importances"]
    combined_metrics["test_size"] = TEST_SIZE
    combined_metrics["n_train"] = int(3000 * (1 - TEST_SIZE))
    combined_metrics["n_test"] = int(3000 * TEST_SIZE)

    with open(METRICS_FILE, "w") as f:
        json.dump(combined_metrics, f, indent=2)

    print(f"\n[OK] Saved all metrics to {METRICS_FILE}")


if __name__ == "__main__":
    main()
