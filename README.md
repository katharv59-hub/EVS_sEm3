# 💧 AI-Based Water Quality Assessment Platform

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/katharv59-hub/EVS_sEm3)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Model R²](https://img.shields.io/badge/WQI%20Model%20R²-0.9964-success)](#model-performance-results)

An Environmental Studies (EVS) Semester 3 project that integrates **IoT-style multi-parameter sensor screening**, **domain-aware environmental data cleaning**, and **machine learning benchmarks** to estimate the continuous Water Quality Index (WQI) and evaluate laboratory drinking water potability.

---

## 📌 Project Overview & Problem Statement

Conventional water testing relies on manual grab-sampling and laboratory incubation (e.g., standard 5-day BOD assays), causing critical delays in identifying pollution spikes. 

This platform evaluates how multi-parameter sensor inputs (pH, Turbidity, Dissolved Oxygen, Temperature, Conductivity, and TDS) can provide rapid, continuous environmental screening. It addresses:
1. **Data Quality Integrity:** Identifies and corrects historical column-inversion anomalies in real CPCB monitoring data.
2. **Leakage-Free Modeling:** Enforces strict pre-split preprocessing in classification pipelines.
3. **Scientific Defensibility:** Distinguishes between simulated sensor benchmark approximations and real-world laboratory compliance.

---

## 🏗️ System Architecture

```text
  ┌─────────────────────────────────────────────────────────────┐
  │                      Data Sourcing                          │
  │  - CPCB India Field Data (1,991 records via GitHub mirror)   │
  │  - Kaggle Potability Lab Data (3,276 records via mirror)    │
  │  - Physics-Calibrated IoT Benchmark (3,000 simulated)       │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 src/data_pipeline.py                        │
  │  - Domain cleaning: repairs swapped CPCB columns            │
  │  - Pre-split preservation: zero data leakage in potability  │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                  src/train_model.py                         │
  │  - Continuous WQI Regression (Gradient Boosting, RF, etc.)  │
  │  - Potability Classification (Balanced Pipeline RF)         │
  └──────────────────────────────┬──────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                       app.py                                │
  │  - Tab 1: Predict & Simulator (WQI + WHO/BIS Diagnostics)   │
  │  - Tab 2: Dataset Explorer (CPCB, Kaggle, Benchmark)        │
  │  - Tab 3: ML Models & Evaluation (Leaderboard + CM)         │
  └─────────────────────────────────────────────────────────────┘
```

---

## 📊 Datasets

| Dataset | Type | Source / Provenance | Size | Key Parameters |
|---|---|---|---|---|
| **CPCB River Data** | Real Field | Central Pollution Control Board, India (via public mirror) | 1,991 rows | Temp, DO, pH, Conductivity, BOD, Nitrate, Coliform |
| **Kaggle Potability** | Real Lab | Global Water Testing Studies (via public mirror) | 3,276 rows | pH, Hardness, Solids, Chloramines, Sulfate, Turbidity |
| **Sensor Benchmark** | Simulated | Physics-calibrated latent condition simulation | 3,000 rows | pH, Turbidity, DO, Temp, Conductivity, TDS, WQI |

*See [DATA_SOURCES.md](DATA_SOURCES.md) for full licensing, retrieval URLs, and cleaning notes.*

---

## 📐 WQI Methodology & Classification

$$\text{WQI} = \sum_{i=1}^{n} w_i \cdot q_i$$

- **Weights ($w_i$):** pH ($0.20$), DO ($0.20$), TDS ($0.20$), Turbidity ($0.15$), Conductivity ($0.15$), Temperature ($0.10$).
- **Classification Bands:**
  - 🟢 **90.0 – 100.0:** Excellent (Pristine freshwater)
  - 🔵 **75.0 – 89.9:** Good (Safe with conventional filtration)
  - 🟡 **50.0 – 74.9:** Moderate (Suitable for irrigation & aquatic life)
  - 🟠 **25.0 – 49.9:** Poor (Degraded; requires intensive treatment)
  - 🔴 **0.0 – 24.9:** Very Poor (Severely polluted; anoxic)

---

## 🏆 Model Performance Results

### 1. WQI Continuous Regression Leaderboard (600 Unseen Test Samples)

| Model Algorithm | $R^2$ Score | MAE (WQI pts) | RMSE | MAPE (%) | 5-Fold CV $R^2$ |
|---|---|---|---|---|---|
| 🥇 **Gradient Boosting** | **0.9964** | **0.5982** | **0.8076** | **1.09%** | **0.9962 ± 0.0004** |
| 🥈 **Random Forest** | **0.9900** | **0.9449** | **1.3481** | **1.75%** | **0.9897 ± 0.0011** |
| 🥉 **Extra Trees** | **0.9892** | **0.9773** | **1.4061** | **1.84%** | **0.9894 ± 0.0010** |
| 🔹 **Linear Regression (Baseline)** | **0.9542** | **2.2098** | **2.8913** | **3.88%** | **0.9540 ± 0.0028** |

> **Interpretation Note:** The high $R^2$ demonstrates that the regression algorithm accurately learns the non-linear relationship defined by the project's benchmark scoring formula. It should not be interpreted as proving universal predictive accuracy across uncalibrated natural lakes or rivers.

### 2. Potability Classification Pipeline (656 Unseen Test Samples, Zero Leakage)
- **Model:** `Pipeline(SimpleImputer + Balanced RandomForestClassifier)`
- **Accuracy:** 65.85% | **Precision:** 57.84% | **Recall:** 46.09% | **F1-Score:** 0.5130 | **ROC-AUC:** 0.6761
- **Confusion Matrix:** $\text{TN} = 314,\ \text{FP} = 86,\ \text{FN} = 138,\ \text{TP} = 118$

---

## 🚀 Setup & Execution

### 1. Clone & Install
```bash
git clone https://github.com/katharv59-hub/EVS_sEm3.git
cd EVS_sEm3
pip install -r requirements.txt
```

### 2. Run Data Pipeline (Download & Clean)
```bash
python src/data_pipeline.py
```

### 3. Train Models
```bash
python src/train_model.py
```

### 4. Run Automated Integrity Tests
```bash
python tests/test_validation.py
```

### 5. Launch Web Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ⚠️ Limitations

- **Simulated WQI Target:** The continuous regression model is trained on a physics-calibrated synthetic benchmark.
- **Potability Detection Limits:** With a test recall of $\approx 46.1\%$, bulk chemical features alone cannot verify drinking water safety without microbial and heavy metal laboratory assays.
- **Educational Scope:** This tool is designed for academic demonstration and rapid screening; it does not constitute statutory regulatory compliance.

---

## 🔮 Future Scope

- Interfacing physical ESP32 microcontrollers with waterproof analog sensors.
- Incorporating enzymatic coliform assays to improve potability classification.
- Dynamic regional calibration for local river basins.

*For full scientific methodology, statistical distributions, and academic references, consult [DATASET_AND_MODEL_REPORT.md](DATASET_AND_MODEL_REPORT.md).*
