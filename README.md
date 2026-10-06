# AI-Based Water Quality Prediction Platform

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/katharv59-hub/EVS_sEm3)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Model R²](https://img.shields.io/badge/Model%20R²-0.9964-success)](#model-performance-leaderboard)

An intelligent water quality assessment and machine learning platform that analyzes real-time sensor metrics, benchmarks against authentic environmental datasets (Indian CPCB and Kaggle Potability), and estimates the continuous Water Quality Index (WQI).

---

## 📌 Key Highlights

- **Real Internet Datasets**: Sourced and cleaned **Central Pollution Control Board (CPCB) India** river monitoring data (1,991 records) and **Kaggle Global Water Potability** dataset (3,276 records).
- **Physics-Calibrated Benchmark**: 3,000 samples generated with latent environmental conditioning and inter-parameter dependencies (Conductivity $\leftrightarrow$ TDS, DO $\leftrightarrow$ Temp).
- **Multi-Model ML Benchmarking**: Evaluated **Gradient Boosting**, **Random Forest**, **Extra Trees**, and **Linear Regression** with 5-fold cross-validation.
- **Top Regression Performance**: **Gradient Boosting Regressor** achieves **$R^2 = 0.9964$**, **$\text{MAE} = 0.5982$**, and **$\text{MAPE} = 1.09\%$**.
- **Potability Classifier**: Laboratory drinkability prediction with **73.86% precision** and **0.6638 ROC-AUC**.
- **Interactive Web App**: Complete Streamlit application with real-time sliders, random sample simulation, parameter diagnostics, dataset explorer, and model analytics.

---

## 🏗️ Architecture & Pipeline

```text
       ┌───────────────────────────────┐      ┌───────────────────────────────┐
       │   Real Internet Datasets      │      │  Physics-Calibrated Simulator │
       │ (CPCB India & Kaggle 3.2k)    │      │    (Latent Factor Modeling)   │
       └──────────────┬────────────────┘      └───────────────┬───────────────┘
                      │                                       │
                      └───────────────────┬───────────────────┘
                                          ▼
                             src/data_pipeline.py
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Standardized Datasets & Summaries   │
                      └───────────────────┬───────────────────┘
                                          ▼
                              src/train_model.py
                        (GB, RF, ExtraTrees, LinearReg)
                                          ▼
                          models/wqi_model.pkl & metrics.json
                                          ▼
                                     app.py
                     (Interactive Streamlit Web Dashboard)
```

---

## 📊 Datasets Handled

| Dataset | Type | Source | Size | Key Parameters |
|---|---|---|---|---|
| **CPCB River Monitoring** | Real Field | Central Pollution Control Board, India | 1,991 rows | Temp, DO, pH, Conductivity, BOD, Nitrate, Coliform |
| **Kaggle Water Potability** | Real Lab | Global Water Testing Studies | 3,276 rows | pH, Hardness, Solids, Chloramines, Sulfate, Turbidity |
| **Sensor Benchmark** | Physical Sim | Correlated Latent Simulation | 3,000 rows | pH, Turbidity, DO, Temp, Conductivity, TDS, WQI |

---

## 🏆 Model Performance Leaderboard

Evaluated on 600 unseen test samples using 5-fold cross-validation:

| Model | $R^2$ Score | MAE | RMSE | MAPE | 5-Fold CV $R^2$ |
|---|---|---|---|---|---|
| 🥇 **Gradient Boosting** | **0.9964** | **0.5982** | **0.8076** | **1.09%** | **0.9962 ± 0.0004** |
| 🥈 **Random Forest** | **0.9900** | **0.9449** | **1.3481** | **1.75%** | **0.9897 ± 0.0011** |
| 🥉 **Extra Trees** | **0.9892** | **0.9773** | **1.4061** | **1.84%** | **0.9894 ± 0.0010** |
| 🔹 **Linear Regression (Baseline)** | **0.9542** | **2.2098** | **2.8913** | **3.88%** | **0.9540 ± 0.0028** |

### Top Predictive Feature Importances:
1. **Conductivity**: 68.6%
2. **Total Dissolved Solids (TDS)**: 11.6%
3. **Dissolved Oxygen (DO)**: 10.9%
4. **Turbidity**: 6.3%
5. **Temperature**: 2.1%
6. **pH**: 0.4%

---

## 📐 Water Quality Index (WQI) Scoring Standard

$$\text{WQI} = \sum_{i=1}^{n} w_i \cdot q_i$$

| Parameter | Unit | Weight ($w_i$) | Ideal Range | Scoring Function Behavior |
|---|---|---|---|---|
| **pH** | — | 0.20 | 6.5 – 8.5 | Quadratic penalty centered at 7.0 |
| **Dissolved Oxygen** | mg/L | 0.20 | $> 6.0$ | Saturating exponential (saturates near 8–10 mg/L) |
| **TDS** | ppm | 0.20 | $< 300$ | Exponential decay |
| **Turbidity** | NTU | 0.15 | $< 5.0$ | Exponential decay |
| **Conductivity** | $\mu$S/cm | 0.15 | $< 400$ | Exponential decay |
| **Temperature** | °C | 0.10 | 20 – 25 | Quadratic penalty centered at 25.0 °C |

### Water Quality Categories
- **90 – 100**: 🟢 **Excellent** (Pristine, drinking quality)
- **75 – 89**: 🔵 **Good** (Safe with basic filtration/disinfection)
- **50 – 74**: 🟡 **Moderate** (Acceptable for irrigation & aquatic life)
- **25 – 49**: 🟠 **Poor** (Significant degradation, requires intensive treatment)
- **0 – 24**: 🔴 **Very Poor** (Severely polluted, toxic)

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/katharv59-hub/EVS_sEm3.git
cd EVS_sEm3
pip install -r requirements.txt
```

### 2. Download Real Datasets & Process
```bash
python src/data_pipeline.py
```

### 3. Train Models
```bash
python src/train_model.py
```

### 4. Launch Dashboard
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
water-quality-ai/
├── data/
│   ├── water_quality.csv                  # Calibrated sensor benchmark dataset
│   ├── cleaned_indian_cpcb_water_quality.csv  # Cleaned CPCB India field dataset
│   ├── cleaned_kaggle_water_potability.csv    # Cleaned Kaggle potability dataset
│   └── dataset_summary.json               # Full statistical summary metadata
├── models/
│   ├── wqi_model.pkl                      # Best trained model (GradientBoosting)
│   ├── potability_model.pkl               # Trained Potability Classifier
│   └── metrics.json                       # Comprehensive evaluation metrics
├── src/
│   ├── data_pipeline.py                   # Real dataset downloader & preprocessor
│   ├── calculate_wqi.py                   # WQI scoring mathematics & scales
│   ├── generate_data.py                   # Latent physics synthetic generator
│   └── train_model.py                     # Multi-model ML training suite
├── app.py                                 # Streamlit 3-tab web dashboard
├── requirements.txt                       # Dependencies
├── DATASET_AND_MODEL_REPORT.md            # In-depth academic & engineering report
└── README.md                              # Project documentation
```

For the comprehensive scientific report with full statistical tables, parameter formulas, and methodology breakdown, see [DATASET_AND_MODEL_REPORT.md](DATASET_AND_MODEL_REPORT.md).
