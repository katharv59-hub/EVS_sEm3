# AI-Based Water Quality Prediction

## Overview

This project demonstrates an AI-based approach for predicting Water Quality Index (WQI) from six water-quality parameters:

- pH
- Turbidity (NTU)
- Dissolved Oxygen (mg/L)
- Temperature (°C)
- Conductivity (µS/cm)
- TDS (ppm)

The prototype uses synthetic environmental data and a Random Forest regression model.

## Architecture

```text
Synthetic Data Generation
        ↓
WQI Target Calculation
        ↓
Training Dataset (3,000 samples)
        ↓
Random Forest Regression (200 trees)
        ↓
Saved Model (.pkl)
        ↓
Streamlit Dashboard
        ↓
New Water Sample → Predicted WQI → Quality Category
```

## WQI Methodology

The project uses a **transparent, project-specific** weighted scoring methodology:

| Parameter | Weight | Scoring Logic |
|---|---|---|
| pH | 20% | Peak score at neutral pH (~7), quadratic penalty for deviation |
| Turbidity | 15% | Exponential decay — lower turbidity scores higher |
| Dissolved Oxygen | 20% | Saturating function — higher DO scores higher |
| Temperature | 10% | Peak near 25°C, quadratic penalty for deviation |
| Conductivity | 15% | Exponential decay — lower conductivity scores higher |
| TDS | 20% | Exponential decay — lower TDS scores higher |

**Final WQI = weighted sum of parameter scores, clipped to 0–100**

> **Note:** This is a project-specific formula for demonstration purposes. It should not be presented as a universally accepted regulatory WQI standard.

## Quality Classification

| WQI Range | Category |
|---|---|
| 90–100 | Excellent |
| 75–89 | Good |
| 50–74 | Moderate |
| 25–49 | Poor |
| 0–24 | Very Poor |

## Synthetic Data

Since hardware data collection is outside the scope of this prototype, the system generates realistic synthetic data using a **latent quality variable** approach:

- A hidden quality condition (0=poor, 1=good) drives all parameters
- TDS and conductivity are physically correlated (TDS ≈ 0.5–0.7 × Conductivity)
- Controlled noise ensures the ML problem is non-trivial
- ~3,000 samples generated with reproducible random seeds

## AI Model

**Random Forest Regressor** (scikit-learn)
- 200 decision trees
- 80/20 train/test split
- Fixed random state for reproducibility

### Model Evaluation

| Metric | Value |
|---|---|
| MAE | 0.9449 |
| RMSE | 1.3481 |
| R² | 0.9900 |

## Project Structure

```text
water-quality-ai/
├── data/
│   └── water_quality.csv       # Generated synthetic dataset
├── models/
│   ├── wqi_model.pkl           # Trained model
│   └── metrics.json            # Saved evaluation metrics
├── src/
│   ├── generate_data.py        # Data generator
│   ├── calculate_wqi.py        # WQI scoring functions
│   └── train_model.py          # ML training pipeline
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python src/generate_data.py
```

### 3. Train the model

```bash
python src/train_model.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

## Limitations

- Data is synthetic, not from real sensors
- The WQI scoring methodology is project-specific
- The model is not validated against laboratory measurements
- This system should not be used for regulatory or health decisions

## Future Work

- ESP32 sensor integration for real-time data collection
- Real environmental data from water bodies
- Laboratory-verified WQI labels for model validation
- Cloud database for persistent storage
- Live monitoring dashboard
- Model retraining with new data
