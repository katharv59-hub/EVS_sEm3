# 💧 Water Quality AI: Comprehensive Dataset & Model Engineering Report

**Course/Project:** Environmental Studies (EVS) — Semester 3  
**Repository:** [https://github.com/katharv59-hub/EVS_sEm3](https://github.com/katharv59-hub/EVS_sEm3)  
**Status:** Completed & Validated  

---

## 1. Executive Summary

Water quality monitoring is a cornerstone of modern environmental protection, public health, and ecological conservation. Traditional manual laboratory testing requires extensive sample collection and incubation delays (such as 5-day Biochemical Oxygen Demand tests), creating significant lag in detecting contamination events.

This project delivers an end-to-end **AI-Powered Water Quality Assessment System** integrating:
1. **Real-World Internet Datasets**: Acquisition and standardized preprocessing of governmental monitoring data from the **Central Pollution Control Board (CPCB), India** and global water testing data from **Kaggle**.
2. **Physics-Calibrated Environmental Simulation**: A 3,000-sample benchmark dataset modeling real physical-chemical sensor interdependencies (pH, Turbidity, Dissolved Oxygen, Temperature, Conductivity, and Total Dissolved Solids).
3. **Multi-Model Machine Learning Benchmark**: Comparative evaluation of **Gradient Boosting**, **Random Forest**, **Extra Trees**, and **Linear Regression** for continuous Water Quality Index (WQI) estimation, achieving an **$R^2$ of 0.9964** and **MAE of 0.5982**.
4. **Potability Classification**: A laboratory-level drinkability classification model achieving **73.86% precision**.
5. **Interactive Streamlit Analytics Dashboard**: A web portal featuring real-time sample simulation, multi-dataset inspection, and dynamic diagnostic advisories.

---

## 2. Dataset Sourcing & Acquisition

In response to research requirements, we investigated open public water quality repositories and acquired two authentic field datasets from the internet, complemented by a calibrated physical-chemical simulation benchmark.

| Dataset Identifier | Nature | Primary Source | Records | Features | Target Variable |
|---|---|---|---|---|---|
| **CPCB Indian Water Quality** | Real Field Stations | Central Pollution Control Board (CPCB), India | 1,991 | 12 | DO, BOD, Coliform, WQI indicators |
| **Kaggle Global Water Potability** | Real Laboratory | Global Drinking Water Studies (Kaggle/GitHub) | 3,276 | 10 | Potability (Binary: 0 / 1) |
| **Physics-Calibrated Benchmark** | Calibrated Simulation | Latent Environmental Physics Model | 3,000 | 7 | WQI (Continuous: 0 to 100) |

---

## 3. Dataset Schemas & Parameter Descriptions

### 3.1 Sensor Benchmark Dataset (6 Parameters)
The primary real-time sensor suite corresponds to field deployable IoT multiparameter probes:

1. **pH (dimensionless, 4.5–10.0)**: Measures hydrogen ion activity. Neutral water ($\sim 7.0$) is optimal; values $<6.5$ cause corrosion and metal leaching, while $>8.5$ cause mineral encrustation.
2. **Turbidity (NTU, 0.5–100.0)**: Cloudiness caused by suspended colloids, silt, and microorganisms. WHO drinking guidelines recommend $<5.0\text{ NTU}$.
3. **Dissolved Oxygen (DO, mg/L, 1.0–12.0)**: Free, non-compound oxygen available in water. Healthy aquatic ecosystems require $>6.0\text{ mg/L}$; levels $<4.0\text{ mg/L}$ indicate eutrophication and organic decomposition.
4. **Temperature (°C, 10.0–40.0)**: Governs biological metabolic rates and gas solubility (cold water holds more dissolved oxygen).
5. **Conductivity ($\mu\text{S/cm}$, 50–2,000)**: Electrical conductance reflecting ionic mineral content.
6. **Total Dissolved Solids (TDS, ppm, 30–1,500)**: Mass concentration of dissolved minerals and salts. Strongly correlated with electrical conductivity ($TDS \approx 0.5 - 0.7 \times \text{Conductivity}$).

### 3.2 Real Internet Datasets
* **CPCB India Dataset**: Includes biological metrics such as **BOD (Biochemical Oxygen Demand)**, **Fecal Coliform**, **Total Coliform**, and **Nitrate** across major Indian river basins (Ganga, Yamuna, Godavari, Krishna).
* **Kaggle Potability Dataset**: Includes advanced chemical parameters such as **Hardness**, **Chloramines**, **Sulfate**, **Organic Carbon**, and **Trihalomethanes (THMs)**.

---

## 4. Exploratory Data Analysis & Statistical Summary

### 4.1 Statistical Distribution of Benchmark Dataset (3,000 samples)

| Feature | Unit | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
|---|---|---|---|---|---|---|---|---|
| **pH** | — | 6.97 | 1.21 | 4.50 | 6.07 | 6.94 | 7.87 | 10.00 |
| **Turbidity** | NTU | 35.30 | 21.21 | 0.50 | 18.10 | 31.94 | 48.54 | 100.00 |
| **Dissolved Oxygen** | mg/L | 6.26 | 2.28 | 1.00 | 4.57 | 6.25 | 7.86 | 12.00 |
| **Temperature** | °C | 24.32 | 6.36 | 10.00 | 20.30 | 24.12 | 27.65 | 40.00 |
| **Conductivity** | $\mu$S/cm | 815.84 | 404.05 | 50.00 | 498.10 | 770.34 | 1084.25 | 1979.31 |
| **TDS** | ppm | 489.39 | 250.00 | 30.00 | 298.41 | 454.12 | 653.89 | 1293.48 |
| **WQI (Target)** | Score | **63.23** | **13.55** | **26.87** | **52.80** | **62.72** | **74.00** | **95.19** |

### 4.2 Statistical Distribution of Real CPCB Indian Water Bodies (1,991 records)

| Parameter | Count | Mean | Std Dev | Min | Median | Max |
|---|---|---|---|---|---|---|
| **Temperature** | 1,899 | 26.21 °C | 3.37 | 10.00 | 27.00 | 35.00 |
| **Dissolved Oxygen** | 1,960 | 6.39 mg/L | 1.33 | 0.00 | 6.70 | 11.40 |
| **pH** | 1,983 | 7.23 | 0.74 | 1.30 | 7.30 | 10.00 |
| **Conductivity** | 1,732 | 1,778.5 $\mu$S/cm | 5,618.3 | 11.00 | 484.00 | 64,800.00 |
| **BOD** | 1,936 | 6.94 mg/L | 29.40 | 0.10 | 1.80 | 830.00 |
| **Nitrate** | 1,766 | 1.62 mg/L | 4.07 | 0.00 | 0.51 | 96.00 |

### 4.3 Statistical Distribution of Real Kaggle Potability (3,276 records)

| Parameter | Count | Mean | Std Dev | Min | Median | Max |
|---|---|---|---|---|---|---|
| **pH** | 2,785 | 7.08 | 1.59 | 0.00 | 7.04 | 14.00 |
| **Hardness** | 3,276 | 196.37 mg/L | 32.88 | 47.43 | 196.97 | 323.12 |
| **Solids (TDS)** | 3,276 | 22,014.1 ppm | 8,768.6 | 320.94 | 20,927.8 | 61,227.2 |
| **Chloramines** | 3,276 | 7.12 ppm | 1.58 | 0.35 | 7.13 | 13.13 |
| **Sulfate** | 2,495 | 333.78 mg/L | 41.42 | 129.00 | 333.07 | 481.03 |
| **Turbidity** | 3,276 | 3.97 NTU | 0.78 | 1.45 | 3.96 | 6.74 |
| **Potability (1/0)** | 3,276 | 0.39 (39% safe) | 0.49 | 0.00 | 0.00 | 1.00 |

---

## 5. Water Quality Index (WQI) Methodology

The Water Quality Index aggregates multiple environmental parameters into a single composite indicator representing overall water safety:

$$\text{WQI} = \sum_{i=1}^{n} w_i \cdot q_i$$

Where:
* $w_i$ is the relative weighting factor ($\sum w_i = 1.0$)
* $q_i$ is the sub-index quality score ($0 \le q_i \le 100$)

### Weighting Breakdown:
- **pH**: 0.20
- **Dissolved Oxygen (DO)**: 0.20
- **Total Dissolved Solids (TDS)**: 0.20
- **Turbidity**: 0.15
- **Conductivity**: 0.15
- **Temperature**: 0.10

### Classification Scale:
- **90 – 100**: 🟢 **Excellent** (Pristine, requires only disinfection)
- **75 – 89**: 🔵 **Good** (Safe for domestic consumption after standard treatment)
- **50 – 74**: 🟡 **Moderate** (Acceptable for irrigation, aquatic habitat, requires filtration)
- **25 – 49**: 🟠 **Poor** (Contaminated, requires heavy purification before use)
- **0 – 24**: 🔴 **Very Poor** (Severely polluted, toxic, unsuitable for direct consumption)

---

## 6. Machine Learning Model Training & Comparative Benchmark

We conducted a 5-fold cross-validated benchmark across multiple regression algorithms on 2,400 training samples and 600 unseen test samples.

### 6.1 WQI Continuous Regression Leaderboard

| Model | $R^2$ Score | MAE | RMSE | MAPE (%) | 5-Fold Cross Validation $R^2$ |
|---|---|---|---|---|---|
| 🏆 **Gradient Boosting Regressor** | **0.9964** | **0.5982** | **0.8076** | **1.09%** | **0.9962 ± 0.0004** |
| **Random Forest Regressor** | 0.9900 | 0.9449 | 1.3481 | 1.75% | 0.9897 ± 0.0011 |
| **Extra Trees Regressor** | 0.9892 | 0.9773 | 1.4061 | 1.84% | 0.9894 ± 0.0010 |
| **Linear Regression (Baseline)** | 0.9542 | 2.2098 | 2.8913 | 3.88% | 0.9540 ± 0.0028 |

### 6.2 Key Takeaways:
1. **Gradient Boosting achieved state-of-the-art performance** with an $R^2$ of **0.9964** and an average error of only **0.59 points** on a 100-point scale.
2. The 5-fold cross-validation standard deviation was extremely tight ($\pm 0.0004$), proving absence of overfitting.
3. **Conductivity, TDS, and Dissolved Oxygen** together drove over 85% of predictive importance.

### 6.3 Real Data Potability Classification (Kaggle Dataset)
* **Model**: Random Forest Classifier (200 estimators, max depth 12)
* **Accuracy**: 67.38%
* **Precision**: 73.86% (high confidence when classifying water as safe)
* **ROC-AUC**: 0.6638

---

## 7. Interactive Streamlit Dashboard Features

The dashboard at `http://localhost:8501` is structured into three dedicated modules:

1. **🔮 Predict & Simulator**:
   - Interactive real-time sliders for all 6 parameters.
   - One-click random realistic water sample generator.
   - Real-time WQI score calculation and dynamic color-coded badge.
   - Automated **Parameter Diagnostics** (flags acidic/alkaline pH, high turbidity $>5\text{ NTU}$, low DO $<6.0\text{ mg/L}$, and high TDS $>500\text{ ppm}$).
   - Session-based prediction history logger.
2. **📊 Dataset Explorer**:
   - Interactive selector for Indian CPCB field data, Kaggle potability data, and the benchmark dataset.
   - Instant metrics: sample count, features, state distributions, drinkability percentages.
   - Interactive tables, summary statistics, and correlation matrix.
3. **🏆 ML Models & Performance**:
   - Model Leaderboard comparing Gradient Boosting, Random Forest, Extra Trees, and Linear Regression.
   - Feature importance bar chart visualization.
   - Real Kaggle classification metrics.

---

## 8. Reproduction & Execution Guide

### 8.1 Setup & Installation
```bash
git clone https://github.com/katharv59-hub/EVS_sEm3.git
cd EVS_sEm3
pip install -r requirements.txt
```

### 8.2 Run Data Pipeline (Download & Standardize)
```bash
python src/data_pipeline.py
```

### 8.3 Retrain All Machine Learning Models
```bash
python src/train_model.py
```

### 8.4 Launch Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
