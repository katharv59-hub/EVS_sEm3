# 💧 Water Quality AI: Comprehensive Dataset, Model & Environmental Engineering Report

**Project Title:** AI-Based Water Quality Prediction Platform  
**Academic Context:** Environmental Studies (EVS) — Semester 3 Project  
**Repository:** [https://github.com/katharv59-hub/EVS_sEm3](https://github.com/katharv59-hub/EVS_sEm3)  
**Status:** Validated, Leakage-Free & Reproducible  

---

## 1. Introduction

Water is an indispensable natural resource sustaining terrestrial ecosystems, human communities, and industrial infrastructure. As urbanisation, industrial expansion, and agricultural intensification accelerate, freshwater ecosystems face unprecedented contamination pressure. Traditional environmental surveillance relies primarily on manual grab-sampling followed by laboratory chemical and biological analysis. While highly accurate, conventional laboratory procedures entail significant incubation delays (e.g., standard 5-day Biochemical Oxygen Demand tests), laboratory backlog, and geographical logistical constraints.

Recent advances in Internet of Things (IoT) multiparameter electronic probes and Machine Learning (ML) enable near-instantaneous estimation of composite environmental health indicators. This report documents the design, data provenance, domain-aware cleaning, machine learning benchmarks, and limitations of the **AI-Based Water Quality Prediction Platform**.

---

## 2. Problem Statement

Contemporary environmental monitoring faces three core operational challenges:
1. **Latency in Pollution Detection:** Manual laboratory testing creates critical lag between a hazardous discharge event and environmental management action.
2. **Sensor Dimensionality vs. Index Interpretation:** Real-time IoT probes generate discrete physical measurements (pH, turbidity, electrical conductivity, dissolved oxygen, temperature, total dissolved solids), but non-specialist decision-makers require an aggregated, standardized index—such as the **Water Quality Index (WQI)**—to determine ecological condition.
3. **Data Quality & Methodological Oversimplifications in AI Systems:** Many academic machine learning prototypes suffer from unacknowledged data quality corruption (e.g., column inversion anomalies in legacy environmental records), data leakage during preprocessing (e.g., imputing missing values prior to train/test partitioning), and scientifically invalid claims equating high synthetic model $R^2$ scores with universal real-world predictive authority.

---

## 3. Objectives

The primary objectives of this Semester 3 EVS project are:
1. **Acquire & Authenticate Environmental Datasets:** Retrieve open public monitoring data (Indian CPCB river surveillance and global drinking water potability laboratory data) via transparent, documented repository mirrors.
2. **Implement Domain-Aware Data Cleaning:** Algorithmically detect and resolve historical data corruption (such as inverted pH and conductivity columns) and enforce valid thermodynamic and environmental boundaries.
3. **Simulate a Physics-Calibrated IoT Sensor Benchmark:** Construct a 3,000-sample correlated dataset reflecting realistic multiparameter probe behavior with natural physicochemical couplings (e.g., Conductivity $\leftrightarrow$ TDS ratio of $0.5 - 0.7$).
4. **Develop a Multi-Model ML Benchmark:** Train and evaluate multiple regression algorithms (Gradient Boosting, Random Forest, Extra Trees, Linear Regression) using 5-fold cross-validation to approximate composite WQI scoring.
5. **Construct a Leakage-Free Classification Pipeline:** Train a balanced Random Forest pipeline on laboratory potability data with strict pre-split partitioning and transparent evaluation (including confusion matrix and recall analysis).
6. **Deploy an Interactive Streamlit Platform:** Build a 3-tab web dashboard featuring real-time sample simulation, parameter threshold diagnostics (aligned with WHO and BIS guidelines), dataset exploration, and transparent educational disclaimers.

---

## 4. Dataset Sources

To balance real-world empirical observation with high-frequency IoT sensor simulation, this project investigates three distinct datasets:

| Dataset Identifier | Nature | Original Authority / Source | Retrieval Mirror | Records | Target Variable |
|---|---|---|---|---|---|
| **CPCB India** | Real Field Stations | Central Pollution Control Board (CPCB), Ministry of Environment, Forest & Climate Change, India | [GitHub Mirror](https://raw.githubusercontent.com/aditikhatri/-Indian-water-quality-analysis-and-prediction/master/water_dataX.csv) | 1,991 | Historical Field Baseline (DO, BOD, Coliform) |
| **Kaggle Potability** | Real Laboratory | Drinking Water Quality Study (Aditya Kadiwal) | [GitHub Mirror](https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/main/water_potability.csv) | 3,276 | `Potability` (Binary: 0=Unsafe, 1=Safe) |
| **Sensor Benchmark** | Physics Simulation | Calibrated environmental simulation with latent factor coupling | Generated locally via `src/generate_data.py` | 3,000 | `WQI` (Continuous: 0 to 100) |

*Full licensing, mirror links, and column schema documentation are maintained in [DATA_SOURCES.md](DATA_SOURCES.md).*

---

## 5. Dataset Preprocessing & Data Quality Repairs

### 5.1 CPCB Data Quality Repair (Resolution of Impossible pH Statistics)
During repository audit, the raw CPCB mirror dataset exhibited physically impossible summary statistics: a pH mean of $\approx 112$ and a maximum of $67,115$. 

Investigation revealed that in rows 1901 to 1990 (89 monitoring records from 2003–2004), the `PH` and `CONDUCTIVITY` columns were inverted in the source mirror file. For example, Station 1435 in the industrial hub of Vapi, Gujarat had recorded a conductivity of $67,115\ \mu\text{S/cm}$ under the `PH` column and its true pH value ($5.0$) under the `CONDUCTIVITY` column.

**Domain-Aware Remediation:**
1. Identified records matching $pH > 14.0$ and $\text{Conductivity} \in [0.0, 14.0]$.
2. Swapped values back to their legitimate physical columns ($89$ records repaired).
3. Applied physical surface water boundaries: flagged $3$ non-physical entry artifacts ($pH < 2.0$) as `NaN`.
4. Enforced thermodynamic ranges on Temperature ($[0, 50]^\circ\text{C}$), Dissolved Oxygen ($[0, 20]\ \text{mg/L}$), and Conductivity ($\ge 0\ \mu\text{S/cm}$).

**Before vs. After Cleaning:**
- **Raw pH:** Count: 1,983 | Mean: 112.09 | Min: 0.00 | Max: 67,115.00
- **Cleaned pH:** Count: 1,980 | Mean: 7.21 | Min: 2.60 | Median: 7.20 | Max: 9.01

### 5.2 Kaggle Potability Preprocessing (Elimination of Data Leakage)
Common community implementations impute missing values (such as Sulfate and pH) over the entire dataset prior to train/test partitioning. This leaks global distribution statistics into the test set.

**Leakage-Free Remediation:**
The dataset was partitioned into an 80% training set (2,620 samples) and a 20% stratified test set (656 samples) *first*. A `SimpleImputer(strategy='median')` was embedded into an `sklearn.pipeline.Pipeline`, ensuring the median was calculated solely from training observations and applied without leakage to test evaluation.

---

## 6. Environmental Parameters & Reference Standards

The primary multiparameter probe configuration monitors six parameters:

1. **pH (dimensionless, 4.5 – 10.0):** Measures hydrogen ion concentration. Neutral water ($\sim 7.0$) supports balanced aquatic life. The acceptable drinking range per **BIS IS 10500:2012 / WHO** is **6.5 to 8.5**. Values $<6.5$ cause corrosion and solubilize heavy metals; values $>8.5$ cause bitter taste and mineral scaling.
2. **Turbidity (NTU, 0.5 – 100.0):** Cloudiness caused by suspended sediments, microalgae, and clay particles. WHO drinking water guidelines recommend $\le 5.0\ \text{NTU}$ to ensure effective disinfection.
3. **Dissolved Oxygen (DO, mg/L, 1.0 – 12.0):** Free molecular oxygen dissolved in water. CPCB Class A/B criteria prescribe $\ge 6.0\ \text{mg/L}$ for healthy aquatic ecosystems and drinking water sources. Levels $<4.0\ \text{mg/L}$ induce hypoxic distress.
4. **Temperature (°C, 10.0 – 40.0):** Ambient aquatic temperature influences biological metabolic rates and gas solubility.
5. **Conductivity ($\mu\text{S/cm}$, 50 – 2,000):** Electrical conductance reflecting ionic concentration. Freshwater upper indicative threshold is typically $1,500\ \mu\text{S/cm}$.
6. **Total Dissolved Solids (TDS, ppm, 30 – 1,500):** Aggregate mass of dissolved inorganic salts and organic matter. Desirable limit under **BIS IS 10500:2012** is $\le 500\ \text{ppm}$.

---

## 7. WQI Methodology & Scoring Standard

The composite Water Quality Index aggregates discrete sub-indices into a continuous scale ($0$ to $100$):

$$\text{WQI} = \sum_{i=1}^{n} w_i \cdot q_i$$

Where:
* $w_i$ is the relative weighting factor ($\sum_{i=1}^n w_i = 1.00$).
* $q_i$ is the parameter-specific sub-index score ($0 \le q_i \le 100$).

### 7.1 Parameter Weights & Sub-Index Scoring Functions
* **pH ($w = 0.20$):** $q = \text{clip}(100 - 3.5 \times (|pH - 7.0|)^2,\ 0,\ 100)$
* **Turbidity ($w = 0.15$):** $q = \text{clip}(100 \times e^{-0.03 \times \text{Turbidity}},\ 0,\ 100)$
* **Dissolved Oxygen ($w = 0.20$):** $q = \text{clip}(100 \times (1 - e^{-0.35 \times \text{DO}}),\ 0,\ 100)$
* **Temperature ($w = 0.10$):** $q = \text{clip}(100 - 0.45 \times (|\text{Temp} - 25.0|)^2,\ 0,\ 100)$
* **Conductivity ($w = 0.15$):** $q = \text{clip}(100 \times e^{-0.0015 \times \text{Conductivity}},\ 0,\ 100)$
* **TDS ($w = 0.20$):** $q = \text{clip}(100 \times e^{-0.002 \times \text{TDS}},\ 0,\ 100)$

### 7.2 Continuous Classification Boundaries
To eliminate classification gaps, the scale uses gapless continuous intervals:
- **90.0 – 100.0:** 🟢 **Excellent** (Pristine, requires minimal disinfection)
- **75.0 – 89.9:** 🔵 **Good** (Safe for domestic consumption after standard treatment)
- **50.0 – 74.9:** 🟡 **Moderate** (Acceptable for irrigation & aquatic fauna; requires treatment)
- **25.0 – 49.9:** 🟠 **Poor** (Significant degradation; requires intensive treatment)
- **0.0 – 24.9:** 🔴 **Very Poor** (Severely polluted; anoxic; unsafe for direct use)

---

## 8. Synthetic Benchmark Generation

Because continuous field IoT probes with authenticated composite WQI labels are not openly accessible in volume, this project employs a **latent environmental condition simulation**:
1. A latent environmental variable $Q \sim U(0, 1)$ simulates environmental water state ($Q \to 1$ pristine, $Q \to 0$ degraded).
2. All parameters are drawn conditionally from $Q$:
   - High $Q$ yields neutral pH, low turbidity, high DO, ambient temperature, and low conductivity/TDS.
   - Low $Q$ induces acidic or alkaline pH drift, high turbidity, hypoxic DO, extreme temperature, and elevated mineral conduction.
3. Natural coupling is enforced between Conductivity and TDS ($\text{TDS} = \text{Conductivity} \times U(0.52, 0.68) + \epsilon$).
4. Controlled Gaussian noise prevents mathematical triviality.

---

## 9. Machine Learning Methodology

The machine learning workflow follows standard reproducible practices:
1. **Train/Test Splitting:** 80% training split (2,400 samples) and 20% test split (600 samples) with fixed seed (`random_state=42`).
2. **K-Fold Cross-Validation:** 5-fold cross-validation on the training set to verify generalisation stability.
3. **Evaluation Metrics:**
   - Coefficient of Determination ($R^2$)
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)

---

## 10. Regression Model Comparison (Actual Benchmark Results)

Evaluated on 600 unseen test samples:

| Model Algorithm | $R^2$ Score | MAE (WQI points) | RMSE | MAPE (%) | 5-Fold CV $R^2$ |
|---|---|---|---|---|---|
| 🥇 **Gradient Boosting Regressor** | **0.9964** | **0.5982** | **0.8076** | **1.09%** | **0.9962 ± 0.0004** |
| 🥈 **Random Forest Regressor** | 0.9900 | 0.9449 | 1.3481 | 1.75% | 0.9897 ± 0.0011 |
| 🥉 **Extra Trees Regressor** | 0.9892 | 0.9773 | 1.4061 | 1.84% | 0.9894 ± 0.0010 |
| 🔹 **Linear Regression (Baseline)** | 0.9542 | 2.2098 | 2.8913 | 3.88% | 0.9540 ± 0.0028 |

### Honest Interpretation of High $R^2$:
The $R^2$ score of $0.9964$ achieved by Gradient Boosting reflects that the ensemble algorithm has successfully learned the non-linear transformation between the six sensor parameters and the synthetic benchmark WQI scoring formula. **This should not be interpreted as proving universal predictive accuracy across uncalibrated natural lakes or rivers.**

---

## 11. Potability Classification (Kaggle Real Dataset)

Evaluated on 656 unseen test samples without data leakage:

| Metric | Result | Description / Practical Meaning |
|---|---|---|
| **Accuracy** | **65.85%** | Overall percentage of correct classifications |
| **Precision** | **57.84%** | When the model predicts water is potable, it is correct 57.8% of the time |
| **Recall** | **46.09%** | The model detects 46.1% of all truly potable water samples |
| **F1-Score** | **0.5130** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.6761** | Area under the Receiver Operating Characteristic curve |

### Confusion Matrix (Test Set, N = 656):
- **True Negatives (TN):** $314$ (Correctly classified non-potable)
- **False Positives (FP):** $86$ (Unsafe water incorrectly classified as safe)
- **False Negatives (FN):** $138$ (Safe water incorrectly flagged as unsafe)
- **True Positives (TP):** $118$ (Correctly classified potable)

### Critical Limitation Discussion:
A recall of $46.09\%$ illustrates that physicochemical parameters (pH, hardness, chloramines, sulfate, TDS) alone cannot reliably classify drinking safety. In real municipal and environmental systems, microbiological pathogens (*E. coli*, coliform bacteria), trace heavy metals (Lead, Arsenic), and pesticides dictate actual drinkability.

---

## 12. Model Feature Importance Interpretation

Feature importances extracted from the Gradient Boosting regressor on the benchmark dataset:
1. **Conductivity:** $68.59\%$
2. **Total Dissolved Solids (TDS):** $11.64\%$
3. **Dissolved Oxygen (DO):** $10.92\%$
4. **Turbidity:** $6.32\%$
5. **Temperature:** $2.08\%$
6. **pH:** $0.44\%$

### Physical Collinearity Consideration:
Conductivity and TDS share strong physical collinearity ($\text{TDS} \propto \text{Conductivity}$). In decision-tree splitting, when two features provide similar partitioning information, the tree selects one early, which inflates its calculated importance relative to the other. Therefore, Conductivity's $68.6\%$ importance is an artifact of the benchmark parameterization and tree splitting, not evidence that conductivity is universally more ecologically critical than pH or DO.

---

## 13. Interactive Streamlit Dashboard

The platform is structured into three dedicated tabs:
1. **🔮 Predict & Simulator:** Real-time sliders with domain input bounds, random sample simulation, WQI score card with continuous category badge, transparent parameter threshold diagnostics against WHO/BIS standards, and session history.
2. **📊 Dataset Explorer:** Multi-dataset selector for cleaned CPCB data, Kaggle potability data, and benchmark data with summary statistics, missingness profiles, and correlation matrices.
3. **🏆 ML Models & Performance:** Model leaderboard, feature importance bar charts, Kaggle potability confusion matrix, and pipeline reproduction instructions.

---

## 14. Environmental Significance

From an Environmental Studies (EVS) perspective, this project demonstrates:
1. **Real-time IoT Screening:** Demonstrates how multi-sensor arrays can serve as early-warning screening mechanisms to alert municipal water authorities before large-scale contamination propagates.
2. **Continuous Indexing:** Shows how complex multi-dimensional chemistry data can be synthesized into an interpretable metric for public awareness.
3. **The Necessity of Biological & Chemical Grounding:** Highlights that machine learning models must be contextualized within thermodynamic limits and biological realities rather than treated as unverified black boxes.

---

## 15. Limitations

To ensure scientific honesty and academic defensibility:
1. **Synthetic WQI Training Target:** The regression model is trained on a simulated dataset whose target is defined by the project's benchmark scoring formula.
2. **Historical & Non-telemetric Field Data:** The CPCB dataset represents historical field monitoring (2003–2014) with intermittent sampling rather than live IoT feeds.
3. **Limited Potability Recall:** The potability classifier achieves $\approx 46\%$ recall due to significant feature overlap and the absence of microbial pathogen data.
4. **Simplified Diagnostic Thresholds:** Real regulatory compliance requires complex laboratory testing protocols and legal sampling procedures that cannot be duplicated by 6 electronic sensors alone.
5. **Educational Disclaimer:** Predictions must not be treated as certified laboratory testing or legal regulatory compliance.

---

## 16. Future Scope

1. **Hardware Integration:** Connect ESP32/Raspberry Pi microcontrollers with physical waterproof pH, turbidity, TDS, and temperature sensors for live telemetry.
2. **Regional Calibration:** Calibrate specific WQI weightings to regional river typologies (e.g., Himalayan snowmelt rivers vs. peninsular rainfed rivers).
3. **Integration of Biological Indicators:** Incorporate rapid microbiological assays (such as enzymatic coliform tests) to dramatically improve potability classification.
4. **Time-Series Forecasting:** Implement LSTM or temporal convolutional networks to predict downstream contamination arrival times following industrial discharge events.

---

## 17. Conclusion

The **AI-Based Water Quality Assessment Platform** demonstrates a complete, technically sound, and scientifically defensible EVS Semester 3 project. By diagnosing and repairing historical CPCB data corruption, eliminating data leakage in classification preprocessing, deploying gapless WQI classification, benchmarking four regression algorithms, and providing honest interpretations of model performance and limitations, this project bridges modern computer science with environmental engineering rigor.
