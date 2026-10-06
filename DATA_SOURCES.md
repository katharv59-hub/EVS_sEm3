# 🌐 Dataset Provenance & Data Sources Documentation

This document describes the provenance, origin, retrieval methodology, and preprocessing applied to all datasets used in this **EVS Semester 3** project.

---

## 1. Central Pollution Control Board (CPCB) India — River Monitoring Data

- **Dataset Identifier:** `cpcb_india`
- **Original Author & Source Authority:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest & Climate Change (MoEFCC), Government of India.
- **Retrieval Method & Mirror:** Public educational GitHub repository mirror (`aditikhatri/-Indian-water-quality-analysis-and-prediction`).
- **Exact Mirror URL:** `https://raw.githubusercontent.com/aditikhatri/-Indian-water-quality-analysis-and-prediction/master/water_dataX.csv`
- **License / Availability:** Open Government Data / Public educational research mirror.
- **Records Count:** 1,991 field station observations across Indian states and river basins (Ganga, Yamuna, Godavari, Krishna, Mandovi, Zuari, etc.) spanning years 2003–2014.
- **Raw Features:** `STATION CODE`, `LOCATIONS`, `STATE`, `Temp`, `D.O. (mg/l)`, `PH`, `CONDUCTIVITY (µmhos/cm)`, `B.O.D. (mg/l)`, `NITRATENAN N+ NITRITENANN (mg/l)`, `FECAL COLIFORM (MPN/100ml)`, `TOTAL COLIFORM (MPN/100ml)Mean`, `year`.
- **Target / Use Case:** Exploratory Data Analysis (EDA) of real-world Indian aquatic environments; field baseline comparison.

### Preprocessing & Data Quality Repairs:
1. **Column-Swap Anomaly Correction:**  
   During auditing, 89 records (indices 1901–1990, stations recorded in 2003–2004) were found to have inverted `PH` and `CONDUCTIVITY` values in the mirror source (e.g., station 1435 in Vapi, Gujarat had `PH = 67,115` and `CONDUCTIVITY = 5.0`). The pipeline algorithmically detected records matching $pH > 14$ and $\text{Conductivity} \in [0, 14]$ and restored them to their true physical columns.
2. **Environmental Domain Boundary Filtering:**  
   In natural surface waters, pH is physically constrained. Three extreme non-physical values ($pH < 2.0$ or $> 12.0$) representing recording artifacts or sensor errors were flagged and set to `NaN`.
   - Resulting clean pH range: **2.60 to 9.01 (mean: 7.21)**.
3. **Data Type Parsing:** String `'NAN'` markers were converted to standard floating-point `NaN`.
4. **Output File:** `data/cleaned_indian_cpcb_water_quality.csv`.

### Limitations of Source:
- The dataset contains historical field sampling data from 2003–2014, not live telemetry.
- Varying sampling frequencies and missing values exist across parameters (e.g., Fecal Coliform missing in 316 samples).
- Retrieved via a public community mirror rather than an automated live CPCB API.

---

## 2. Kaggle Global Water Potability Dataset

- **Dataset Identifier:** `kaggle_potability`
- **Original Source:** Drinking Water Quality Dataset published on Kaggle by Aditya Kadiwal, sourced from global water testing studies.
- **Retrieval Method & Mirror:** Public educational GitHub repository mirror (`Sarthak-1408/Water-Potability`).
- **Exact Mirror URL:** `https://raw.githubusercontent.com/Sarthak-1408/Water-Potability/main/water_potability.csv`
- **License / Availability:** CC0 Public Domain.
- **Records Count:** 3,276 water sample laboratory measurements.
- **Features (9 parameters):**
  1. `ph`: pH of water ($0.0 - 14.0$).
  2. `Hardness`: Capacity to precipitate soap (mg/L).
  3. `Solids`: Total Dissolved Solids / TDS (ppm).
  4. `Chloramines`: Disinfectant residual concentration (ppm).
  5. `Sulfate`: Dissolved sulfate concentration (mg/L).
  6. `Conductivity`: Electrical conductivity ($\mu\text{S/cm}$).
  7. `Organic_carbon`: Total organic carbon (ppm).
  8. `Trihalomethanes`: Disinfection byproduct concentration ($\mu\text{g/L}$).
  9. `Turbidity`: Light attenuation / clarity (NTU).
- **Target Variable:** `Potability` (Binary: `1` = Potable/Safe for human consumption, `0` = Non-potable/Unsafe). Class balance: 61.0% Non-potable (1,998) vs. 39.0% Potable (1,278).

### Preprocessing & Leakage Prevention:
1. **Zero Data Leakage:** Unlike common online scripts that perform median imputation over the entire dataset *prior* to splitting, this pipeline leaves missing values untouched in the cleaned CSV (`ph`: 491 missing, `Sulfate`: 781 missing, `Trihalomethanes`: 162 missing).
2. **Pipeline Imputation:** Median imputation (`SimpleImputer(strategy='median')`) is fitted strictly on the 80% training split during model training inside an `sklearn.pipeline.Pipeline`.
3. **Output File:** `data/cleaned_kaggle_water_potability.csv`.

### Limitations of Source:
- Laboratory metrics do not include biological pathogen counts (e.g., *E. coli*), heavy metals (e.g., Lead, Arsenic), or pesticide traces.
- Classes exhibit substantial overlap across features, resulting in limited classification recall ($\sim 46\%$).
- Predictions from this model cannot replace certified laboratory microbiological testing.

---

## 3. Physics-Calibrated Sensor Benchmark Dataset

- **Dataset Identifier:** `synthetic_benchmark`
- **Original Author & Source:** Generated for this project via [`src/generate_data.py`](file:///d:/PYTHON%20MYSELF/Anti-G%20projects/EVS/water-quality-ai/src/generate_data.py).
- **Motivation:** Practical multiparameter IoT environmental monitoring typically deploys 6 specific electronic probes (pH, Turbidity, DO, Temperature, Conductivity, and TDS). Real continuous 6-probe IoT field data with matched composite WQI labels are rarely published openly in sufficient volume for clean regression training.
- **Records Count:** 3,000 synthetic observations.
- **Features (6 sensor inputs):**
  1. `pH`: 4.5 – 10.0 (dimensionless).
  2. `Turbidity`: 0.5 – 100.0 NTU.
  3. `Dissolved Oxygen`: 1.0 – 12.0 mg/L.
  4. `Temperature`: 10.0 – 40.0 °C.
  5. `Conductivity`: 50.0 – 2,000.0 $\mu\text{S/cm}$.
  6. `TDS`: 30.0 – 1,500.0 ppm.
- **Target Variable:** `WQI` (Continuous: 0 to 100) calculated via project benchmark scoring formula in [`src/calculate_wqi.py`](file:///d:/PYTHON%20MYSELF/Anti-G%20projects/EVS/water-quality-ai/src/calculate_wqi.py).

### Simulation Methodology:
1. **Latent Quality Condition ($Q \in [0, 1]$):** Drives mutual parameter variance. Good water ($Q \to 1$) yields neutral pH, low turbidity, high DO, moderate temperature, and low conductivity/TDS. Poor water ($Q \to 0$) yields skewed pH, high turbidity, hypoxic DO, extreme temperature, and elevated conductivity/TDS.
2. **Physical Inter-parameter Coupling:**  
   - Total Dissolved Solids is derived directly from Conductivity with realistic natural freshwater conversion factors ($\text{TDS} \approx 0.5 \text{ to } 0.7 \times \text{Conductivity}$).
   - Controlled Gaussian noise is added to prevent trivial algebraic determinism.
3. **Output File:** `data/water_quality.csv`.

### Limitations & Clear Distinction:
- **This is SIMULATED data, NOT real hardware sensor readings.**
- High model regression accuracy ($R^2 \approx 0.996$) demonstrates that the machine learning algorithm successfully learns the mathematical relationship defined by the benchmark scoring formula, NOT real-world universal water chemistry.
