"""
AI-Based Water Quality Intelligence Platform
=============================================
Environmental Studies (EVS) Semester 3 Academic & Machine Learning Project
Interactive Streamlit Dashboard with:
1. Predict & Simulator: Multi-sensor input validation, model-based WQI estimate,
   transparent parameter diagnostics, and educational disclaimers.
2. Dataset Explorer: Exploration of cleaned CPCB India field data, Kaggle potability
   data, and physics-calibrated sensor benchmark data.
3. ML Models & Performance: Comparative regression leaderboard, confusion matrix,
   and scientifically honest feature importance & limitation interpretations.

Usage:
    streamlit run app.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from src.calculate_wqi import (
    FEATURES,
    TARGET,
    QUALITY_CATEGORIES,
    REFERENCE_THRESHOLDS,
    get_quality_category,
    get_category_color,
    evaluate_threshold_diagnostics,
)

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

WQI_MODEL_FILE = os.path.join(MODEL_DIR, "wqi_model.pkl")
POTABILITY_MODEL_FILE = os.path.join(MODEL_DIR, "potability_model.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
SUMMARY_FILE = os.path.join(DATA_DIR, "dataset_summary.json")

BENCHMARK_FILE = os.path.join(DATA_DIR, "water_quality.csv")
CPCB_FILE = os.path.join(DATA_DIR, "cleaned_indian_cpcb_water_quality.csv")
KAGGLE_FILE = os.path.join(DATA_DIR, "cleaned_kaggle_water_potability.csv")

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Water Quality Assessment Platform",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Professional Presentation
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0d47a1;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #455a64;
        margin-top: 0;
        margin-bottom: 1.1rem;
    }
    .wqi-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        border: 1px solid #90caf9;
        margin-bottom: 1rem;
        box-shadow: 0 3px 10px rgba(13, 71, 161, 0.07);
    }
    .wqi-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .wqi-label {
        font-size: 0.9rem;
        color: #37474f;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }
    .quality-badge {
        display: inline-block;
        padding: 0.4rem 1.6rem;
        border-radius: 20px;
        font-size: 1.15rem;
        font-weight: 700;
        color: white;
        margin-top: 0.4rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .section-header {
        font-size: 1.22rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 1.3rem;
        margin-bottom: 0.7rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.3rem;
    }
    .disclaimer-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        font-size: 0.88rem;
        color: #92400e;
        margin-bottom: 1rem;
    }
    .concept-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Resource Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load trained models from disk if present."""
    wqi_mod = joblib.load(WQI_MODEL_FILE) if os.path.exists(WQI_MODEL_FILE) else None
    pot_mod = joblib.load(POTABILITY_MODEL_FILE) if os.path.exists(POTABILITY_MODEL_FILE) else None
    return wqi_mod, pot_mod


@st.cache_data
def load_metrics():
    """Load evaluation metrics JSON."""
    if not os.path.exists(METRICS_FILE):
        return None
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


@st.cache_data
def load_datasets():
    """Load datasets for explorer tab."""
    data = {}
    if os.path.exists(BENCHMARK_FILE):
        data["benchmark"] = pd.read_csv(BENCHMARK_FILE)
    if os.path.exists(CPCB_FILE):
        data["cpcb"] = pd.read_csv(CPCB_FILE)
    if os.path.exists(KAGGLE_FILE):
        data["kaggle"] = pd.read_csv(KAGGLE_FILE)
    return data


def generate_random_sample() -> dict:
    """Generate realistic correlated random water sample using latent variable."""
    rng = np.random.default_rng()
    q = rng.uniform(0.15, 0.85)

    ph = np.clip(7.0 + (1 - q) * rng.choice([-1, 1]) * rng.uniform(0.5, 2.2) + rng.normal(0, 0.2), 4.8, 9.6)
    turbidity = np.clip((1 - q) * rng.uniform(15, 75) + q * rng.uniform(0.5, 7.0) + rng.normal(0, 1.5), 0.5, 95.0)
    do = np.clip(q * rng.uniform(7.0, 11.0) + (1 - q) * rng.uniform(1.8, 5.0) + rng.normal(0, 0.3), 1.2, 11.5)
    temp = np.clip((rng.uniform(20, 26) if q > 0.5 else rng.choice([rng.uniform(11, 15), rng.uniform(33, 38)])) + rng.normal(0, 0.8), 11.0, 39.0)
    conductivity = np.clip((1 - q) * rng.uniform(650, 1750) + q * rng.uniform(70, 320) + rng.normal(0, 25), 60.0, 1950.0)
    tds = np.clip(conductivity * rng.uniform(0.52, 0.68) + rng.normal(0, 12), 35.0, 1400.0)

    return {
        "pH": round(float(ph), 2),
        "Turbidity": round(float(turbidity), 2),
        "Dissolved Oxygen": round(float(do), 2),
        "Temperature": round(float(temp), 2),
        "Conductivity": round(float(conductivity), 1),
        "TDS": round(float(tds), 1),
    }


# Load cached assets
wqi_model, potability_pipeline = load_models()
metrics = load_metrics()
datasets = load_datasets()

# ---------------------------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">💧 AI Water Quality Assessment Platform</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Environmental Studies (EVS) Semester 3 · Multiparameter Sensor Assessment & Machine Learning Benchmark</p>',
    unsafe_allow_html=True,
)

# Mandatory Educational Disclaimer
st.markdown(
    '<div class="disclaimer-box">'
    '<strong>⚠️ Academic Notice & Disclaimer:</strong> This platform is an educational and analytical research demonstration. '
    'Model outputs represent mathematical estimates and should not be treated as laboratory certification or a substitute for statutory regulatory water testing.'
    '</div>',
    unsafe_allow_html=True,
)

tab_predict, tab_datasets, tab_models = st.tabs([
    "🔮 Predict & Simulator",
    "📊 Dataset Explorer (CPCB, Kaggle & Benchmark)",
    "🏆 ML Models & Performance",
])


# ===========================================================================
# TAB 1: PREDICT & SIMULATOR
# ===========================================================================
with tab_predict:
    st.sidebar.markdown("### 🔬 Water Sample Inputs")
    if st.sidebar.button("🎲 Generate Realistic Sample", use_container_width=True):
        st.session_state["sample"] = generate_random_sample()

    defaults = st.session_state.get("sample", {
        "pH": 7.2, "Turbidity": 4.5, "Dissolved Oxygen": 7.8,
        "Temperature": 24.5, "Conductivity": 280.0, "TDS": 175.0,
    })

    # Sliders for physical inputs
    ph = st.sidebar.slider("pH", 0.0, 14.0, float(defaults["pH"]), 0.1, format="%.1f")
    turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 120.0, float(defaults["Turbidity"]), 0.5, format="%.1f")
    do_val = st.sidebar.slider("Dissolved Oxygen (mg/L)", 0.0, 15.0, float(defaults["Dissolved Oxygen"]), 0.1, format="%.1f")
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, float(defaults["Temperature"]), 0.5, format="%.1f")
    conductivity = st.sidebar.slider("Conductivity (µS/cm)", 0.0, 2500.0, float(defaults["Conductivity"]), 10.0, format="%.0f")
    tds = st.sidebar.slider("TDS (ppm)", 0.0, 1800.0, float(defaults["TDS"]), 10.0, format="%.0f")

    # Domain Input Validation
    input_errors = []
    if ph < 4.0 or ph > 10.5:
        input_errors.append(f"pH value ({ph:.1f}) is extreme for environmental surface water.")
    if temperature < 5.0 or temperature > 45.0:
        input_errors.append(f"Temperature ({temperature:.1f} °C) is outside typical ambient freshwater limits.")
    if do_val < 0.5:
        input_errors.append(f"Dissolved oxygen ({do_val:.1f} mg/L) indicates near-complete anoxia / severe sepsis.")

    if input_errors:
        for err in input_errors:
            st.sidebar.warning(f"⚠️ {err}")

    predict_btn = st.sidebar.button("🔍 Calculate Water Quality Assessment", use_container_width=True, type="primary")

    # Process prediction
    current_inputs = {
        "pH": ph,
        "Turbidity": turbidity,
        "Dissolved Oxygen": do_val,
        "Temperature": temperature,
        "Conductivity": conductivity,
        "TDS": tds,
    }

    if predict_btn or "last_prediction" in st.session_state:
        if wqi_model is not None:
            input_df = pd.DataFrame([current_inputs], columns=FEATURES)
            pred_wqi = float(np.clip(wqi_model.predict(input_df)[0], 0.0, 100.0))
            category = get_quality_category(pred_wqi)
            color = get_category_color(category)

            st.session_state["last_prediction"] = {
                "wqi": pred_wqi,
                "category": category,
                "color": color,
                "inputs": current_inputs,
            }

            if predict_btn:
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].append({
                    "pH": ph, "Turbidity": turbidity, "DO": do_val,
                    "Temp": temperature, "Conductivity": conductivity, "TDS": tds,
                    "Estimated WQI": round(pred_wqi, 1), "Category": category,
                })

    pred = st.session_state.get("last_prediction")
    if pred:
        col_res, col_diag = st.columns([1, 1.2])

        with col_res:
            # WQI Result Card
            st.markdown(
                f'<div class="wqi-card">'
                f'<div class="wqi-label">Model-Based Water Quality Index (WQI) Estimate</div>'
                f'<div class="wqi-value" style="color: {pred["color"]}">{pred["wqi"]:.1f}</div>'
                f'<span class="quality-badge" style="background-color: {pred["color"]}">{pred["category"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Continuous Quality Scale Reference
            st.markdown('<div class="section-header">📊 WQI Classification Scale (Continuous)</div>', unsafe_allow_html=True)
            scale_df = pd.DataFrame([
                {"Range": "90.0 – 100.0", "Category": "Excellent", "Description": "Pristine freshwater; requires minimal treatment."},
                {"Range": "75.0 – 89.9", "Category": "Good", "Description": "Safe for domestic use with conventional filtration."},
                {"Range": "50.0 – 74.9", "Category": "Moderate", "Description": "Suitable for irrigation & aquatic habitat; needs treatment."},
                {"Range": "25.0 – 49.9", "Category": "Poor", "Description": "Substantial contamination; requires intensive purification."},
                {"Range": "0.0 – 24.9", "Category": "Very Poor", "Description": "Severely degraded/hypoxic; unfit for direct use."},
            ])
            st.dataframe(scale_df, use_container_width=True, hide_index=True)

        with col_diag:
            st.markdown('<div class="section-header">📋 Parameter Threshold Diagnostics</div>', unsafe_allow_html=True)
            diagnostics = evaluate_threshold_diagnostics(pred["inputs"])
            diag_df = pd.DataFrame(diagnostics)[["parameter", "value", "threshold", "unit", "status", "standard"]]
            diag_df.columns = ["Parameter", "Input Value", "Threshold", "Unit", "Diagnostic Status", "Reference Standard"]
            st.dataframe(diag_df, use_container_width=True, hide_index=True)

            warnings_count = sum(1 for d in diagnostics if "Threshold" not in d["status"] or d["severity"] == "Warning")
            if warnings_count == 0:
                st.success("✅ **No configured threshold-based warnings were triggered for the selected parameters.**")
            else:
                st.warning(f"⚠️ **{warnings_count} parameter warning(s) flagged above against reference standards.**")

        # Clarification of 3 Concepts
        st.markdown('<div class="section-header">💡 Important Concept Distinction</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            **1. Water Quality Index (WQI)**  
            A continuous composite indicator (0–100) reflecting general environmental condition.  
            *A high WQI does not automatically guarantee that water is safe for unboiled human consumption.*
            """)
        with c2:
            st.markdown("""
            **2. Potability Classification**  
            A statistical classification of human drinkability based on chemical laboratory parameters.  
            *Biological microbes, viruses, and toxic metals require independent laboratory culture tests.*
            """)
        with c3:
            st.markdown("""
            **3. Threshold Diagnostics**  
            Deterministic checks against individual parameter guidelines (e.g. BIS IS 10500 / WHO).  
            *Identifies specific stress factors such as acidic pH or excessive turbidity independently of composite WQI.*
            """)

    history = st.session_state.get("history", [])
    if history:
        st.markdown("---")
        st.markdown('<div class="section-header">📜 Session Prediction History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 2: DATASET EXPLORER
# ===========================================================================
with tab_datasets:
    st.markdown('<p class="section-header">📁 Environmental Water Quality Datasets</p>', unsafe_allow_html=True)
    st.write(
        "Inspect the authenticated datasets used across this project: "
        "**Real-world Indian CPCB river data**, **Kaggle laboratory water potability data**, "
        "and the **Physics-calibrated sensor benchmark dataset**."
    )

    dataset_choice = st.radio(
        "Select Dataset:",
        [
            "🌐 Indian CPCB River Monitoring Dataset (1,991 field records)",
            "🧪 Kaggle Global Water Potability Dataset (3,276 laboratory records)",
            "🔬 Physics-Calibrated Sensor Benchmark Dataset (3,000 simulated samples)",
        ],
        horizontal=True,
    )

    if "CPCB" in dataset_choice and "cpcb" in datasets:
        df_curr = datasets["cpcb"]
        st.markdown("""
        **Dataset Overview**: Central Pollution Control Board (CPCB), Ministry of Environment, Forest & Climate Change, Government of India.
        - **Source & Retrieval**: Public educational GitHub mirror (`aditikhatri/-Indian-water-quality-analysis-and-prediction`).
        - **Data Quality Repairs**: An automated domain check repaired 89 historical records (2003–2004) where pH and Conductivity were inverted in the mirror source, and set 3 non-physical recording artifacts ($pH < 2.0$) to NaN.
        - **Key Parameters**: Temperature, Dissolved Oxygen (DO), pH, Electrical Conductivity, BOD, Nitrate, Coliform bacteria.
        """)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Field Observations", f"{len(df_curr):,}")
        m2.metric("Cleaned pH Range", f"{df_curr['pH'].min():.2f} – {df_curr['pH'].max():.2f}")
        m3.metric("Average Dissolved Oxygen", f"{df_curr['Dissolved_Oxygen'].mean():.2f} mg/L")
        m4.metric("Average BOD", f"{df_curr['BOD'].mean():.2f} mg/L")

        st.markdown("#### Sample Records (First 10 rows)")
        st.dataframe(df_curr.head(10), use_container_width=True)

        st.markdown("#### Statistical Distribution Summary")
        st.dataframe(df_curr.describe().round(2), use_container_width=True)

    elif "Kaggle" in dataset_choice and "kaggle" in datasets:
        df_curr = datasets["kaggle"]
        st.markdown("""
        **Dataset Overview**: Global Drinking Water Quality & Potability Study (Kaggle Community Dataset by Aditya Kadiwal).
        - **Retrieval**: Public educational mirror (`Sarthak-1408/Water-Potability`).
        - **Data Leakage Safeguard**: Missing values are preserved in the dataset so that imputation is strictly fitted on training splits only during pipeline training.
        - **Target**: `Potability` (1 = Potable, 0 = Non-potable).
        """)
        pot_count = int((df_curr["Potability"] == 1).sum())
        non_pot_count = int((df_curr["Potability"] == 0).sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Water Samples", f"{len(df_curr):,}")
        m2.metric("Potable (Safe)", f"{pot_count} ({pot_count/len(df_curr)*100:.1f}%)")
        m3.metric("Non-Potable", f"{non_pot_count} ({non_pot_count/len(df_curr)*100:.1f}%)")

        st.markdown("#### Sample Records (First 10 rows)")
        st.dataframe(df_curr.head(10), use_container_width=True)

        st.markdown("#### Missing Value Profile & Statistical Summary")
        st.dataframe(df_curr.describe().round(2), use_container_width=True)

    elif "Benchmark" in dataset_choice and "benchmark" in datasets:
        df_curr = datasets["benchmark"]
        st.markdown("""
        **Dataset Overview**: Physics-Calibrated Environmental Sensor Benchmark Dataset.
        - **Nature**: SIMULATED multiparameter environmental data generated for this project via latent condition modeling ($Q \in [0, 1]$).
        - **Physical Rules**: Includes natural coupling between Conductivity and TDS ($\text{TDS} \approx 0.5 - 0.7 \times \text{Conductivity}$) and gas solubility dynamics.
        - **Target**: Continuous Water Quality Index (WQI, 0–100) calculated via project benchmark scoring formula.
        """)
        m1, m2, m3 = st.columns(3)
        m1.metric("Simulated Samples", f"{len(df_curr):,}")
        m2.metric("Sensor Features", f"{len(FEATURES)}")
        m3.metric("Mean Benchmark WQI", f"{df_curr['WQI'].mean():.2f}")

        st.markdown("#### Sample Records (First 10 rows)")
        st.dataframe(df_curr.head(10), use_container_width=True)

        st.markdown("#### Summary Statistics")
        st.dataframe(df_curr.describe().round(2), use_container_width=True)

        st.markdown("#### Feature Correlation Matrix")
        st.dataframe(df_curr.corr().round(3), use_container_width=True)


# ===========================================================================
# TAB 3: ML MODELS & PERFORMANCE
# ===========================================================================
with tab_models:
    st.markdown('<p class="section-header">🏆 Machine Learning Model Evaluation & Leaderboard</p>', unsafe_allow_html=True)

    if metrics and "wqi_regression_models" in metrics:
        reg_models = metrics["wqi_regression_models"]
        leaderboard_data = []
        for name, m in reg_models.items():
            leaderboard_data.append({
                "Model Algorithm": name,
                "R² Score": m["r2"],
                "MAE": m["mae"],
                "RMSE": m["rmse"],
                "MAPE (%)": f"{m.get('mape', 'N/A')}%",
                "5-Fold CV R²": f"{m.get('cv_r2_mean', 'N/A')} ± {m.get('cv_r2_std', 'N/A')}",
            })
        leaderboard_df = pd.DataFrame(leaderboard_data).sort_values("R² Score", ascending=False)

        st.markdown("#### 1. WQI Regression Leaderboard (Continuous Estimation)")
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        best_name = metrics.get("best_wqi_model", "GradientBoosting")
        st.info(
            f"🔍 **Interpretation of High R² ({reg_models[best_name]['r2']:.4f}):** "
            "Because this model is trained on the benchmark dataset where WQI is generated from the project's "
            "mathematical scoring formula, the high $R^2$ indicates that the Gradient Boosting algorithm successfully learns "
            "and approximates this multi-sensor non-linear relationship. It should not be misinterpreted as proving "
            "predictive accuracy over arbitrary uncalibrated real-world water bodies."
        )

        col_fi, col_pot = st.columns([1, 1.2])

        with col_fi:
            st.markdown("#### 📈 Model Feature Importances (Benchmark)")
            best_fi = reg_models[best_name].get("feature_importances", {})
            if best_fi:
                fi_df = pd.DataFrame(
                    sorted(best_fi.items(), key=lambda x: x[1], reverse=True),
                    columns=["Parameter", "Importance Ratio"]
                )
                st.bar_chart(fi_df.set_index("Parameter"), horizontal=True, color="#1565c0")
                st.caption(
                    "Note: Conductivity and TDS share high physical collinearity, which causes tree-based algorithms "
                    "to assign the dominant split importance to Conductivity in this benchmark."
                )

        with col_pot:
            st.markdown("#### 🧪 Potability Classification (Kaggle Real Dataset)")
            pot_m = metrics.get("potability_classification", {})
            if pot_m:
                pm1, pm2, pm3, pm4 = st.columns(4)
                pm1.metric("Accuracy", f"{pot_m['accuracy']*100:.1f}%")
                pm2.metric("Precision", f"{pot_m['precision']*100:.1f}%")
                pm3.metric("Recall", f"{pot_m['recall']*100:.1f}%")
                pm4.metric("ROC-AUC", f"{pot_m['roc_auc']:.4f}")

                cm = pot_m.get("confusion_matrix", {})
                if cm:
                    cm_df = pd.DataFrame([
                        {"Actual Condition": "Non-Potable (Actual 0)", "Predicted Non-Potable (0)": f"TN = {cm['true_negatives']}", "Predicted Potable (1)": f"FP = {cm['false_positives']}"},
                        {"Actual Condition": "Potable (Actual 1)", "Predicted Non-Potable (0)": f"FN = {cm['false_negatives']}", "Predicted Potable (1)": f"TP = {cm['true_positives']}"},
                    ])
                    st.markdown("**Confusion Matrix (Evaluated on 656 Unseen Test Samples):**")
                    st.dataframe(cm_df, use_container_width=True, hide_index=True)

                st.caption(
                    "⚠️ **Honest Classifier Limitation:** While balanced class weighting improves recall to 46.1% and ROC-AUC to 0.676, "
                    "chemical laboratory features alone cannot reliably certify drinking water safety without biological pathogen testing."
                )

    st.markdown("---")
    st.markdown("""
    #### 🔄 Automated Pipeline Reproduction Commands:
    ```bash
    python src/data_pipeline.py    # Downloads datasets, performs domain cleaning, exports summaries
    python src/train_model.py      # Trains regression models and leakage-free potability pipeline
    python tests/test_validation.py # Runs system and scientific integrity validation suite
    ```
    """)
