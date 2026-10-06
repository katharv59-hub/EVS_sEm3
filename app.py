"""
AI-Based Water Quality Prediction & Analytics Dashboard
======================================================
Interactive Streamlit application featuring:
1. WQI Prediction Simulator (Random Forest / Gradient Boosting)
2. Comprehensive Dataset Explorer (CPCB India, Kaggle Potability, and Physics Benchmark)
3. Model Benchmark Leaderboard & Feature Importance Analytics

Usage:
    streamlit run app.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.calculate_wqi import (
    FEATURES, TARGET, QUALITY_CATEGORIES,
    get_quality_category, get_category_color,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE = os.path.join(MODEL_DIR, "wqi_model.pkl")
POTABILITY_MODEL_FILE = os.path.join(MODEL_DIR, "potability_model.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
DATA_FILE = os.path.join(DATA_DIR, "water_quality.csv")
CPCB_FILE = os.path.join(DATA_DIR, "cleaned_indian_cpcb_water_quality.csv")
KAGGLE_FILE = os.path.join(DATA_DIR, "cleaned_kaggle_water_potability.csv")

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Water Quality Intelligence System",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for Premium Look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0d47a1;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #546e7a;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    .wqi-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #90caf9;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.08);
    }
    .wqi-value {
        font-size: 3.6rem;
        font-weight: 800;
        margin: 0.4rem 0;
    }
    .wqi-label {
        font-size: 0.95rem;
        color: #455a64;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    .quality-badge {
        display: inline-block;
        padding: 0.45rem 1.6rem;
        border-radius: 20px;
        font-size: 1.15rem;
        font-weight: 700;
        color: white;
        margin-top: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .metric-container {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e40af;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .param-table {
        width: 100%;
        border-collapse: collapse;
    }
    .param-table td {
        padding: 0.6rem 1rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .param-name { color: #334155; font-weight: 500; }
    .param-value { color: #1e40af; font-weight: 700; text-align: right; }
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 1.4rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
    }
    .info-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        font-size: 0.92rem;
        color: #166534;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper Functions & Resource Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load trained models from disk."""
    wqi_mod = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None
    pot_mod = joblib.load(POTABILITY_MODEL_FILE) if os.path.exists(POTABILITY_MODEL_FILE) else None
    return wqi_mod, pot_mod

@st.cache_data
def load_metrics():
    """Load saved training metrics."""
    if not os.path.exists(METRICS_FILE):
        return None
    with open(METRICS_FILE, "r") as f:
        return json.load(f)

@st.cache_data
def load_datasets():
    """Load benchmark and real datasets."""
    datasets = {}
    if os.path.exists(DATA_FILE):
        datasets["synthetic"] = pd.read_csv(DATA_FILE)
    if os.path.exists(CPCB_FILE):
        datasets["cpcb"] = pd.read_csv(CPCB_FILE)
    if os.path.exists(KAGGLE_FILE):
        datasets["kaggle"] = pd.read_csv(KAGGLE_FILE)
    return datasets

def generate_random_sample() -> dict:
    """Generate realistic correlated random water sample."""
    rng = np.random.default_rng()
    quality = rng.uniform(0.1, 0.9)
    ph = np.clip(7.0 + (1 - quality) * rng.choice([-1, 1]) * rng.uniform(0.5, 2.5) + rng.normal(0, 0.2), 4.5, 10.0)
    turbidity = np.clip((1 - quality) * rng.uniform(20, 80) + quality * rng.uniform(0.5, 8) + rng.normal(0, 2), 0.5, 100.0)
    do = np.clip(quality * rng.uniform(7, 11) + (1 - quality) * rng.uniform(1.5, 5) + rng.normal(0, 0.4), 1.0, 12.0)
    temp = np.clip((rng.uniform(19, 27) if quality > 0.5 else rng.choice([rng.uniform(10, 16), rng.uniform(33, 39)])) + rng.normal(0, 1), 10.0, 40.0)
    conductivity = np.clip((1 - quality) * rng.uniform(700, 1800) + quality * rng.uniform(60, 350) + rng.normal(0, 30), 50.0, 2000.0)
    tds = np.clip(conductivity * rng.uniform(0.5, 0.7) + rng.normal(0, 15), 30.0, 1500.0)
    return {
        "pH": round(float(ph), 2),
        "Turbidity": round(float(turbidity), 2),
        "Dissolved Oxygen": round(float(do), 2),
        "Temperature": round(float(temp), 2),
        "Conductivity": round(float(conductivity), 2),
        "TDS": round(float(tds), 2),
    }

# Load assets
model, potability_model = load_models()
metrics = load_metrics()
datasets = load_datasets()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">💧 AI Water Quality Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Environmental Data Analytics & Machine Learning Prediction System</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Top-Level Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_datasets, tab_models = st.tabs([
    "🔮 Predict & Simulator",
    "📊 Dataset Explorer (Internet & Benchmark)",
    "🏆 ML Models & Performance"
])

# ===========================================================================
# TAB 1: PREDICTION & SIMULATOR
# ===========================================================================
with tab_predict:
    st.sidebar.markdown("## 🔬 Water Sample Inputs")
    if st.sidebar.button("🎲 Generate Random Sample", use_container_width=True):
        st.session_state["sample"] = generate_random_sample()

    defaults = st.session_state.get("sample", {
        "pH": 7.0, "Turbidity": 5.0, "Dissolved Oxygen": 8.0,
        "Temperature": 25.0, "Conductivity": 300.0, "TDS": 200.0,
    })

    ph = st.sidebar.slider("pH", 4.5, 10.0, float(defaults["pH"]), 0.1, format="%.1f")
    turbidity = st.sidebar.slider("Turbidity (NTU)", 0.5, 100.0, float(defaults["Turbidity"]), 0.5, format="%.1f")
    do_val = st.sidebar.slider("Dissolved Oxygen (mg/L)", 1.0, 12.0, float(defaults["Dissolved Oxygen"]), 0.1, format="%.1f")
    temperature = st.sidebar.slider("Temperature (°C)", 10.0, 40.0, float(defaults["Temperature"]), 0.5, format="%.1f")
    conductivity = st.sidebar.slider("Conductivity (µS/cm)", 50.0, 2000.0, float(defaults["Conductivity"]), 10.0, format="%.0f")
    tds = st.sidebar.slider("TDS (ppm)", 30.0, 1500.0, float(defaults["TDS"]), 10.0, format="%.0f")

    predict_clicked = st.sidebar.button("🔍 Predict Water Quality", use_container_width=True, type="primary")

    if predict_clicked or "prediction" in st.session_state:
        input_data = pd.DataFrame([{
            "pH": ph,
            "Turbidity": turbidity,
            "Dissolved Oxygen": do_val,
            "Temperature": temperature,
            "Conductivity": conductivity,
            "TDS": tds,
        }], columns=FEATURES)

        if model is not None:
            predicted_wqi = float(np.clip(model.predict(input_data)[0], 0, 100))
            category = get_quality_category(predicted_wqi)
            color = get_category_color(category)

            st.session_state["prediction"] = {
                "wqi": predicted_wqi,
                "category": category,
                "color": color,
                "inputs": {"pH": ph, "Turbidity": turbidity, "Dissolved Oxygen": do_val, "Temperature": temperature, "Conductivity": conductivity, "TDS": tds}
            }

            if predict_clicked:
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].append({
                    "pH": ph, "Turbidity": turbidity, "DO": do_val,
                    "Temp": temperature, "Cond": conductivity, "TDS": tds,
                    "WQI": round(predicted_wqi, 1), "Category": category,
                })

    pred = st.session_state.get("prediction")
    if pred:
        col_res, col_det = st.columns([1, 1])
        with col_res:
            st.markdown(
                f'<div class="wqi-card">'
                f'<div class="wqi-label">Estimated Water Quality Index</div>'
                f'<div class="wqi-value" style="color: {pred["color"]}">{pred["wqi"]:.1f}</div>'
                f'<span class="quality-badge" style="background-color: {pred["color"]}">{pred["category"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="section-header">📊 WQI Classification Scale</div>', unsafe_allow_html=True)
            scale_df = pd.DataFrame(QUALITY_CATEGORIES, columns=["Min", "Max", "Category"])
            st.dataframe(scale_df, use_container_width=True, hide_index=True)

        with col_det:
            st.markdown('<div class="section-header">🧪 Active Sample Readings</div>', unsafe_allow_html=True)
            inputs = pred["inputs"]
            p_table = f"""
            <table class="param-table">
                <tr><td class="param-name">pH</td><td class="param-value">{inputs['pH']:.1f}</td></tr>
                <tr><td class="param-name">Turbidity</td><td class="param-value">{inputs['Turbidity']:.1f} NTU</td></tr>
                <tr><td class="param-name">Dissolved Oxygen</td><td class="param-value">{inputs['Dissolved Oxygen']:.1f} mg/L</td></tr>
                <tr><td class="param-name">Temperature</td><td class="param-value">{inputs['Temperature']:.1f} °C</td></tr>
                <tr><td class="param-name">Conductivity</td><td class="param-value">{inputs['Conductivity']:.0f} µS/cm</td></tr>
                <tr><td class="param-name">Total Dissolved Solids (TDS)</td><td class="param-value">{inputs['TDS']:.0f} ppm</td></tr>
            </table>
            """
            st.markdown(p_table, unsafe_allow_html=True)

            # Parameter Advisory
            st.markdown('<div class="section-header">💡 Parameter Diagnostics</div>', unsafe_allow_html=True)
            advisories = []
            if inputs["pH"] < 6.5:
                advisories.append("⚠️ **Acidic pH (<6.5)**: Risk of pipe corrosion and heavy metal leaching.")
            elif inputs["pH"] > 8.5:
                advisories.append("⚠️ **Alkaline pH (>8.5)**: May cause mineral scaling and bitter taste.")
            if inputs["Turbidity"] > 5.0:
                advisories.append(f"⚠️ **High Turbidity ({inputs['Turbidity']:.1f} NTU)**: Exceeds WHO guideline of 5 NTU.")
            if inputs["Dissolved Oxygen"] < 6.0:
                advisories.append(f"⚠️ **Low Dissolved Oxygen ({inputs['Dissolved Oxygen']:.1f} mg/L)**: Can stress aquatic organisms.")
            if inputs["TDS"] > 500:
                advisories.append(f"⚠️ **High TDS ({inputs['TDS']:.0f} ppm)**: Recommended aesthetic threshold is 500 ppm.")
            if not advisories:
                advisories.append("✅ **All parameters are within normal drinking/aquatic ranges!**")
            for adv in advisories:
                st.write(adv)

    history = st.session_state.get("history", [])
    if history:
        st.markdown("---")
        st.markdown('<div class="section-header">📜 Prediction History (Current Session)</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2: DATASET EXPLORER
# ===========================================================================
with tab_datasets:
    st.markdown('<p class="section-header">📁 Environmental Water Quality Datasets</p>', unsafe_allow_html=True)
    st.write(
        "Here you can explore **real internet datasets** downloaded from public environmental repositories "
        "alongside the **calibrated physics-informed benchmark dataset**."
    )

    dataset_choice = st.radio(
        "Select Dataset to Inspect:",
        [
            "🔬 Synthetic Sensor Benchmark Dataset (3,000 samples)",
            "🌐 Indian CPCB River & Water Bodies Dataset (1,991 field records)",
            "🧪 Kaggle Global Water Potability Dataset (3,276 water bodies)"
        ],
        horizontal=True
    )

    if "Synthetic" in dataset_choice and "synthetic" in datasets:
        df_curr = datasets["synthetic"]
        st.markdown("""
        **Dataset Overview**: Calibrated environmental simulation with physical inter-parameter correlations.
        - **Features**: pH, Turbidity, Dissolved Oxygen, Temperature, Conductivity, TDS
        - **Target**: Water Quality Index (WQI, 0–100)
        """)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", f"{df_curr.shape[0]:,}")
        c2.metric("Parameters", f"{df_curr.shape[1]}")
        c3.metric("Average WQI", f"{df_curr['WQI'].mean():.2f}")

        st.markdown("#### Sample Records (First 10 rows)")
        st.dataframe(df_curr.head(10), use_container_width=True)

        st.markdown("#### Summary Statistics (mean, std, percentiles)")
        st.dataframe(df_curr.describe().round(2), use_container_width=True)

        st.markdown("#### Feature Correlation Matrix")
        st.dataframe(df_curr.corr().round(3), use_container_width=True)

    elif "Indian CPCB" in dataset_choice and "cpcb" in datasets:
        df_curr = datasets["cpcb"]
        st.markdown("""
        **Dataset Overview**: Central Pollution Control Board (CPCB) Ministry of Environment, India.
        - **Monitored Rivers/Water Bodies**: Across Indian states (Ganga, Yamuna, Godavari, Krishna, etc.)
        - **Key Parameters**: Temperature, Dissolved Oxygen (DO), pH, Conductivity, BOD, Nitrate, Coliform
        """)
        c1, c2, c3 = st.columns(3)
        c1.metric("Field Monitoring Records", f"{df_curr.shape[0]:,}")
        c2.metric("Monitored States", f"{df_curr['State'].nunique()}")
        c3.metric("Average Dissolved Oxygen", f"{df_curr['Dissolved_Oxygen'].mean():.2f} mg/L")

        st.markdown("#### Sample Records (First 10 rows)")
        st.dataframe(df_curr.head(10), use_container_width=True)

        st.markdown("#### Summary Statistics of Field Measurements")
        st.dataframe(df_curr.describe().round(2), use_container_width=True)

    elif "Kaggle" in dataset_choice and "kaggle" in datasets:
        df_curr = datasets["kaggle"]
        st.markdown("""
        **Dataset Overview**: Kaggle Global Water Potability Dataset.
        - **Objective**: Laboratory water chemistry evaluation for human drinkability (Potability = 0 / 1)
        - **Features**: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity
        """)
        potable_pct = (df_curr['Potability'].mean() * 100)
        c1, c2, c3 = st.columns(3)
        c1.metric("Water Samples", f"{df_curr.shape[0]:,}")
        c2.metric("Potable (Drinkable)", f"{potable_pct:.1f}%")
        c3.metric("Non-Potable", f"{100 - potable_pct:.1f}%")

        st.markdown("#### Sample Records (First 10 rows)")
        st.dataframe(df_curr.head(10), use_container_width=True)

        st.markdown("#### Summary Statistics")
        st.dataframe(df_curr.describe().round(2), use_container_width=True)

# ===========================================================================
# TAB 3: MODEL BENCHMARKING & PERFORMANCE
# ===========================================================================
with tab_models:
    st.markdown('<p class="section-header">🏆 Machine Learning Model Evaluation & Leaderboard</p>', unsafe_allow_html=True)

    if metrics and "wqi_regression_models" in metrics:
        reg_models = metrics["wqi_regression_models"]
        leaderboard_data = []
        for name, m in reg_models.items():
            leaderboard_data.append({
                "Model": name,
                "R² Score": m["r2"],
                "MAE": m["mae"],
                "RMSE": m["rmse"],
                "MAPE (%)": m.get("mape", "N/A"),
                "5-Fold CV R²": f"{m.get('cv_r2_mean', 'N/A')} ± {m.get('cv_r2_std', 'N/A')}"
            })
        leaderboard_df = pd.DataFrame(leaderboard_data).sort_values("R² Score", ascending=False)

        st.markdown("#### WQI Regression Models Leaderboard")
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

        # Highlight best model
        best_name = metrics.get("best_wqi_model", "GradientBoosting")
        st.success(f"🏆 **Best Performing Model**: **{best_name}** with **R² = {reg_models[best_name]['r2']:.4f}** and **MAE = {reg_models[best_name]['mae']:.4f}**")

        col_fi1, col_fi2 = st.columns([1, 1])
        with col_fi1:
            st.markdown("#### 📈 Feature Importance (WQI Regression)")
            best_fi = reg_models[best_name].get("feature_importances", {})
            if best_fi:
                fi_df = pd.DataFrame(sorted(best_fi.items(), key=lambda x: x[1], reverse=True), columns=["Parameter", "Importance"])
                st.bar_chart(fi_df.set_index("Parameter"), horizontal=True, color="#1565c0")

        with col_fi2:
            st.markdown("#### 🧪 Potability Classifier (Kaggle Real Data)")
            pot_metrics = metrics.get("potability_classification", {})
            if pot_metrics:
                m1, m2 = st.columns(2)
                m1.metric("Accuracy", f"{pot_metrics.get('accuracy', 0)*100:.1f}%")
                m2.metric("Precision", f"{pot_metrics.get('precision', 0)*100:.1f}%")
                m3, m4 = st.columns(2)
                m3.metric("Recall", f"{pot_metrics.get('recall', 0)*100:.1f}%")
                m4.metric("ROC-AUC", f"{pot_metrics.get('roc_auc', 0):.4f}")

    st.markdown("---")
    st.markdown("""
    #### ⚙️ How to Retrain Models:
    Run the following command in terminal to refresh data pipelines and train all models:
    ```bash
    python src/data_pipeline.py
    python src/train_model.py
    ```
    """)
