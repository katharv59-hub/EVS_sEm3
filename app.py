"""
AI-Based Water Quality Prediction — Streamlit Dashboard

Provides a clean interface for predicting Water Quality Index (WQI)
from six water-quality parameters using a trained Random Forest model.

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
MODEL_FILE = os.path.join(BASE_DIR, "models", "wqi_model.pkl")
METRICS_FILE = os.path.join(BASE_DIR, "models", "metrics.json")
DATA_FILE = os.path.join(BASE_DIR, "data", "water_quality.csv")


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Water Quality Prediction",
    page_icon="💧",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Header styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0d47a1;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #546e7a;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    /* WQI result card */
    .wqi-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #90caf9;
        margin-bottom: 1rem;
    }
    .wqi-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .wqi-label {
        font-size: 1rem;
        color: #546e7a;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .quality-badge {
        display: inline-block;
        padding: 0.4rem 1.5rem;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
        margin-top: 0.5rem;
    }

    /* Metric cards */
    .metric-container {
        background: #f5f5f5;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1565c0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #757575;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Parameter table */
    .param-table {
        width: 100%;
        border-collapse: collapse;
    }
    .param-table td {
        padding: 0.6rem 1rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .param-name { color: #424242; font-weight: 500; }
    .param-value { color: #1565c0; font-weight: 700; text-align: right; }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #37474f;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #e3f2fd;
        padding-bottom: 0.3rem;
    }

    /* Info box */
    .info-box {
        background: #e8f5e9;
        border-left: 4px solid #43a047;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #2e7d32;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metrics():
    """Load saved metrics from disk."""
    if not os.path.exists(METRICS_FILE):
        return None
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


def generate_random_sample() -> dict:
    """
    Generate a realistic random water sample with correlated values.
    Uses the same latent-quality approach as the data generator.
    """
    rng = np.random.default_rng()
    quality = rng.uniform(0.1, 0.9)  # Avoid extremes for more interesting demos

    ph = 7.0 + (1 - quality) * rng.choice([-1, 1]) * rng.uniform(0.5, 2.5)
    ph += rng.normal(0, 0.2)
    ph = np.clip(ph, 4.5, 10.0)

    turbidity = (1 - quality) * rng.uniform(20, 80) + quality * rng.uniform(0.5, 8)
    turbidity += rng.normal(0, 2)
    turbidity = np.clip(turbidity, 0.5, 100.0)

    do = quality * rng.uniform(7, 11) + (1 - quality) * rng.uniform(1.5, 5)
    do += rng.normal(0, 0.4)
    do = np.clip(do, 1.0, 12.0)

    if quality > 0.5:
        temp = rng.uniform(19, 27)
    else:
        temp = rng.choice([rng.uniform(10, 16), rng.uniform(33, 39)])
    temp += rng.normal(0, 1)
    temp = np.clip(temp, 10.0, 40.0)

    conductivity = (1 - quality) * rng.uniform(700, 1800) + quality * rng.uniform(60, 350)
    conductivity += rng.normal(0, 30)
    conductivity = np.clip(conductivity, 50.0, 2000.0)

    tds = conductivity * rng.uniform(0.5, 0.7) + rng.normal(0, 15)
    tds = np.clip(tds, 30.0, 1500.0)

    return {
        "pH": round(float(ph), 2),
        "Turbidity": round(float(turbidity), 2),
        "Dissolved Oxygen": round(float(do), 2),
        "Temperature": round(float(temp), 2),
        "Conductivity": round(float(conductivity), 2),
        "TDS": round(float(tds), 2),
    }


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">💧 AI-Based Water Quality Prediction System</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning–Based Water Quality Assessment</p>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="info-box">'
    'Enter water-quality parameters below to estimate the Water Quality Index (WQI) '
    'using a trained Random Forest machine-learning model.'
    '</div>',
    unsafe_allow_html=True,
)

# Load model and metrics
model = load_model()
metrics = load_metrics()

if model is None:
    st.error(
        "⚠️ Trained model not found. Please run the training pipeline first:\n\n"
        "```\npython src/generate_data.py\npython src/train_model.py\n```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar — Input Panel
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🔬 Water Sample Parameters")

# Generate Sample button
if st.sidebar.button("🎲 Generate Random Sample", use_container_width=True):
    sample = generate_random_sample()
    st.session_state["sample"] = sample

# Get current sample values (defaults or generated)
defaults = st.session_state.get("sample", {
    "pH": 7.0, "Turbidity": 5.0, "Dissolved Oxygen": 8.0,
    "Temperature": 25.0, "Conductivity": 300.0, "TDS": 200.0,
})

ph = st.sidebar.slider("pH", min_value=4.5, max_value=10.0,
                         value=defaults["pH"], step=0.1, format="%.1f")
turbidity = st.sidebar.slider("Turbidity (NTU)", min_value=0.5, max_value=100.0,
                                value=defaults["Turbidity"], step=0.5, format="%.1f")
do_val = st.sidebar.slider("Dissolved Oxygen (mg/L)", min_value=1.0, max_value=12.0,
                             value=defaults["Dissolved Oxygen"], step=0.1, format="%.1f")
temperature = st.sidebar.slider("Temperature (°C)", min_value=10.0, max_value=40.0,
                                  value=defaults["Temperature"], step=0.5, format="%.1f")
conductivity = st.sidebar.slider("Conductivity (µS/cm)", min_value=50.0, max_value=2000.0,
                                   value=defaults["Conductivity"], step=10.0, format="%.0f")
tds = st.sidebar.slider("TDS (ppm)", min_value=30.0, max_value=1500.0,
                          value=defaults["TDS"], step=10.0, format="%.0f")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
predict_clicked = st.sidebar.button("🔍 Predict Water Quality",
                                     use_container_width=True,
                                     type="primary")

if predict_clicked:
    # Build input DataFrame with exact feature names
    input_data = pd.DataFrame([{
        "pH": ph,
        "Turbidity": turbidity,
        "Dissolved Oxygen": do_val,
        "Temperature": temperature,
        "Conductivity": conductivity,
        "TDS": tds,
    }], columns=FEATURES)

    try:
        predicted_wqi = float(model.predict(input_data)[0])
        predicted_wqi = np.clip(predicted_wqi, 0, 100)
        category = get_quality_category(predicted_wqi)
        color = get_category_color(category)

        # Store in session state for persistence
        st.session_state["prediction"] = {
            "wqi": predicted_wqi,
            "category": category,
            "color": color,
            "inputs": {
                "pH": ph, "Turbidity": turbidity,
                "Dissolved Oxygen": do_val, "Temperature": temperature,
                "Conductivity": conductivity, "TDS": tds,
            }
        }

        # Add to history
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append({
            "pH": ph, "Turbidity": turbidity, "DO": do_val,
            "Temp": temperature, "Cond": conductivity, "TDS": tds,
            "WQI": round(predicted_wqi, 1), "Category": category,
        })

    except Exception as e:
        st.error(f"Prediction error: {e}")


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

# Show prediction results if available
pred = st.session_state.get("prediction")

if pred:
    col_result, col_params = st.columns([1, 1])

    with col_result:
        # ── WQI Result Card ──
        st.markdown(
            f'<div class="wqi-card">'
            f'<div class="wqi-label">Predicted Water Quality Index</div>'
            f'<div class="wqi-value" style="color: {pred["color"]}">'
            f'{pred["wqi"]:.1f}</div>'
            f'<span class="quality-badge" style="background-color: {pred["color"]}">'
            f'{pred["category"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Quality Scale Reference ──
        st.markdown('<div class="section-header">📊 Quality Scale</div>',
                    unsafe_allow_html=True)
        scale_df = pd.DataFrame(QUALITY_CATEGORIES, columns=["Min", "Max", "Category"])
        st.dataframe(scale_df, use_container_width=True, hide_index=True)

    with col_params:
        # ── Parameter Summary ──
        st.markdown('<div class="section-header">📋 Parameter Readings</div>',
                    unsafe_allow_html=True)

        units = {
            "pH": "",
            "Turbidity": " NTU",
            "Dissolved Oxygen": " mg/L",
            "Temperature": " °C",
            "Conductivity": " µS/cm",
            "TDS": " ppm",
        }

        param_html = '<table class="param-table">'
        for param, value in pred["inputs"].items():
            unit = units.get(param, "")
            param_html += (
                f'<tr>'
                f'<td class="param-name">{param}</td>'
                f'<td class="param-value">{value}{unit}</td>'
                f'</tr>'
            )
        param_html += '</table>'
        st.markdown(param_html, unsafe_allow_html=True)

else:
    st.info("👈 Adjust parameters in the sidebar and click **Predict Water Quality** to see results.")


# ---------------------------------------------------------------------------
# Model Performance & Feature Importance
# ---------------------------------------------------------------------------
if metrics:
    st.markdown("---")

    col_metrics, col_importance = st.columns([1, 1])

    with col_metrics:
        st.markdown('<div class="section-header">🎯 Model Performance</div>',
                    unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-label">MAE</div>'
                f'<div class="metric-value">{metrics["MAE"]:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-label">RMSE</div>'
                f'<div class="metric-value">{metrics["RMSE"]:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-label">R²</div>'
                f'<div class="metric-value">{metrics["R2"]:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.caption(
            f"Trained on {metrics.get('train_size', 'N/A')} samples · "
            f"Tested on {metrics.get('test_size', 'N/A')} samples"
        )

    with col_importance:
        st.markdown('<div class="section-header">📈 Feature Importance</div>',
                    unsafe_allow_html=True)

        importances = metrics.get("feature_importances", {})
        if importances:
            imp_df = pd.DataFrame(
                sorted(importances.items(), key=lambda x: x[1], reverse=True),
                columns=["Feature", "Importance"],
            )
            st.bar_chart(imp_df.set_index("Feature"), horizontal=True,
                         color="#1565c0")
        else:
            st.write("Feature importances not available.")


# ---------------------------------------------------------------------------
# Prediction History (optional — session-based)
# ---------------------------------------------------------------------------
history = st.session_state.get("history", [])
if history:
    st.markdown("---")
    st.markdown('<div class="section-header">📜 Prediction History (This Session)</div>',
                unsafe_allow_html=True)
    hist_df = pd.DataFrame(history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Footer — How It Works
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("ℹ️ How It Works"):
    st.markdown("""
    **AI-Based Water Quality Prediction** uses a trained **Random Forest Regressor** to
    estimate the Water Quality Index (WQI) from six measurable water parameters.

    **Pipeline:**
    1. **Synthetic Data**: ~3,000 realistic water samples generated with correlated parameters
    2. **WQI Scoring**: A transparent, project-specific weighted formula calculates the target WQI
    3. **ML Training**: A Random Forest model learns the parameter → WQI relationship
    4. **Prediction**: New samples are scored using the trained model

    **Limitations:**
    - Uses synthetic data (not real sensor readings)
    - WQI formula is project-specific, not a regulatory standard
    - Not validated against laboratory measurements
    """)
