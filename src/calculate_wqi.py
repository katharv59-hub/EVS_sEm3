"""
Water Quality Index (WQI) Calculation & Threshold Diagnostics Module
=====================================================================
Single source of truth for:
1. Feature definitions and weights
2. Parameter sub-index scoring functions
3. Continuous quality category classification
4. Domain reference thresholds and diagnostic assessments

NOTE: This is an educational, physics-informed benchmark scoring formula
designed for demonstration and research purposes. It is NOT an official
statutory regulatory standard (such as full NSF-WQI or BIS IS 10500:2012),
though individual parameter reference thresholds are aligned with WHO and
BIS guidelines where noted.
"""

from typing import Dict, List, Tuple, Any
import numpy as np


# ---------------------------------------------------------------------------
# Feature Definitions & Weights (Must sum strictly to 1.0)
# ---------------------------------------------------------------------------
FEATURES = [
    "pH",
    "Turbidity",
    "Dissolved Oxygen",
    "Temperature",
    "Conductivity",
    "TDS",
]

TARGET = "WQI"

WEIGHTS: Dict[str, float] = {
    "pH": 0.20,
    "Turbidity": 0.15,
    "Dissolved Oxygen": 0.20,
    "Temperature": 0.10,
    "Conductivity": 0.15,
    "TDS": 0.20,
}

# Ensure weights sum to 1.0
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum exactly to 1.0"


# ---------------------------------------------------------------------------
# Quality Categories (Continuous, gapless boundaries)
# ---------------------------------------------------------------------------
# Format: (min_inclusive, max_exclusive_or_inclusive, label, color)
QUALITY_CATEGORIES: List[Tuple[float, float, str, str]] = [
    (90.0, 100.0, "Excellent", "#00c853"),
    (75.0, 90.0, "Good", "#64dd17"),
    (50.0, 75.0, "Moderate", "#ffd600"),
    (25.0, 50.0, "Poor", "#ff6d00"),
    (0.0, 25.0, "Very Poor", "#dd2c00"),
]


# ---------------------------------------------------------------------------
# Parameter Reference Thresholds (WHO / BIS IS 10500:2012 Guidelines)
# ---------------------------------------------------------------------------
REFERENCE_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "pH": {
        "min": 6.5,
        "max": 8.5,
        "unit": "dimensionless",
        "standard": "BIS IS 10500:2012 / WHO",
        "description": "Acceptable drinking water range (6.5 to 8.5).",
    },
    "Turbidity": {
        "max": 5.0,
        "unit": "NTU",
        "standard": "WHO Guideline / BIS Permissible Limit",
        "description": "Recommended threshold is <= 5.0 NTU for domestic water.",
    },
    "Dissolved Oxygen": {
        "min": 6.0,
        "unit": "mg/L",
        "standard": "CPCB Class A/B Freshwater Criteria",
        "description": "Healthy surface water requires >= 6.0 mg/L for aquatic life.",
    },
    "Temperature": {
        "min": 15.0,
        "max": 32.0,
        "unit": "°C",
        "standard": "Environmental Surface Water Reference",
        "description": "Typical tropical/subtropical surface water thermal range.",
    },
    "Conductivity": {
        "max": 1500.0,
        "unit": "µS/cm",
        "standard": "WHO Indicative Guidelines",
        "description": "Freshwater electrical conductivity upper indicative limit.",
    },
    "TDS": {
        "max": 500.0,
        "unit": "ppm",
        "standard": "BIS IS 10500:2012 Acceptable Limit",
        "description": "Desirable drinking water TDS limit is <= 500 ppm.",
    },
}


# ---------------------------------------------------------------------------
# Individual Parameter Scoring Functions (0 to 100 scale)
# ---------------------------------------------------------------------------
def ph_score(ph: float) -> float:
    """
    Score pH on a 0-100 scale.
    Neutral pH (~7.0) receives the optimal score (100).
    A quadratic penalty reflects biological stress as pH deviates from neutrality.
    """
    ph_clamped = np.clip(float(ph), 0.0, 14.0)
    deviation = abs(ph_clamped - 7.0)
    score = 100.0 - 3.5 * (deviation ** 2)
    return float(np.clip(score, 0.0, 100.0))


def turbidity_score(turbidity: float) -> float:
    """
    Score turbidity on a 0-100 scale.
    Lower turbidity scores higher via smooth exponential decay.
    """
    turb_clamped = max(0.0, float(turbidity))
    score = 100.0 * np.exp(-0.03 * turb_clamped)
    return float(np.clip(score, 0.0, 100.0))


def do_score(do: float) -> float:
    """
    Score dissolved oxygen on a 0-100 scale.
    Higher DO indicates higher oxygenation; saturates smoothly above 8-10 mg/L.
    """
    do_clamped = max(0.0, float(do))
    score = 100.0 * (1.0 - np.exp(-0.35 * do_clamped))
    return float(np.clip(score, 0.0, 100.0))


def temperature_score(temp: float) -> float:
    """
    Score temperature on a 0-100 scale.
    Moderate surface water temperature (~25°C) receives optimal score.
    """
    temp_val = float(temp)
    deviation = abs(temp_val - 25.0)
    score = 100.0 - 0.45 * (deviation ** 2)
    return float(np.clip(score, 0.0, 100.0))


def conductivity_score(conductivity: float) -> float:
    """
    Score electrical conductivity on a 0-100 scale.
    Lower mineral/salt conduction scores higher via smooth exponential decay.
    """
    cond_clamped = max(0.0, float(conductivity))
    score = 100.0 * np.exp(-0.0015 * cond_clamped)
    return float(np.clip(score, 0.0, 100.0))


def tds_score(tds: float) -> float:
    """
    Score Total Dissolved Solids on a 0-100 scale.
    Lower dissolved solids score higher via exponential decay.
    """
    tds_clamped = max(0.0, float(tds))
    score = 100.0 * np.exp(-0.002 * tds_clamped)
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Composite WQI Calculation
# ---------------------------------------------------------------------------
def calculate_wqi(
    ph: float,
    turbidity: float,
    do: float,
    temp: float,
    conductivity: float,
    tds: float,
) -> float:
    """
    Calculate composite Water Quality Index (0-100).

    WQI = sum(weight_i * score_i) for all 6 parameters.
    """
    wqi = (
        WEIGHTS["pH"] * ph_score(ph)
        + WEIGHTS["Turbidity"] * turbidity_score(turbidity)
        + WEIGHTS["Dissolved Oxygen"] * do_score(do)
        + WEIGHTS["Temperature"] * temperature_score(temp)
        + WEIGHTS["Conductivity"] * conductivity_score(conductivity)
        + WEIGHTS["TDS"] * tds_score(tds)
    )
    return float(np.clip(wqi, 0.0, 100.0))


def calculate_wqi_row(row) -> float:
    """Convenience wrapper for calculating WQI on a pandas DataFrame row."""
    return calculate_wqi(
        ph=row["pH"],
        turbidity=row["Turbidity"],
        do=row["Dissolved Oxygen"],
        temp=row["Temperature"],
        conductivity=row["Conductivity"],
        tds=row["TDS"],
    )


# ---------------------------------------------------------------------------
# Category Classification & Diagnostics (Gapless)
# ---------------------------------------------------------------------------
def get_quality_category(wqi: float) -> str:
    """
    Map WQI score to category with gapless continuous intervals.
    """
    wqi_val = float(wqi)
    if wqi_val >= 90.0:
        return "Excellent"
    elif wqi_val >= 75.0:
        return "Good"
    elif wqi_val >= 50.0:
        return "Moderate"
    elif wqi_val >= 25.0:
        return "Poor"
    else:
        return "Very Poor"


def get_category_color(category: str) -> str:
    """Return hex color code associated with quality category."""
    color_map = {cat: color for _, _, cat, color in QUALITY_CATEGORIES}
    return color_map.get(category, "#9e9e9e")


def evaluate_threshold_diagnostics(inputs: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Evaluate parameters against configured domain reference thresholds.
    Returns a list of structured diagnostic dictionaries.
    """
    diagnostics = []

    # 1. pH
    ph_val = float(inputs.get("pH", 7.0))
    ph_ref = REFERENCE_THRESHOLDS["pH"]
    if ph_val < ph_ref["min"]:
        diagnostics.append({
            "parameter": "pH",
            "value": f"{ph_val:.1f}",
            "threshold": f"{ph_ref['min']} – {ph_ref['max']}",
            "unit": ph_ref["unit"],
            "status": "Below Reference (Acidic)",
            "severity": "Warning",
            "interpretation": "Elevated acidity. May accelerate pipe corrosion and heavy metal dissolution.",
            "standard": ph_ref["standard"],
        })
    elif ph_val > ph_ref["max"]:
        diagnostics.append({
            "parameter": "pH",
            "value": f"{ph_val:.1f}",
            "threshold": f"{ph_ref['min']} – {ph_ref['max']}",
            "unit": ph_ref["unit"],
            "status": "Above Reference (Alkaline)",
            "severity": "Warning",
            "interpretation": "Elevated alkalinity. May cause mineral encrustation and bitter taste.",
            "standard": ph_ref["standard"],
        })
    else:
        diagnostics.append({
            "parameter": "pH",
            "value": f"{ph_val:.1f}",
            "threshold": f"{ph_ref['min']} – {ph_ref['max']}",
            "unit": ph_ref["unit"],
            "status": "Within Configured Threshold",
            "severity": "Normal",
            "interpretation": "Neutral to slightly alkaline; within standard drinking water range.",
            "standard": ph_ref["standard"],
        })

    # 2. Turbidity
    turb_val = float(inputs.get("Turbidity", 0.0))
    turb_ref = REFERENCE_THRESHOLDS["Turbidity"]
    if turb_val > turb_ref["max"]:
        diagnostics.append({
            "parameter": "Turbidity",
            "value": f"{turb_val:.1f}",
            "threshold": f"<= {turb_ref['max']}",
            "unit": turb_ref["unit"],
            "status": "Exceeds Reference Threshold",
            "severity": "Warning",
            "interpretation": "Suspended matter exceeds aesthetic guideline; potential microbial shelter.",
            "standard": turb_ref["standard"],
        })
    else:
        diagnostics.append({
            "parameter": "Turbidity",
            "value": f"{turb_val:.1f}",
            "threshold": f"<= {turb_ref['max']}",
            "unit": turb_ref["unit"],
            "status": "Within Configured Threshold",
            "severity": "Normal",
            "interpretation": "Clear appearance meeting general clarity guideline.",
            "standard": turb_ref["standard"],
        })

    # 3. Dissolved Oxygen
    do_val = float(inputs.get("Dissolved Oxygen", 0.0))
    do_ref = REFERENCE_THRESHOLDS["Dissolved Oxygen"]
    if do_val < do_ref["min"]:
        diagnostics.append({
            "parameter": "Dissolved Oxygen",
            "value": f"{do_val:.1f}",
            "threshold": f">= {do_ref['min']}",
            "unit": do_ref["unit"],
            "status": "Below Reference Threshold",
            "severity": "Warning",
            "interpretation": "Sub-optimal aeration; oxygen depletion indicates organic contamination or thermal stress.",
            "standard": do_ref["standard"],
        })
    else:
        diagnostics.append({
            "parameter": "Dissolved Oxygen",
            "value": f"{do_val:.1f}",
            "threshold": f">= {do_ref['min']}",
            "unit": do_ref["unit"],
            "status": "Within Configured Threshold",
            "severity": "Normal",
            "interpretation": "Healthy oxygenation supporting aquatic life and aerobic biological balance.",
            "standard": do_ref["standard"],
        })

    # 4. Temperature
    temp_val = float(inputs.get("Temperature", 25.0))
    temp_ref = REFERENCE_THRESHOLDS["Temperature"]
    if temp_val < temp_ref["min"] or temp_val > temp_ref["max"]:
        diagnostics.append({
            "parameter": "Temperature",
            "value": f"{temp_val:.1f}",
            "threshold": f"{temp_ref['min']} – {temp_ref['max']}",
            "unit": temp_ref["unit"],
            "status": "Outside Reference Range",
            "severity": "Info",
            "interpretation": "Temperature is outside typical ambient freshwater reference; influences oxygen solubility.",
            "standard": temp_ref["standard"],
        })
    else:
        diagnostics.append({
            "parameter": "Temperature",
            "value": f"{temp_val:.1f}",
            "threshold": f"{temp_ref['min']} – {temp_ref['max']}",
            "unit": temp_ref["unit"],
            "status": "Within Configured Threshold",
            "severity": "Normal",
            "interpretation": "Standard ambient aquatic temperature range.",
            "standard": temp_ref["standard"],
        })

    # 5. Conductivity
    cond_val = float(inputs.get("Conductivity", 0.0))
    cond_ref = REFERENCE_THRESHOLDS["Conductivity"]
    if cond_val > cond_ref["max"]:
        diagnostics.append({
            "parameter": "Conductivity",
            "value": f"{cond_val:.0f}",
            "threshold": f"<= {cond_ref['max']}",
            "unit": cond_ref["unit"],
            "status": "Exceeds Reference Threshold",
            "severity": "Warning",
            "interpretation": "High ionic concentration indicating substantial dissolved dissolved minerals or industrial runoff.",
            "standard": cond_ref["standard"],
        })
    else:
        diagnostics.append({
            "parameter": "Conductivity",
            "value": f"{cond_val:.0f}",
            "threshold": f"<= {cond_ref['max']}",
            "unit": cond_ref["unit"],
            "status": "Within Configured Threshold",
            "severity": "Normal",
            "interpretation": "Moderate ionic conductance consistent with typical freshwater.",
            "standard": cond_ref["standard"],
        })

    # 6. TDS
    tds_val = float(inputs.get("TDS", 0.0))
    tds_ref = REFERENCE_THRESHOLDS["TDS"]
    if tds_val > tds_ref["max"]:
        diagnostics.append({
            "parameter": "TDS",
            "value": f"{tds_val:.0f}",
            "threshold": f"<= {tds_ref['max']}",
            "unit": tds_ref["unit"],
            "status": "Exceeds Reference Threshold",
            "severity": "Warning",
            "interpretation": "Dissolved solids exceed acceptable aesthetic threshold; may taste saline or mineralized.",
            "standard": tds_ref["standard"],
        })
    else:
        diagnostics.append({
            "parameter": "TDS",
            "value": f"{tds_val:.0f}",
            "threshold": f"<= {tds_ref['max']}",
            "unit": tds_ref["unit"],
            "status": "Within Configured Threshold",
            "severity": "Normal",
            "interpretation": "Total dissolved solids within acceptable drinking guideline.",
            "standard": tds_ref["standard"],
        })

    return diagnostics
