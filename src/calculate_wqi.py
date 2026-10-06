"""
Water Quality Index (WQI) Calculation Module

This module implements a transparent, project-specific WQI scoring methodology.
Each water-quality parameter is scored individually (0-100), and scores are
combined using a weighted average to produce the final WQI.

NOTE: This is a project-specific scoring formula created for demonstration
purposes. It is NOT a universally accepted regulatory WQI standard.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Quality Category Thresholds (configurable in one place)
# ---------------------------------------------------------------------------
QUALITY_CATEGORIES = [
    (90, 100, "Excellent"),
    (75, 89, "Good"),
    (50, 74, "Moderate"),
    (25, 49, "Poor"),
    (0, 24, "Very Poor"),
]

# ---------------------------------------------------------------------------
# WQI Component Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "pH": 0.20,
    "Turbidity": 0.15,
    "Dissolved Oxygen": 0.20,
    "Temperature": 0.10,
    "Conductivity": 0.15,
    "TDS": 0.20,
}

# Feature list used for ML training — order matters for consistency
FEATURES = [
    "pH",
    "Turbidity",
    "Dissolved Oxygen",
    "Temperature",
    "Conductivity",
    "TDS",
]

TARGET = "WQI"


# ---------------------------------------------------------------------------
# Individual Parameter Scoring Functions (each returns 0-100)
# ---------------------------------------------------------------------------

def ph_score(ph: float) -> float:
    """
    Score pH on 0-100 scale.
    Ideal pH is near 7 (neutral). Score decreases as pH moves away from 7.
    Uses a quadratic penalty for smoother behavior near the optimum.
    """
    deviation = abs(ph - 7.0)
    # Quadratic penalty: max deviation of ~3 units maps to score ~0
    score = 100 - 3.5 * (deviation ** 2)
    return float(np.clip(score, 0, 100))


def turbidity_score(turbidity: float) -> float:
    """
    Score turbidity on 0-100 scale.
    Lower turbidity is better. Uses an exponential decay for smooth behavior.
    """
    # At turbidity=0 → score=100, at turbidity~50 → score~20
    score = 100 * np.exp(-0.03 * turbidity)
    return float(np.clip(score, 0, 100))


def do_score(do: float) -> float:
    """
    Score dissolved oxygen on 0-100 scale.
    Higher DO is better, with diminishing returns above ~8 mg/L.
    Uses a saturating function.
    """
    # Sigmoid-like curve: score rises steeply for low DO, plateaus around 8-10
    score = 100 * (1 - np.exp(-0.35 * do))
    return float(np.clip(score, 0, 100))


def temperature_score(temp: float) -> float:
    """
    Score temperature on 0-100 scale.
    Moderate temperatures (~25°C) receive the highest score.
    Score decreases as temperature moves away from the preferred range.
    """
    preferred = 25.0
    deviation = abs(temp - preferred)
    # Quadratic penalty: deviation of ~15°C maps to score ~0
    score = 100 - 0.45 * (deviation ** 2)
    return float(np.clip(score, 0, 100))


def conductivity_score(conductivity: float) -> float:
    """
    Score conductivity on 0-100 scale.
    Lower-to-moderate conductivity is better for this simplified index.
    Uses an exponential decay.
    """
    # At conductivity=0 → score=100, at conductivity~1000 → score~22
    score = 100 * np.exp(-0.0015 * conductivity)
    return float(np.clip(score, 0, 100))


def tds_score(tds: float) -> float:
    """
    Score TDS on 0-100 scale.
    Lower TDS receives a higher score.
    Uses an exponential decay.
    """
    # At TDS=0 → score=100, at TDS~800 → score~22
    score = 100 * np.exp(-0.002 * tds)
    return float(np.clip(score, 0, 100))


# ---------------------------------------------------------------------------
# Combined WQI Calculation
# ---------------------------------------------------------------------------

def calculate_wqi(ph: float, turbidity: float, do: float,
                  temp: float, conductivity: float, tds: float) -> float:
    """
    Calculate the composite Water Quality Index (0-100).

    WQI = 0.20*pH_score + 0.15*turbidity_score + 0.20*DO_score
        + 0.10*temperature_score + 0.15*conductivity_score + 0.20*TDS_score

    Parameters
    ----------
    ph : float - pH value
    turbidity : float - Turbidity in NTU
    do : float - Dissolved Oxygen in mg/L
    temp : float - Temperature in °C
    conductivity : float - Conductivity in µS/cm
    tds : float - Total Dissolved Solids in ppm

    Returns
    -------
    float - WQI score clipped to [0, 100]
    """
    wqi = (
        WEIGHTS["pH"] * ph_score(ph)
        + WEIGHTS["Turbidity"] * turbidity_score(turbidity)
        + WEIGHTS["Dissolved Oxygen"] * do_score(do)
        + WEIGHTS["Temperature"] * temperature_score(temp)
        + WEIGHTS["Conductivity"] * conductivity_score(conductivity)
        + WEIGHTS["TDS"] * tds_score(tds)
    )
    return float(np.clip(wqi, 0, 100))


def calculate_wqi_row(row) -> float:
    """Convenience wrapper for applying calculate_wqi to a DataFrame row."""
    return calculate_wqi(
        ph=row["pH"],
        turbidity=row["Turbidity"],
        do=row["Dissolved Oxygen"],
        temp=row["Temperature"],
        conductivity=row["Conductivity"],
        tds=row["TDS"],
    )


# ---------------------------------------------------------------------------
# Quality Category Mapping
# ---------------------------------------------------------------------------

def get_quality_category(wqi: float) -> str:
    """
    Map a WQI score to a quality category.

    Returns one of: Excellent, Good, Moderate, Poor, Very Poor
    """
    for low, high, category in QUALITY_CATEGORIES:
        if low <= wqi <= high:
            return category
    # Fallback for edge cases (e.g. exactly 0 or rounding)
    if wqi >= 90:
        return "Excellent"
    if wqi < 25:
        return "Very Poor"
    return "Unknown"


def get_category_color(category: str) -> str:
    """Return a hex color for each quality category (for dashboard display)."""
    colors = {
        "Excellent": "#00c853",   # green
        "Good": "#64dd17",        # light green
        "Moderate": "#ffd600",    # yellow
        "Poor": "#ff6d00",        # orange
        "Very Poor": "#dd2c00",   # red
    }
    return colors.get(category, "#9e9e9e")
