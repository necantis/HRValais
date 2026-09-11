"""
utils/ibm_data.py
Loader and mapper for the IBM HR Employee Attrition dataset (WA_Fn-UseC_-HR-Employee-Attrition.csv).
Provides mapped pillars and longitudinal structures for Mixed Models and Timeseries dashboards.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

IBM_DATASET_LABEL = "📊 Dataset démo IBM (Attrition)"

def _scale(val, lo, hi, out_lo=1.0, out_hi=5.0) -> float:
    if pd.isna(val):
        return 3.0
    return float(np.clip(out_lo + (val - lo) / (hi - lo) * (out_hi - out_lo), out_lo, out_hi))

def map_ibm_row_to_pillars(row: pd.Series) -> dict:
    """Map an IBM HR row to the 7 HR Valais pillars (1-5 scale)."""
    # Recrutement
    r1 = _scale(row.get("JobSatisfaction", 3), 1, 4)
    r2 = _scale(row.get("EnvironmentSatisfaction", 3), 1, 4)
    r3 = _scale(row.get("RelationshipSatisfaction", 3), 1, 4)

    # Compétences
    c4 = _scale(row.get("TrainingTimesLastYear", 2), 0, 6)
    c5 = _scale(row.get("JobLevel", 2), 1, 5)
    c6 = _scale(row.get("JobInvolvement", 3), 1, 4)

    # Performance
    p7 = _scale(row.get("PerformanceRating", 3), 1, 4)
    p8 = _scale(row.get("JobSatisfaction", 3), 1, 4)
    p9 = _scale(row.get("WorkLifeBalance", 3), 1, 4)

    # Rémunération
    monthly = row.get("MonthlyIncome", 5000)
    rem10 = _scale(monthly, 1009, 19999)
    pct_hike = row.get("PercentSalaryHike", 12)
    rem11 = _scale(pct_hike, 11, 25)
    rem12 = _scale(row.get("StockOptionLevel", 1), 0, 3)

    # QVT
    overtime_pen = 0 if str(row.get("OverTime", "No")) == "Yes" else 2
    qvt13 = _scale(row.get("WorkLifeBalance", 3), 1, 4)
    qvt14 = _scale(row.get("EnvironmentSatisfaction", 3), 1, 4)
    qvt15 = max(1.0, _scale(row.get("WorkLifeBalance", 3), 1, 4) - overtime_pen)

    # Droit
    d16 = _scale(row.get("YearsAtCompany", 5), 0, 40)
    d17 = _scale(row.get("YearsWithCurrManager", 3), 0, 20)
    d18 = _scale(row.get("NumCompaniesWorked", 2), 0, 9, out_lo=5, out_hi=1)

    # Transverse
    t19 = _scale(row.get("RelationshipSatisfaction", 3), 1, 4)
    t20 = _scale(row.get("JobSatisfaction", 3), 1, 4)
    t21 = _scale(row.get("TotalWorkingYears", 10), 0, 40)

    def avg(*vals): return float(np.mean(vals))

    return dict(
        recrutement_avg=avg(r1, r2, r3),
        competences_avg=avg(c4, c5, c6),
        performance_avg=avg(p7, p8, p9),
        remuneration_avg=avg(rem10, rem11, rem12),
        qvt_avg=avg(qvt13, qvt14, qvt15),
        droit_avg=avg(d16, d17, d18),
        transverse_avg=avg(t19, t20, t21),
    )

@st.cache_data(ttl=3600)
def load_ibm_dataset() -> pd.DataFrame:
    """Load and map the full IBM HR Attrition dataset to HR Valais format."""
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(csv_path)

    rows = []
    for idx, r in raw.iterrows():
        p = map_ibm_row_to_pillars(r)
        attrition = str(r.get("Attrition", "No")) == "Yes"
        avg_score = np.mean([p["recrutement_avg"], p["competences_avg"], p["performance_avg"], p["qvt_avg"]])
        if attrition:
            state = "Resigned"
        elif avg_score >= 4.0:
            state = "Highly Engaged"
        elif avg_score >= 3.0:
            state = "Content"
        elif avg_score >= 2.0:
            state = "Passively Looking"
        else:
            state = "Resigned"

        tenure = int(r.get("YearsAtCompany", 1))
        month_idx = ((tenure * 3 + idx) % 24) + 1

        rows.append({
            "month_index": month_idx,
            "engagement_state": state,
            "attrition_flag": attrition,
            "attrition_int": 1 if attrition else 0,
            "resigned_int": 1 if state == "Resigned" else 0,
            **p,
        })
    return pd.DataFrame(rows)
