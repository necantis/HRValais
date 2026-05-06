"""
db/seed.py
Cold-start seeder for HR Valais.

Populates:
  - firms (3 demo firms)
  - users (5 demo users across roles)
  - ofs_macro_data (from .px files, with synthetic fallback)
  - survey_responses (mapped from IBM CSV + synthetic longitudinal CSV)

All operations are idempotent: existing rows are not duplicated.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from db.database import get_session, DATA_DIR
from db.models import Firm, User, SurveyResponse, OFSMacroData
from utils.auth import hash_password
from utils.ofs_parser import load_ofs_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo credentials (shown in UI for testing)
# ---------------------------------------------------------------------------
DEMO_USERS = [
    {
        "username": "Employee1_firmA",
        "password": "password123",
        "role": "employee",
        "display_name": "Alice (Firm A)",
        "firm_key": "firm_a",
    },
    {
        "username": "Employee1_firmB",
        "password": "password123",
        "role": "employee",
        "display_name": "Bob (Firm B)",
        "firm_key": "firm_b",
    },
    {
        "username": "Manager1_firmA",
        "password": "password123",
        "role": "hr_manager",
        "display_name": "Chloé (Firm A)",
        "firm_key": "firm_a",
    },
    {
        "username": "Manager1_firmB",
        "password": "password123",
        "role": "hr_manager",
        "display_name": "David (Firm B)",
        "firm_key": "firm_b",
    },
    {
        "username": "admin1",
        "password": "password123",
        "role": "admin",
        "display_name": "Charlie Admin",
        "firm_key": "firm_c",
    },
]

DEMO_FIRMS = {
    "firm_a": {"name": "Alpina Services SA", "industry_domain": "Services"},
    "firm_b": {"name": "Rhône Industrie Sàrl", "industry_domain": "Manufacturing"},
    "firm_c": {"name": "HR Valais (Platform)", "industry_domain": "Human Resources"},
}

# ---------------------------------------------------------------------------
# IBM HR Analytics → 7 Pillar mapping
# ---------------------------------------------------------------------------
# We derive approximate 1-5 scores from IBM's 1-4 / 1-5 / continuous variables.

def _map_ibm_to_pillars(row: pd.Series) -> dict:
    """Map one IBM HR row to 21 individual Likert items (1-5 scale)."""

    def scale(val, lo, hi, out_lo=1, out_hi=5) -> float:
        """Linear rescale from [lo,hi] → [out_lo, out_hi], clipped."""
        if pd.isna(val):
            return 3.0
        return float(np.clip(out_lo + (val - lo) / (hi - lo) * (out_hi - out_lo), out_lo, out_hi))

    # Recrutement
    r1 = scale(row.get("JobSatisfaction", 3), 1, 4)
    r2 = scale(row.get("EnvironmentSatisfaction", 3), 1, 4)
    r3 = scale(row.get("RelationshipSatisfaction", 3), 1, 4)

    # Compétences
    c4 = scale(row.get("TrainingTimesLastYear", 2), 0, 6)
    c5 = scale(row.get("JobLevel", 2), 1, 5)
    c6 = scale(row.get("JobInvolvement", 3), 1, 4)

    # Performance
    p7 = scale(row.get("PerformanceRating", 3), 1, 4)
    p8 = scale(row.get("JobSatisfaction", 3), 1, 4)  # proxy
    p9 = scale(row.get("WorkLifeBalance", 3), 1, 4)

    # Rémunération
    monthly = row.get("MonthlyIncome", 5000)
    rem10 = scale(monthly, 1009, 19999)
    pct_hike = row.get("PercentSalaryHike", 12)
    rem11 = scale(pct_hike, 11, 25)
    rem12 = scale(row.get("StockOptionLevel", 1), 0, 3)

    # QVT
    overtime_pen = 0 if str(row.get("OverTime", "No")) == "Yes" else 2
    qvt13 = scale(row.get("WorkLifeBalance", 3), 1, 4)
    qvt14 = scale(row.get("EnvironmentSatisfaction", 3), 1, 4)
    qvt15 = max(1.0, scale(row.get("WorkLifeBalance", 3), 1, 4) - overtime_pen)

    # Droit
    d16 = scale(row.get("YearsAtCompany", 5), 0, 40)
    d17 = scale(row.get("YearsWithCurrManager", 3), 0, 20)
    d18 = scale(row.get("NumCompaniesWorked", 2), 0, 9, out_lo=5, out_hi=1)  # more companies → lower

    # Transverse
    t19 = scale(row.get("RelationshipSatisfaction", 3), 1, 4)
    t20 = scale(row.get("JobSatisfaction", 3), 1, 4)
    t21 = scale(row.get("TotalWorkingYears", 10), 0, 40)

    def avg(*vals): return float(np.mean(vals))

    return dict(
        recrutement_q1=r1, recrutement_q2=r2, recrutement_q3=r3, recrutement_q4=r1, recrutement_q5=r2, recrutement_q6=r3,
        competences_q7=c4, competences_q8=c5, competences_q9=c6, competences_q10=c4,
        performance_q11=p7, performance_q12=p8, performance_q13=p9, performance_q14=p7,
        remuneration_q15=rem10, remuneration_q16=rem11, remuneration_q17=rem12, remuneration_q18=rem10,
        qvt_q19=qvt13, qvt_q20=qvt14, qvt_q21=qvt15, qvt_q22=qvt13, qvt_q23=qvt14, qvt_q24=qvt15,
        droit_q25=d16, droit_q26=d17, droit_q27=d18, droit_q28=d16, droit_q29=d17,
        transverse_q30=t19, transverse_q31=t20, transverse_q32=t21, transverse_q33=t19,
        recrutement_avg=avg(r1, r2, r3),
        competences_avg=avg(c4, c5, c6),
        performance_avg=avg(p7, p8, p9),
        remuneration_avg=avg(rem10, rem11, rem12),
        qvt_avg=avg(qvt13, qvt14, qvt15),
        droit_avg=avg(d16, d17, d18),
        transverse_avg=avg(t19, t20, t21),
    )


# ---------------------------------------------------------------------------
# Main seed function
# ---------------------------------------------------------------------------

def run_seed() -> None:
    from sqlalchemy import text as _text
    with get_session() as session:
        # Temporarily disable FK enforcement so bulk inserts can use pseudo user_ids
        session.execute(_text("PRAGMA foreign_keys=OFF"))

        _seed_firms(session)
        firm_map = {f.name: f.firm_id for f in session.query(Firm).all()}
        firm_key_to_id = {
            "firm_a": firm_map.get("Alpina Services SA"),
            "firm_b": firm_map.get("Rh\u00f4ne Industrie S\u00e0rl"),
            "firm_c": firm_map.get("HR Valais (Platform)"),
        }

        _seed_users(session, firm_key_to_id)
        session.flush()

        _seed_ofs(session)
        _seed_ibm_responses(session, firm_key_to_id)
        _seed_synthetic_longitudinal(session, firm_key_to_id)

        session.execute(_text("PRAGMA foreign_keys=ON"))

    logger.info("Seed complete.")


# ---------------------------------------------------------------------------
# Individual seeders
# ---------------------------------------------------------------------------

def _seed_firms(session) -> None:
    existing = {f.name for f in session.query(Firm).all()}
    added = False
    for key, info in DEMO_FIRMS.items():
        if info["name"] not in existing:
            session.add(Firm(
                firm_id=str(uuid.uuid4()),
                name=info["name"],
                industry_domain=info["industry_domain"],
            ))
            added = True
    if added:
        session.flush()
    logger.info("Firms seeded.")


def _seed_users(session, firm_key_to_id: dict) -> None:
    existing = {u.username for u in session.query(User).all()}
    added = False
    for u in DEMO_USERS:
        if u["username"] not in existing:
            session.add(User(
                user_id=str(uuid.uuid4()),
                username=u["username"],
                role=u["role"],
                display_name=u["display_name"],
                hashed_password=hash_password(u["password"]),
                firm_id=firm_key_to_id.get(u["firm_key"]) if u["firm_key"] else None,
            ))
            added = True
    if added:
        session.flush()
    logger.info("Users seeded.")


def _seed_ofs(session) -> None:
    if session.query(OFSMacroData).count() > 0:
        logger.info("OFS data already present - skipping.")
        return

    wages_df, turnover_df, is_real = load_ofs_data(DATA_DIR)

    def safe_year(val):
        try:
            return int(val) if pd.notna(val) else None
        except (ValueError, TypeError):
            return None

    def safe_float(val):
        try:
            return float(val) if pd.notna(val) else None
        except (ValueError, TypeError):
            return None

    # Insert wages
    for _, row in wages_df.iterrows():
        session.add(OFSMacroData(
            source_file=row.get("source_file", "_201"),
            industry_domain=str(row.get("industry_domain", "Unknown")),
            professional_position=str(row.get("professional_position", "Unknown")),
            age_bracket=str(row.get("age_bracket", "Unknown")),
            gender=str(row.get("gender", "Unknown")),
            year=safe_year(row.get("year")),
            gross_monthly_median_wage=safe_float(row.get("gross_monthly_median_wage")),
            turnover_rate=None,
        ))

    # Insert turnover
    for _, row in turnover_df.iterrows():
        session.add(OFSMacroData(
            source_file=row.get("source_file", "_206"),
            industry_domain=str(row.get("industry_domain", "Unknown")),
            professional_position=str(row.get("professional_position", "Unknown")),
            age_bracket=str(row.get("age_bracket", "Unknown")),
            gender=str(row.get("gender", "Unknown")),
            year=safe_year(row.get("year")),
            gross_monthly_median_wage=None,
            turnover_rate=safe_float(row.get("turnover_rate")),
        ))

    session.flush()
    flag = "real" if is_real else "SYNTHETIC FALLBACK"
    logger.info(f"OFS data seeded ({flag}).")


def _seed_ibm_responses(session, firm_key_to_id: dict) -> None:
    """Load IBM HR Attrition CSV, map to 7 pillars, assign half to firm_a, half to firm_b."""
    ibm_path = DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not ibm_path.exists():
        logger.warning("IBM CSV not found — skipping.")
        return

    # If survey_responses already has rows from IBM, skip
    if session.query(SurveyResponse).count() > 10:
        logger.info("Survey responses already seeded — skipping IBM import.")
        return

    df = pd.read_csv(ibm_path)
    user_map = {u.username: u.user_id for u in session.query(User).all()}

    firm_a_id = firm_key_to_id["firm_a"]
    firm_b_id = firm_key_to_id["firm_b"]

    # Use pseudo-employee IDs (IBM rows don't map to real users — use a sentinel UUID)
    PSEUDO_EMPLOYEE = "00000000-0000-0000-0000-000000000001"

    STATES = ["Highly Engaged", "Content", "Passively Looking", "Resigned"]

    base_date = datetime(2022, 1, 1)
    batch = []
    for i, (_, row) in enumerate(df.iterrows()):
        firm_id = firm_a_id if i % 2 == 0 else firm_b_id
        pillars = _map_ibm_to_pillars(row)
        avg_score = np.mean([
            pillars["recrutement_avg"], pillars["competences_avg"],
            pillars["performance_avg"], pillars["qvt_avg"]
        ])
        # Derive engagement state from average score
        if avg_score >= 4.0:
            state = "Highly Engaged"
        elif avg_score >= 3.0:
            state = "Content"
        elif avg_score >= 2.0:
            state = "Passively Looking"
        else:
            state = "Resigned"

        attrition = str(row.get("Attrition", "No")) == "Yes"
        ts = base_date + timedelta(days=i // 5)  # batch into ~5 per day

        batch.append(SurveyResponse(
            response_id=str(uuid.uuid4()),
            user_id=PSEUDO_EMPLOYEE,
            firm_id=firm_id,
            timestamp=ts,
            month_index=None,
            age=int(row.get("Age", 30)),
            position="Upper Management" if str(row.get("JobLevel", 1)) in ["4", "5"] else "Execution",
            gender="Hommes" if str(row.get("Gender", "Male")) == "Male" else "Femmes",
            engagement_state=state,
            attrition_flag=attrition,
            **pillars,
        ))

    session.bulk_save_objects(batch)
    session.flush()
    logger.info(f"IBM HR responses seeded: {len(batch)} rows.")


def _seed_synthetic_longitudinal(session, firm_key_to_id: dict) -> None:
    """Load synthetic longitudinal CSV — time-series records per employee."""
    syn_path = DATA_DIR / "synthetic_longitudinal_hr.csv"
    if not syn_path.exists():
        logger.warning("Synthetic longitudinal CSV not found — skipping.")
        return

    # Only seed if fewer than 1000 responses
    existing = session.query(SurveyResponse).count()
    if existing > 1000:
        logger.info("Longitudinal data appear already seeded — skipping.")
        return

    df = pd.read_csv(syn_path)
    firm_a_id = firm_key_to_id["firm_a"]

    base_date = datetime(2022, 1, 1)
    PSEUDO_EMPLOYEE = "00000000-0000-0000-0000-000000000002"

    batch = []
    for _, row in df.iterrows():
        avg = (
            row["recrutement_avg"] + row["competences_avg"] + row["performance_avg"]
            + row["remuneration_avg"] + row["qvt_avg"] + row["droit_avg"]
            + row["transverse_avg"]
        ) / 7

        # Derive individual q scores from pillar averages (add ±0.3 noise)
        rng = np.random.default_rng(int(abs(row["recrutement_avg"] * 1000)) % 9999)

        def jitter(avg_val, n=3):
            vals = rng.normal(avg_val, 0.3, n)
            return [float(np.clip(v, 1, 5)) for v in vals]

        r = jitter(row["recrutement_avg"])
        c = jitter(row["competences_avg"])
        p = jitter(row["performance_avg"])
        rem = jitter(row["remuneration_avg"])
        q = jitter(row["qvt_avg"])
        d = jitter(row["droit_avg"])
        t = jitter(row["transverse_avg"])

        month_idx = int(row["month_index"])
        ts = base_date + timedelta(days=30 * (month_idx - 1))

        batch.append(SurveyResponse(
            response_id=str(uuid.uuid4()),
            user_id=PSEUDO_EMPLOYEE,
            firm_id=firm_a_id,
            timestamp=ts,
            month_index=month_idx,
            age=int(row.get("age", 30)),
            position=str(row.get("position", "Execution")),
            gender=str(row.get("gender", "Unknown")),
            engagement_state=str(row.get("state", "Content")),
            attrition_flag=str(row.get("state", "Content")) == "Resigned",
            recrutement_q1=r[0], recrutement_q2=r[1], recrutement_q3=r[2], recrutement_q4=r[0], recrutement_q5=r[1], recrutement_q6=r[2],
            competences_q7=c[0], competences_q8=c[1], competences_q9=c[2], competences_q10=c[0],
            performance_q11=p[0], performance_q12=p[1], performance_q13=p[2], performance_q14=p[0],
            remuneration_q15=rem[0], remuneration_q16=rem[1], remuneration_q17=rem[2], remuneration_q18=rem[0],
            qvt_q19=q[0], qvt_q20=q[1], qvt_q21=q[2], qvt_q22=q[0], qvt_q23=q[1], qvt_q24=q[2],
            droit_q25=d[0], droit_q26=d[1], droit_q27=d[2], droit_q28=d[0], droit_q29=d[1],
            transverse_q30=t[0], transverse_q31=t[1], transverse_q32=t[2], transverse_q33=t[0],
            recrutement_avg=float(row["recrutement_avg"]),
            competences_avg=float(row["competences_avg"]),
            performance_avg=float(row["performance_avg"]),
            remuneration_avg=float(row["remuneration_avg"]),
            qvt_avg=float(row["qvt_avg"]),
            droit_avg=float(row["droit_avg"]),
            transverse_avg=float(row["transverse_avg"]),
        ))

        if len(batch) >= 2000:
            session.bulk_save_objects(batch)
            session.flush()
            batch = []

    if batch:
        session.bulk_save_objects(batch)
        session.flush()

    logger.info(f"Synthetic longitudinal data seeded: {len(df)} rows.")
