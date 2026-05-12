"""
db/models.py
SQLAlchemy ORM models for HR Valais.
Arrays are stored as JSON strings (SQLite has no native array type).

Survey scale (v2): 0 = Je ne sais pas | 1 = Faible | 2 = Partiel | 3 = Avancé | 4 = Optimal
32 questions across 7 dimensions.
"""

import json
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Firm(Base):
    __tablename__ = "firms"

    firm_id = Column(String, primary_key=True, default=_new_uuid)
    name = Column(String, nullable=False)
    industry_domain = Column(String, nullable=True)

    users = relationship("User", back_populates="firm")
    survey_responses = relationship("SurveyResponse", back_populates="firm")
    monthly_uploads = relationship("MonthlyUpload", back_populates="firm")


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, default=_new_uuid)
    firm_id = Column(String, ForeignKey("firms.firm_id"), nullable=True)
    username = Column(String, nullable=False, unique=True)
    role = Column(String, nullable=False)          # 'employee' | 'hr_manager' | 'admin'
    hashed_password = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    max_surveys_per_year = Column(Integer, default=1)

    firm = relationship("Firm", back_populates="users")
    survey_responses = relationship("SurveyResponse", back_populates="user")


class SurveyResponse(Base):
    """
    One row per survey submission.

    Scale (v2): 0 = Je ne sais pas | 1–4 Likert
    Pillar averages exclude 0 (Je ne sais pas) values to avoid distortion.

    Dimensions:
      Recrutement        : q1–q6   (6 questions)
      Compétences        : q7–q10  (4 questions)
      Performance        : q11–q14 (4 questions)
      Rémunération       : q15–q18 (4 questions)
      QVT                : q19–q24 (6 questions)
      Droit du travail   : q25–q29 (5 questions)
      Transverses        : q30–q33 (4 questions)
    """
    __tablename__ = "survey_responses"

    response_id = Column(String, primary_key=True, default=_new_uuid)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    firm_id = Column(String, ForeignKey("firms.firm_id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    month_index = Column(Integer, nullable=True)   # for synthetic longitudinal data

    # ── Pilier 1 : Recrutement (q1–q6) ──────────────────────────────────────
    recrutement_q1 = Column(Float, nullable=True)   # annonce / missions
    recrutement_q2 = Column(Float, nullable=True)   # canaux de diffusion
    recrutement_q3 = Column(Float, nullable=True)   # critères de sélection
    recrutement_q4 = Column(Float, nullable=True)   # structure entretiens
    recrutement_q5 = Column(Float, nullable=True)   # grille évaluation
    recrutement_q6 = Column(Float, nullable=True)   # onboarding

    # ── Pilier 2 : Gestion des compétences (q7–q10) ─────────────────────────
    competences_q7  = Column(Float, nullable=True)  # descriptif de poste
    competences_q8  = Column(Float, nullable=True)  # profil de compétences
    competences_q9  = Column(Float, nullable=True)  # catalogue/plan formation
    competences_q10 = Column(Float, nullable=True)  # évolution compétences

    # ── Pilier 3 : Évaluation & Performance (q11–q14) ───────────────────────
    performance_q11 = Column(Float, nullable=True)  # évaluation formelle annuelle
    performance_q12 = Column(Float, nullable=True)  # objectifs annuels
    performance_q13 = Column(Float, nullable=True)  # feedback structuré
    performance_q14 = Column(Float, nullable=True)  # reconnaissance

    # ── Pilier 4 : Rémunération (q15–q18) ───────────────────────────────────
    remuneration_q15 = Column(Float, nullable=True)  # politique salariale
    remuneration_q16 = Column(Float, nullable=True)  # compétitivité salariale
    remuneration_q17 = Column(Float, nullable=True)  # fiches de salaire
    remuneration_q18 = Column(Float, nullable=True)  # avantages

    # ── Pilier 5 : QVT (q19–q24) ────────────────────────────────────────────
    qvt_q19 = Column(Float, nullable=True)  # risques santé/sécurité
    qvt_q20 = Column(Float, nullable=True)  # mesures protection
    qvt_q21 = Column(Float, nullable=True)  # flexibilité horaires/lieu
    qvt_q22 = Column(Float, nullable=True)  # actions si pas de flexibilité
    qvt_q23 = Column(Float, nullable=True)  # écoute collaborateurs
    qvt_q24 = Column(Float, nullable=True)  # fidélisation long terme

    # ── Pilier 6 : Droit du travail (q25–q29) ───────────────────────────────
    droit_q25 = Column(Float, nullable=True)  # éléments essentiels contrat
    droit_q26 = Column(Float, nullable=True)  # droits et obligations
    droit_q27 = Column(Float, nullable=True)  # réalité du poste vs contrat
    droit_q28 = Column(Float, nullable=True)  # adaptation conditions contrat
    droit_q29 = Column(Float, nullable=True)  # fin de rapport de travail

    # ── Pilier 7 : Thématiques transverses (q30–q33) ────────────────────────
    transverse_q30 = Column(Float, nullable=True)  # outils numériques RH
    transverse_q31 = Column(Float, nullable=True)  # IA / éthique / données
    transverse_q32 = Column(Float, nullable=True)  # marque employeur
    transverse_q33 = Column(Float, nullable=True)  # protection des données

    # ── Derived pillar averages (NULL for 0-only responses) ─────────────────
    recrutement_avg  = Column(Float, nullable=True)
    competences_avg  = Column(Float, nullable=True)
    performance_avg  = Column(Float, nullable=True)
    remuneration_avg = Column(Float, nullable=True)
    qvt_avg          = Column(Float, nullable=True)
    droit_avg        = Column(Float, nullable=True)
    transverse_avg   = Column(Float, nullable=True)

    # ── Employee metadata mirrored for time-series ───────────────────────────
    age              = Column(Integer, nullable=True)
    position         = Column(String, nullable=True)   # "Execution" | "Upper Management"
    gender           = Column(String, nullable=True)
    engagement_state = Column(String, nullable=True)   # Markov label
    attrition_flag   = Column(Boolean, nullable=True)

    free_text_feedback = Column(Text, nullable=True)

    user = relationship("User", back_populates="survey_responses")
    firm = relationship("Firm", back_populates="survey_responses")


class OFSMacroData(Base):
    __tablename__ = "ofs_macro_data"

    ofs_id = Column(Integer, primary_key=True, autoincrement=True)
    source_file = Column(String, nullable=True)    # '_201' | '_206'
    industry_domain = Column(String, nullable=True)
    professional_position = Column(String, nullable=True)
    age_bracket = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    gross_monthly_median_wage = Column(Float, nullable=True)
    turnover_rate = Column(Float, nullable=True)


class MonthlyUpload(Base):
    __tablename__ = "monthly_uploads"

    upload_id = Column(String, primary_key=True, default=_new_uuid)
    firm_id = Column(String, ForeignKey("firms.firm_id"), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    filename = Column(String, nullable=True)
    row_count = Column(Integer, nullable=True)
    raw_csv = Column(Text, nullable=True)   # Store raw CSV as text for auditability

    firm = relationship("Firm", back_populates="monthly_uploads")

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id = Column(String, primary_key=True, default=_new_uuid)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=True)
    firm_id = Column(String, ForeignKey("firms.firm_id"), nullable=True)
    action_type = Column(String, nullable=False)
    action_value = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

