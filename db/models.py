"""
db/models.py
SQLAlchemy ORM models for HR Valais.
Arrays are stored as JSON strings (SQLite has no native array type).
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

    firm = relationship("Firm", back_populates="users")
    survey_responses = relationship("SurveyResponse", back_populates="user")


class SurveyResponse(Base):
    __tablename__ = "survey_responses"

    response_id = Column(String, primary_key=True, default=_new_uuid)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    firm_id = Column(String, ForeignKey("firms.firm_id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    month_index = Column(Integer, nullable=True)   # for synthetic longitudinal data

    # Pillar scores — stored as comma-separated "q1,q2,q3" strings
    # Individual questions (1-5 Likert)
    recrutement_q1 = Column(Float, nullable=True)
    recrutement_q2 = Column(Float, nullable=True)
    recrutement_q3 = Column(Float, nullable=True)

    competences_q4 = Column(Float, nullable=True)
    competences_q5 = Column(Float, nullable=True)
    competences_q6 = Column(Float, nullable=True)

    performance_q7 = Column(Float, nullable=True)
    performance_q8 = Column(Float, nullable=True)
    performance_q9 = Column(Float, nullable=True)

    remuneration_q10 = Column(Float, nullable=True)
    remuneration_q11 = Column(Float, nullable=True)
    remuneration_q12 = Column(Float, nullable=True)

    qvt_q13 = Column(Float, nullable=True)
    qvt_q14 = Column(Float, nullable=True)
    qvt_q15 = Column(Float, nullable=True)

    droit_q16 = Column(Float, nullable=True)
    droit_q17 = Column(Float, nullable=True)
    droit_q18 = Column(Float, nullable=True)

    transverse_q19 = Column(Float, nullable=True)
    transverse_q20 = Column(Float, nullable=True)
    transverse_q21 = Column(Float, nullable=True)

    # Derived pillar averages (convenience columns for fast querying)
    recrutement_avg = Column(Float, nullable=True)
    competences_avg = Column(Float, nullable=True)
    performance_avg = Column(Float, nullable=True)
    remuneration_avg = Column(Float, nullable=True)
    qvt_avg = Column(Float, nullable=True)
    droit_avg = Column(Float, nullable=True)
    transverse_avg = Column(Float, nullable=True)

    # Employee metadata mirrored for time-series
    age = Column(Integer, nullable=True)
    position = Column(String, nullable=True)   # "Execution" | "Upper Management"
    gender = Column(String, nullable=True)
    engagement_state = Column(String, nullable=True)  # Markov label
    attrition_flag = Column(Boolean, nullable=True)

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
