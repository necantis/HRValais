"""
pages/2_Dashboard_Internal.py
Dashboard 1 — k-anonymized internal survey results for the HR Manager's firm.
Access: hr_manager only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.auth import require_role, get_current_user
from utils.pdf_generator import SURVEY_STRUCTURE, URL_MAPPING
from db.database import get_session
from db.models import SurveyResponse

require_role("hr_manager")
user = get_current_user()

st.title("📊 Tableau de bord interne")
st.caption(f"Données agrégées pour **{user['firm_name']}**")

# ---------------------------------------------------------------------------
# Load data for this firm
# ---------------------------------------------------------------------------
PILLARS = [
    "recrutement_avg", "competences_avg", "performance_avg",
    "remuneration_avg", "qvt_avg", "droit_avg", "transverse_avg"
]
PILLAR_LABELS = [
    "Recrutement", "Gestion des compétences", "Évaluation & Performance",
    "Rémunération", "Qualité de vie au travail (QVT)", "Droit du travail", "Thématiques transverses"
]

# Get all individual question columns
Q_COLS = [
    "recrutement_q1", "recrutement_q2", "recrutement_q3", "recrutement_q4", "recrutement_q5", "recrutement_q6",
    "competences_q7", "competences_q8", "competences_q9", "competences_q10",
    "performance_q11", "performance_q12", "performance_q13", "performance_q14",
    "remuneration_q15", "remuneration_q16", "remuneration_q17", "remuneration_q18",
    "qvt_q19", "qvt_q20", "qvt_q21", "qvt_q22", "qvt_q23", "qvt_q24",
    "droit_q25", "droit_q26", "droit_q27", "droit_q28", "droit_q29",
    "transverse_q30", "transverse_q31", "transverse_q32", "transverse_q33"
]

@st.cache_data(ttl=60, show_spinner="Chargement des données…")
def _load_firm_data(firm_id: str) -> pd.DataFrame:
    with get_session() as session:
        entities = [getattr(SurveyResponse, p) for p in PILLARS] + [getattr(SurveyResponse, q) for q in Q_COLS]
        rows = (
            session.query(SurveyResponse)
            .filter_by(firm_id=firm_id)
            .with_entities(*entities)
            .all()
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=PILLARS + Q_COLS)

df = _load_firm_data(user["firm_id"])

if df.empty:
    st.warning("⚠️ Aucune donnée disponible pour cette entreprise.")
    st.stop()

# Filter out 0s (Je ne sais pas) by replacing with NaN
df_filtered = df.replace(0, np.nan)

# ---------------------------------------------------------------------------
# Radar chart (Spider Chart)
# ---------------------------------------------------------------------------
pillar_means = df_filtered[PILLARS].mean().values

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=list(pillar_means) + [pillar_means[0]],
    theta=PILLAR_LABELS + [PILLAR_LABELS[0]],
    fill="toself",
    name=user["firm_name"],
    line_color="#4F8EF7",
    fillcolor="rgba(79,142,247,0.2)",
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[1, 4])),
    title="Scores moyens par dimension",
    template="plotly_dark",
    height=420,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------------------------
# Diagnostic Engine
# ---------------------------------------------------------------------------
st.header("Moteur de diagnostic")

# Create a mapping for dimension base URLs
def get_base_url(dimension_name):
    # Example transformation: 'Évaluation & Performance' -> 'evaluation-performance'
    import re
    import unicodedata
    name = dimension_name.lower()
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    name = re.sub(r'[^a-z0-9\s-]', '', name)
    name = re.sub(r'[\s]+', '-', name.strip())
    # Handle specific overrides if needed based on URL structure
    if "qvt" in name:
        name = "qualite-de-vie-au-travail"
    elif "evaluation" in name:
        name = "evaluation-et-performance"
    return f"https://www.hr-valais.ch/fiches-rh-pme/francais/{name}"

for i, pillar_name in enumerate(PILLAR_LABELS):
    score = pillar_means[i]
    if pd.notna(score) and score < 3.0:
        base_url = get_base_url(pillar_name)
        st.warning(f"**Attention**: La dimension **{pillar_name}** a un score moyen faible ({score:.2f}). [Consulter le guide de base]({base_url})")

st.subheader("Feedback spécifique")

q_idx = 0
for i, (dimension_name, questions) in enumerate(SURVEY_STRUCTURE):
    for q_text in questions:
        col_name = Q_COLS[q_idx]
        q_avg = df_filtered[col_name].mean()
        
        if pd.notna(q_avg) and q_avg < 3.0:
            fiche_url = URL_MAPPING.get(q_text, "#")
            st.error(f"**Score critique ({q_avg:.2f})** : {q_text}\n\n👉 **Fiche pratique recommandée** : {fiche_url}")
        
        q_idx += 1

st.divider()
st.caption("Fin du rapport de diagnostic.")
