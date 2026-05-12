"""
pages/2_Dashboard_Internal.py
Dashboard 1 — k-anonymized internal survey results for the HR Manager's firm, and global view for Admin.
Access: hr_manager, admin.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.auth import require_role, get_current_user
from utils.pdf_generator import SURVEY_STRUCTURE, URL_MAPPING
from db.database import get_session
from db.models import SurveyResponse, Firm

require_role("hr_manager", "admin")
user = get_current_user()
is_admin = user["role"] == "admin"

# ---------------------------------------------------------------------------
# Data Loading Constants
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
def _load_all_data() -> pd.DataFrame:
    with get_session() as session:
        entities = [SurveyResponse.firm_id] + [getattr(SurveyResponse, p) for p in PILLARS] + [getattr(SurveyResponse, q) for q in Q_COLS]
        rows = session.query(SurveyResponse).with_entities(*entities).all()
        # Fetch firm names to join
        firms = session.query(Firm).all()
        firm_map = {f.firm_id: f.name for f in firms}
        
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows, columns=["firm_id"] + PILLARS + Q_COLS)
    df["firm_name"] = df["firm_id"].map(firm_map).fillna("Inconnue")
    return df

df_all = _load_all_data()

if df_all.empty:
    st.warning("⚠️ Aucune donnée de sondage disponible dans la base de données.")
    st.stop()

# Replace 0 (Je ne sais pas) with NaN globally
df_all_filtered = df_all.replace(0, np.nan)

# ===========================================================================
# ADMIN DASHBOARD
# ===========================================================================
if is_admin:
    st.title("📊 Tableau de bord global (Admin)")
    st.caption("Vue comparative de toutes les entreprises")

    # 1. Multi-firm Spider Graph
    fig_radar = go.Figure()
    for firm_name, grp in df_all_filtered.groupby("firm_name"):
        means = grp[PILLARS].mean().values
        fig_radar.add_trace(go.Scatterpolar(
            r=list(means) + [means[0]],
            theta=PILLAR_LABELS + [PILLAR_LABELS[0]],
            fill="none",
            name=firm_name,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[1, 4])),
        title="Scores moyens par dimension et par entreprise",
        template="plotly_dark",
        height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.divider()

    # 2. Boxplot for 33 Questions
    st.subheader("Distribution des scores par question (Boxplot)")
    st.markdown("Répartition des notes pour les 33 questions individuelles.")
    df_q = df_all_filtered[["firm_name"] + Q_COLS].melt(id_vars=["firm_name"], var_name="Question", value_name="Score")
    # Clean question labels (e.g., 'qvt_q19' -> 'Q19')
    df_q["Question"] = df_q["Question"].apply(lambda x: x.split("_")[-1].upper())
    
    fig_box = px.box(
        df_q.dropna(subset=["Score"]), 
        x="Question", 
        y="Score", 
        color="firm_name",
        title="Boxplot : 33 Questions", 
        template="plotly_dark", 
        points="all"
    )
    fig_box.update_yaxes(range=[0.8, 4.2])
    st.plotly_chart(fig_box, use_container_width=True)
    st.divider()

    # 3. Violin Plot for 7 Pillars
    st.subheader("Distribution des scores par pilier (Violin Plot)")
    st.markdown("Densité de probabilité et distribution des 7 dimensions principales.")
    df_p = df_all_filtered[["firm_name"] + PILLARS].melt(id_vars=["firm_name"], var_name="Dimension", value_name="Score")
    dim_map = dict(zip(PILLARS, PILLAR_LABELS))
    df_p["Dimension"] = df_p["Dimension"].map(dim_map)
    
    fig_violin = px.violin(
        df_p.dropna(subset=["Score"]), 
        x="Dimension", 
        y="Score", 
        color="firm_name",
        title="Violin Plot : 7 Piliers", 
        template="plotly_dark", 
        box=True, 
        points="all"
    )
    fig_violin.update_yaxes(range=[0.8, 4.2])
    st.plotly_chart(fig_violin, use_container_width=True)

    st.stop()


# ===========================================================================
# HR MANAGER DASHBOARD
# ===========================================================================
st.title("📊 Tableau de bord interne")
st.caption(f"Données agrégées pour **{user['firm_name']}** et Benchmark global")

df_firm = df_all_filtered[df_all_filtered["firm_id"] == user["firm_id"]]
df_other = df_all_filtered[df_all_filtered["firm_id"] != user["firm_id"]]

if df_firm.empty:
    st.warning("⚠️ Aucune donnée disponible pour votre entreprise.")
    st.stop()

# ---------------------------------------------------------------------------
# Radar chart (Spider Chart) with Benchmark
# ---------------------------------------------------------------------------
fig_radar = go.Figure()

# Plot Current Firm
pillar_means_firm = df_firm[PILLARS].mean().values
fig_radar.add_trace(go.Scatterpolar(
    r=list(pillar_means_firm) + [pillar_means_firm[0]],
    theta=PILLAR_LABELS + [PILLAR_LABELS[0]],
    fill="toself",
    name=user["firm_name"],
    line_color="#4F8EF7",
    fillcolor="rgba(79,142,247,0.2)",
))

# Plot Benchmark (Average of all OTHER firms)
if not df_other.empty:
    pillar_means_other = df_other[PILLARS].mean().values
    fig_radar.add_trace(go.Scatterpolar(
        r=list(pillar_means_other) + [pillar_means_other[0]],
        theta=PILLAR_LABELS + [PILLAR_LABELS[0]],
        fill="none",
        name="Benchmark (Autres entreprises)",
        line=dict(color="#A0AEC0", dash="dash", width=2),
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[1, 4])),
    title="Scores de l'entreprise vs Benchmark global",
    template="plotly_dark",
    height=450,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------------------------
# Diagnostic Engine
# ---------------------------------------------------------------------------
st.header("Moteur de diagnostic")

def get_base_url(dimension_name):
    import re
    import unicodedata
    name = dimension_name.lower()
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    name = re.sub(r'[^a-z0-9\s-]', '', name)
    name = re.sub(r'[\s]+', '-', name.strip())
    if "qvt" in name:
        name = "qualite-de-vie-au-travail"
    elif "evaluation" in name:
        name = "evaluation-et-performance"
    return f"https://www.hr-valais.ch/fiches-rh-pme/francais/{name}"

for i, pillar_name in enumerate(PILLAR_LABELS):
    score = pillar_means_firm[i]
    if pd.notna(score) and score < 3.0:
        base_url = get_base_url(pillar_name)
        st.warning(f"**Attention**: La dimension **{pillar_name}** a un score moyen faible ({score:.2f}). [Consulter le guide de base]({base_url})")

st.subheader("Feedback spécifique")

q_idx = 0
for i, (dimension_name, questions) in enumerate(SURVEY_STRUCTURE):
    for q_text in questions:
        col_name = Q_COLS[q_idx]
        q_avg = df_firm[col_name].mean()
        
        if pd.notna(q_avg) and q_avg < 3.0:
            fiche_url = URL_MAPPING.get(q_text, "#")
            st.error(f"**Score critique ({q_avg:.2f})** : {q_text}\n\n👉 **Fiche pratique recommandée** : {fiche_url}")
        
        q_idx += 1

st.divider()
st.caption("Fin du rapport de diagnostic.")
