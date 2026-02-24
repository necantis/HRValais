"""
pages/3_Dashboard_Benchmarking.py
Dashboard 2 — OFS macro-data benchmarking: gross monthly wages by position, age, gender.
Access: hr_manager only.
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
from db.database import get_session
from db.models import OFSMacroData, SurveyResponse

require_role("hr_manager")
user = get_current_user()

st.title("📈 Benchmarking OFS — Salaires et mobilité")
st.caption("Données: Office Fédéral de la Statistique (OFS), Suisse")

# ---------------------------------------------------------------------------
# Load OFS data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Chargement des données OFS…")
def _load_ofs() -> pd.DataFrame:
    with get_session() as session:
        rows = session.query(OFSMacroData).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "source_file": r.source_file,
        "industry_domain": r.industry_domain,
        "professional_position": r.professional_position,
        "age_bracket": r.age_bracket,
        "gender": r.gender,
        "year": r.year,
        "gross_monthly_median_wage": r.gross_monthly_median_wage,
        "turnover_rate": r.turnover_rate,
    } for r in rows])

ofs_df = _load_ofs()

# Detect if synthetic
is_synthetic = False
if not ofs_df.empty and ofs_df["source_file"].str.contains("synthetic").any():
    is_synthetic = True
    st.warning(
        "⚠️ **Données synthétiques** : Le parser des fichiers OFS .px n'a pas pu extraire "
        "les données réelles. Les graphiques ci-dessous utilisent des estimations calibrées sur l'OFS.",
        icon="⚠️",
    )

if ofs_df.empty:
    st.error("Aucune donnée OFS disponible.")
    st.stop()

wages_df = ofs_df[ofs_df["gross_monthly_median_wage"].notna()].copy()
turnover_df = ofs_df[ofs_df["turnover_rate"].notna()].copy()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["💼 Par position", "🎂 Par tranche d'âge", "⚧️ Par genre"])

with tab1:
    st.subheader("Salaire médian brut mensuel par position professionnelle")
    if not wages_df.empty and "professional_position" in wages_df.columns:
        pos_df = wages_df.groupby("professional_position")["gross_monthly_median_wage"].median().reset_index()
        fig = px.bar(
            pos_df,
            x="professional_position", y="gross_monthly_median_wage",
            color="professional_position",
            labels={"professional_position": "Position", "gross_monthly_median_wage": "Salaire médian (CHF/mois)"},
            template="plotly_dark",
            title="Salaire brut mensuel médian OFS — par position",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Données de position non disponibles.")

with tab2:
    st.subheader("Salaire médian brut mensuel par tranche d'âge")
    if not wages_df.empty and "age_bracket" in wages_df.columns:
        age_df = wages_df.groupby(["age_bracket", "professional_position"])["gross_monthly_median_wage"].median().reset_index()
        fig2 = px.bar(
            age_df,
            x="age_bracket", y="gross_monthly_median_wage",
            color="professional_position",
            barmode="group",
            labels={"age_bracket": "Tranche d'âge", "gross_monthly_median_wage": "Salaire médian (CHF/mois)"},
            template="plotly_dark",
            title="Salaire médian par âge et position",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Données d'âge non disponibles.")

with tab3:
    st.subheader("Salaire médian brut mensuel par genre")
    if not wages_df.empty and "gender" in wages_df.columns:
        gen_df = wages_df.groupby(["gender", "professional_position"])["gross_monthly_median_wage"].median().reset_index()
        fig3 = px.bar(
            gen_df,
            x="gender", y="gross_monthly_median_wage",
            color="professional_position",
            barmode="group",
            labels={"gender": "Genre", "gross_monthly_median_wage": "Salaire médian (CHF/mois)"},
            template="plotly_dark",
            title="Écart salarial par genre",
        )
        # Gender pay gap annotation
        if "Hommes" in gen_df["gender"].values and "Femmes" in gen_df["gender"].values:
            m_avg = gen_df[gen_df["gender"] == "Hommes"]["gross_monthly_median_wage"].mean()
            f_avg = gen_df[gen_df["gender"] == "Femmes"]["gross_monthly_median_wage"].mean()
            gap = (m_avg - f_avg) / m_avg * 100
            st.caption(f"📊 Écart salarial moyen hommes/femmes : **{gap:.1f}%**")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Données de genre non disponibles.")

# ---------------------------------------------------------------------------
# Turnover / Mobilité
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔄 Taux de turnover OFS (données de référence pour Dashboard 4)")

if not turnover_df.empty:
    turn_pos = turnover_df.groupby("professional_position")["turnover_rate"].mean().reset_index()
    fig_t = px.bar(
        turn_pos,
        x="professional_position", y="turnover_rate",
        color="professional_position",
        labels={"professional_position": "Position", "turnover_rate": "Taux de turnover moyen"},
        template="plotly_dark",
        title="Taux de turnover OFS par position (classe de référence pour Markov Chain)",
    )
    fig_t.update_traces(texttemplate="%{y:.1%}", textposition="outside")
    fig_t.update_yaxes(tickformat=".0%")
    fig_t.update_layout(showlegend=False)
    st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("""
<div class="caveat-box">
⚠️ <strong>Note méthodologique</strong> : Ces taux servent de <em>probabilités de référence</em>
pour la chaîne de Markov et le modèle bayésien (Dashboard 4). Ils ne sont pas directement
transposables à votre entreprise sans ajustement contextuel.
</div>
""", unsafe_allow_html=True)
else:
    st.info("Données de turnover non disponibles.")

# ---------------------------------------------------------------------------
# Raw OFS table (optional)
# ---------------------------------------------------------------------------
with st.expander("🗃️ Données OFS brutes"):
    st.dataframe(ofs_df.round(2), use_container_width=True)
