"""
pages/2_Dashboard_Internal.py
Dashboard 1 — k-anonymized internal survey results for the HR Manager's firm.
Access: hr_manager only.
"""

import sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.auth import require_role, get_current_user
from utils.kanon import kanonymize
from db.database import get_session
from db.models import SurveyResponse

require_role("hr_manager")
user = get_current_user()

st.title("📊 Tableau de bord interne")
st.caption(f"Données agrégées pour **{user['firm_name']}** — k-anonymisées (k ≥ 5)")

# ---------------------------------------------------------------------------
# Load data for this firm
# ---------------------------------------------------------------------------
PILLARS = ["recrutement_avg", "competences_avg", "performance_avg",
           "remuneration_avg", "qvt_avg", "droit_avg", "transverse_avg"]
PILLAR_LABELS = ["Recrutement", "Compétences", "Performance",
                 "Rémunération", "QVT", "Droit", "Transverse"]

@st.cache_data(ttl=60, show_spinner="Chargement des données…")
def _load_firm_data(firm_id: str) -> pd.DataFrame:
    with get_session() as session:
        rows = (
            session.query(SurveyResponse)
            .filter_by(firm_id=firm_id)
            .with_entities(
                SurveyResponse.timestamp,
                SurveyResponse.month_index,
                SurveyResponse.age,
                SurveyResponse.position,
                SurveyResponse.gender,
                SurveyResponse.engagement_state,
                *[getattr(SurveyResponse, p) for p in PILLARS],
            )
            .all()
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "timestamp", "month_index", "age", "position", "gender", "engagement_state"
    ] + PILLARS)

df = _load_firm_data(user["firm_id"])

if df.empty:
    st.warning("⚠️ Aucune donnée disponible pour cette entreprise.")
    st.stop()

total_responses = len(df)
K = 5

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Réponses totales", total_responses)
col2.metric("Score moyen global", f"{df[PILLARS].mean().mean():.2f} / 5")
attrition_pct = (df["engagement_state"] == "Resigned").mean() * 100
col3.metric("Taux Resigned", f"{attrition_pct:.1f}%")
highly_engaged_pct = (df["engagement_state"] == "Highly Engaged").mean() * 100
col4.metric("Très engagés", f"{highly_engaged_pct:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------
pillar_means = df[PILLARS].mean().values

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
    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    title="Scores moyens par pilier (k-anonymisé)",
    template="plotly_dark",
    height=420,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------------------------
# Pillar trend over time
# ---------------------------------------------------------------------------
st.subheader("📈 Évolution temporelle par pilier")

if "month_index" in df.columns and df["month_index"].notna().any():
    trend_df = (
        df.dropna(subset=["month_index"])
        .groupby("month_index")[PILLARS]
        .mean()
        .reset_index()
    )
    # Only show months with ≥ k responses (k-anon)
    counts = df.dropna(subset=["month_index"]).groupby("month_index").size()
    valid_months = counts[counts >= K].index
    trend_df = trend_df[trend_df["month_index"].isin(valid_months)]

    if trend_df.empty:
        st.info(f"Pas assez de données par mois (k={K}) pour afficher la tendance.")
    else:
        fig_trend = px.line(
            trend_df.melt(id_vars="month_index", value_vars=PILLARS,
                          var_name="Pilier", value_name="Score moyen"),
            x="month_index", y="Score moyen", color="Pilier",
            labels={"month_index": "Mois"},
            title="Tendance temporelle des scores (groupes ≥ 5 réponses)",
            template="plotly_dark",
        )
        # Rename legend
        label_map = dict(zip(PILLARS, PILLAR_LABELS))
        fig_trend.for_each_trace(lambda t: t.update(name=label_map.get(t.name, t.name)))
        st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("Données temporelles non disponibles pour vos réponses directes.")

# ---------------------------------------------------------------------------
# Breakdown by position (k-anon)
# ---------------------------------------------------------------------------
st.subheader("👥 Scores par position professionnelle (k-anonymisés)")
if df["position"].notna().any():
    anon_pos = kanonymize(
        df.dropna(subset=["position"]),
        group_cols=["position"],
        value_cols=PILLARS,
        k=K,
    )
    if anon_pos.empty:
        st.info(f"Groupes trop petits (N < {K}) — données supprimées pour confidentialité.")
    else:
        anon_pos = anon_pos.rename(columns=dict(zip(PILLARS, PILLAR_LABELS)))
        st.dataframe(
            anon_pos[["position", "_count"] + PILLAR_LABELS].rename(
                columns={"_count": "N (réponses)"}
            ).round(2),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Breakdown by engagement state (k-anon)
# ---------------------------------------------------------------------------
st.subheader("🔗 Répartition par état d'engagement")
if df["engagement_state"].notna().any():
    state_counts = df["engagement_state"].value_counts().reset_index()
    state_counts.columns = ["État", "Nombre"]
    # Suppress states with N < k
    state_counts = state_counts[state_counts["Nombre"] >= K]
    if state_counts.empty:
        st.info("Données insuffisantes (k-anonymisation).")
    else:
        fig_bar = px.bar(
            state_counts, x="État", y="Nombre",
            color="État",
            color_discrete_map={
                "Highly Engaged": "#22C55E",
                "Content": "#4F8EF7",
                "Passively Looking": "#F59E0B",
                "Resigned": "#EF4444",
            },
            template="plotly_dark",
            title="Répartition des états d'engagement (groupes ≥ 5)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
st.divider()
anon_download = kanonymize(df, group_cols=["month_index", "position", "gender"],
                            value_cols=PILLARS, k=K)
st.download_button(
    "⬇️ Télécharger les données agrégées (CSV)",
    data=anon_download.to_csv(index=False).encode("utf-8"),
    file_name=f"hrvalais_aggregated_{user['firm_id'][:8]}.csv",
    mime="text/csv",
)
