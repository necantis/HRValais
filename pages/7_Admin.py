"""
pages/7_Admin.py
Admin Dashboard — global cross-tenant aggregated view.
Access: admin only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from utils.auth import require_role, get_current_user
from db.database import get_session, DB_PATH
from db.models import Firm, User, SurveyResponse, OFSMacroData, MonthlyUpload

require_role("admin")
user = get_current_user()

st.title("⚙️ Administration HR Valais")
st.caption(f"Vue globale agrégée — {datetime.now().strftime('%d %B %Y %H:%M')}")

# ---------------------------------------------------------------------------
# DB Health Metrics
# ---------------------------------------------------------------------------
st.subheader("🗄️ Santé de la base de données")

@st.cache_data(ttl=30, show_spinner="Lecture de la base de données…")
def _load_counts() -> dict:
    with get_session() as session:
        return {
            "firms": session.query(Firm).count(),
            "users": session.query(User).count(),
            "responses": session.query(SurveyResponse).count(),
            "ofs_rows": session.query(OFSMacroData).count(),
            "uploads": session.query(MonthlyUpload).count(),
        }

counts = _load_counts()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Entreprises", counts["firms"])
c2.metric("Utilisateurs", counts["users"])
c3.metric("Réponses sondage", counts["responses"])
c4.metric("Lignes OFS", counts["ofs_rows"])
c5.metric("Imports CSV", counts["uploads"])

db_size_mb = DB_PATH.stat().st_size / (1024 * 1024) if DB_PATH.exists() else 0
st.caption(f"📦 Taille DB : **{db_size_mb:.2f} MB** · Chemin : `{DB_PATH}`")

# ---------------------------------------------------------------------------
# Falsifiability flags across all firms
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔍 Statut des modèles par entreprise (falsifiabilité)")

PILLARS = ["recrutement_avg", "competences_avg", "performance_avg",
           "remuneration_avg", "qvt_avg", "droit_avg", "transverse_avg"]

@st.cache_data(ttl=60)
def _load_cross_firm() -> pd.DataFrame:
    with get_session() as session:
        firms = session.query(Firm).all()
        rows = []
        for firm in firms:
            responses = (
                session.query(SurveyResponse)
                .filter_by(firm_id=firm.firm_id)
                .with_entities(
                    SurveyResponse.engagement_state,
                    SurveyResponse.attrition_flag,
                    SurveyResponse.month_index,
                    *[getattr(SurveyResponse, p) for p in PILLARS],
                )
                .all()
            )
            if not responses:
                continue
            df_firm = pd.DataFrame(responses,
                columns=["engagement_state", "attrition_flag", "month_index"] + PILLARS)

            attrition_rate = df_firm["attrition_flag"].mean() if df_firm["attrition_flag"].notna().any() else np.nan
            resigned_rate = (df_firm["engagement_state"] == "Resigned").mean()
            avg_score = df_firm[PILLARS].mean().mean()
            n = len(df_firm)

            # Anonymise firm name with hash (show only first 6 chars of SHA-256)
            firm_hash = hashlib.sha256(firm.firm_id.encode()).hexdigest()[:6].upper()
            rows.append({
                "Entreprise (hash)": f"FIRM-{firm_hash}",
                "N réponses": n,
                "Score global moyen": round(avg_score, 2),
                "Taux attrition": f"{attrition_rate:.1%}" if pd.notna(attrition_rate) else "N/A",
                "Taux Resigned": f"{resigned_rate:.1%}",
                "Données longitudinales": "✅" if df_firm["month_index"].notna().any() else "❌",
                "Alerte retraining": "🔴" if n < 50 else "✅",
            })
    return pd.DataFrame(rows)

cross_firm_df = _load_cross_firm()
if not cross_firm_df.empty:
    st.dataframe(cross_firm_df, use_container_width=True)
else:
    st.info("Aucune donnée inter-entreprise disponible.")

# ---------------------------------------------------------------------------
# Cross-firm pillar heatmap (anonymized)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📊 Comparaison inter-Firmes — Scores par pilier (anonymisés)")

@st.cache_data(ttl=60)
def _pillar_comparison() -> pd.DataFrame:
    with get_session() as session:
        firms = session.query(Firm).all()
        rows = []
        for firm in firms:
            resp = (
                session.query(SurveyResponse)
                .filter_by(firm_id=firm.firm_id)
                .with_entities(*[getattr(SurveyResponse, p) for p in PILLARS])
                .all()
            )
            if len(resp) < 5:
                continue
            df_f = pd.DataFrame(resp, columns=PILLARS)
            means = df_f.mean().to_dict()
            firm_hash = hashlib.sha256(firm.firm_id.encode()).hexdigest()[:6].upper()
            means["Entreprise"] = f"FIRM-{firm_hash}"
            rows.append(means)
    return pd.DataFrame(rows)

pillar_df = _pillar_comparison()
PILLAR_LABELS = ["Recrutement", "Compétences", "Performance",
                 "Rémunération", "QVT", "Droit", "Transverse"]

if not pillar_df.empty:
    pillar_df_plot = pillar_df.rename(columns=dict(zip(PILLARS, PILLAR_LABELS)))
    pillar_df_plot = pillar_df_plot.set_index("Entreprise")

    fig = px.imshow(
        pillar_df_plot,
        color_continuous_scale="RdYlGn",
        zmin=1, zmax=5,
        text_auto=".2f",
        title="Scores moyens par pilier — toutes entreprises (noms anonymisés)",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Données insuffisantes pour la comparaison inter-firmes (minimum 5 réponses par firme).")

# ---------------------------------------------------------------------------
# OFS Macro-data overview
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📋 Données OFS globales")

@st.cache_data(ttl=300)
def _load_ofs_full() -> pd.DataFrame:
    with get_session() as session:
        rows = session.query(OFSMacroData).all()
    return pd.DataFrame([{
        "Source": r.source_file, "Secteur": r.industry_domain,
        "Position": r.professional_position, "Âge": r.age_bracket,
        "Genre": r.gender, "Année": r.year,
        "Salaire médian (CHF)": r.gross_monthly_median_wage,
        "Taux turnover": r.turnover_rate,
    } for r in rows]) if rows else pd.DataFrame()

ofs_full = _load_ofs_full()
if not ofs_full.empty:
    is_synth = ofs_full["Source"].str.contains("synthetic").any()
    if is_synth:
        st.warning("⚠️ Données OFS synthétiques (fallback actif — parser .px non résolu).")
    with st.expander(f"Voir les {len(ofs_full)} lignes OFS"):
        st.dataframe(ofs_full.round(2), use_container_width=True)
else:
    st.info("Aucune donnée OFS chargée.")

# ---------------------------------------------------------------------------
# Users list (admin only — no passwords shown)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("👤 Utilisateurs enregistrés")

@st.cache_data(ttl=30)
def _load_users() -> pd.DataFrame:
    with get_session() as session:
        users = session.query(User).all()
        firms = {f.firm_id: f.name for f in session.query(Firm).all()}
    return pd.DataFrame([{
        "Utilisateur": u.username,
        "Nom": u.display_name,
        "Rôle": u.role,
        "Entreprise": firms.get(u.firm_id, "—"),
    } for u in users])

users_df = _load_users()
st.dataframe(users_df, use_container_width=True)

if st.button("🔄 Vider le cache et rafraîchir", type="secondary"):
    st.cache_data.clear()
    st.rerun()
