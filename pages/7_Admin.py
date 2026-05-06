"""
pages/7_Admin.py
Admin Dashboard — global cross-tenant aggregated view or firm management.
Access: admin, hr_manager.
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
import uuid
from datetime import datetime

from utils.auth import require_role, get_current_user, hash_password
from db.database import get_session, DB_PATH
from db.models import Firm, User, SurveyResponse, OFSMacroData, MonthlyUpload

require_role("admin", "hr_manager")
user = get_current_user()
is_admin = user["role"] == "admin"

st.title("⚙️ Administration HR Valais" if is_admin else f"⚙️ Administration - {user.get('firm_name', 'Firme')}")
st.caption(f"Vue globale agrégée — {datetime.now().strftime('%d %B %Y %H:%M')}" if is_admin else f"Vue entreprise — {datetime.now().strftime('%d %B %Y %H:%M')}")

# ---------------------------------------------------------------------------
# Firm & User Management
# ---------------------------------------------------------------------------
st.subheader("🛠️ Gestion des Comptes")

if is_admin:
    with st.expander("➕ Créer une Entreprise"):
        with st.form("create_firm_form"):
            new_firm_name = st.text_input("Nom de l'entreprise")
            domain = st.text_input("Domaine industriel")
            submitted_firm = st.form_submit_button("Créer l'entreprise")
            if submitted_firm and new_firm_name:
                with get_session() as session:
                    new_firm_id = str(uuid.uuid4())
                    session.add(Firm(firm_id=new_firm_id, name=new_firm_name, industry_domain=domain))
                    # Create manager
                    session.add(User(
                        user_id=str(uuid.uuid4()), firm_id=new_firm_id, username=f"manager_{new_firm_name[:3].lower()}",
                        role="hr_manager", hashed_password=hash_password("password123"), display_name=f"Manager ({new_firm_name})"
                    ))
                    # Create employee
                    session.add(User(
                        user_id=str(uuid.uuid4()), firm_id=new_firm_id, username=f"employee_{new_firm_name[:3].lower()}",
                        role="employee", hashed_password=hash_password("password123"), display_name=f"Employee ({new_firm_name})"
                    ))
                st.success("Entreprise et comptes créés avec succès (mot de passe par défaut: password123).")
                st.rerun()

with st.expander("👤 Créer un Utilisateur"):
    with st.form("create_user_form"):
        new_username = st.text_input("Nom d'utilisateur")
        new_display = st.text_input("Nom affiché")
        new_password = st.text_input("Mot de passe", type="password")
        
        if is_admin:
            with get_session() as session:
                firm_map = {f.name: f.firm_id for f in session.query(Firm).all()}
            firm_name_sel = st.selectbox("Entreprise", ["Aucune"] + list(firm_map.keys()))
            new_role = st.selectbox("Rôle", ["employee", "hr_manager", "admin"])
        else:
            firm_name_sel = user["firm_name"]
            st.text_input("Entreprise", value=firm_name_sel, disabled=True)
            new_role = "employee"
            st.text_input("Rôle", value="Employee", disabled=True)
            
        submitted_user = st.form_submit_button("Créer l'utilisateur")
        if submitted_user and new_username and new_password:
            with get_session() as session:
                target_firm_id = None
                if is_admin:
                    if firm_name_sel != "Aucune":
                        target_firm_id = firm_map[firm_name_sel]
                else:
                    target_firm_id = user["firm_id"]
                    
                session.add(User(
                    user_id=str(uuid.uuid4()), firm_id=target_firm_id, username=new_username,
                    role=new_role, hashed_password=hash_password(new_password), display_name=new_display
                ))
            st.success("Utilisateur créé avec succès.")
            st.rerun()

with st.expander("🔢 Modifier les limites de sondages"):
    with st.form("update_limit_form"):
        with get_session() as session:
            if is_admin:
                all_users = session.query(User).all()
            else:
                all_users = session.query(User).filter_by(firm_id=user["firm_id"]).all()
            user_map_limits = {f"{u.username} ({u.role})": u.user_id for u in all_users}
        
        if user_map_limits:
            target_u = st.selectbox("Sélectionner un utilisateur", list(user_map_limits.keys()))
            new_limit = st.number_input("Nouvelle limite (soumissions par an)", min_value=1, value=1)
            if st.form_submit_button("Mettre à jour"):
                with get_session() as session:
                    u_id = user_map_limits[target_u]
                    u_db = session.query(User).get(u_id)
                    u_db.max_surveys_per_year = new_limit
                st.success("Limite mise à jour.")
                st.rerun()
        else:
            st.info("Aucun utilisateur disponible.")
            st.form_submit_button("Mettre à jour", disabled=True)

st.divider()

# ---------------------------------------------------------------------------
# Users list (No passwords shown)
# ---------------------------------------------------------------------------
st.subheader("👤 Utilisateurs enregistrés")

@st.cache_data(ttl=30)
def _load_users(firm_id=None) -> pd.DataFrame:
    with get_session() as session:
        if firm_id:
            users_db = session.query(User).filter_by(firm_id=firm_id).all()
        else:
            users_db = session.query(User).all()
        firms_db = {f.firm_id: f.name for f in session.query(Firm).all()}
        return pd.DataFrame([{
            "Utilisateur": u.username,
            "Nom": u.display_name,
            "Rôle": u.role,
            "Entreprise": firms_db.get(u.firm_id, "—"),
            "Limite/An": getattr(u, 'max_surveys_per_year', 1)
        } for u in users_db])

users_df = _load_users(None if is_admin else user["firm_id"])
st.dataframe(users_df, use_container_width=True)

if is_admin:
    # ---------------------------------------------------------------------------
    # DB Health Metrics
    # ---------------------------------------------------------------------------
    st.divider()
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
            firms_q = session.query(Firm).all()
            rows = []
            for firm in firms_q:
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
    
                # Anonymise firm name with hash
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
            firms_db = session.query(Firm).all()
            rows = []
            for firm in firms_db:
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
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame([{
                "Source": r.source_file, "Secteur": r.industry_domain,
                "Position": r.professional_position, "Âge": r.age_bracket,
                "Genre": r.gender, "Année": r.year,
                "Salaire médian (CHF)": r.gross_monthly_median_wage,
                "Taux turnover": r.turnover_rate,
            } for r in rows])
    
    ofs_full = _load_ofs_full()
    if not ofs_full.empty:
        is_synth = ofs_full["Source"].str.contains("synthetic").any()
        if is_synth:
            st.warning("⚠️ Données OFS synthétiques (fallback actif — parser .px non résolu).")
        with st.expander(f"Voir les {len(ofs_full)} lignes OFS"):
            st.dataframe(ofs_full.round(2), use_container_width=True)
    else:
        st.info("Aucune donnée OFS chargée.")

if st.button("🔄 Vider le cache et rafraîchir", type="secondary"):
    st.cache_data.clear()
    st.rerun()
