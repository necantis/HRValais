"""
pages/7_Admin.py
Admin Dashboard — global cross-tenant aggregated view or firm management.
Access: admin, hr_manager, employee.
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
from db.models import Firm, User, SurveyResponse, OFSMacroData, MonthlyUpload, ActivityLog

require_role("admin", "hr_manager", "employee")
user = get_current_user()
is_admin = user["role"] == "admin"
is_employee = user["role"] == "employee"

st.title("⚙️ Administration HR Valais" if is_admin else f"⚙️ Administration - {user.get('firm_name', 'Firme')}")
st.caption(f"Vue globale agrégée — {datetime.now().strftime('%d %B %Y %H:%M')}" if is_admin else f"Vue entreprise — {datetime.now().strftime('%d %B %Y %H:%M')}")

if is_employee:
    st.subheader("🔐 Changer mon mot de passe")
    with st.form("change_my_password"):
        new_pass = st.text_input("Nouveau mot de passe", type="password")
        confirm_pass = st.text_input("Confirmer le mot de passe", type="password")
        if st.form_submit_button("Mettre à jour"):
            if new_pass and new_pass == confirm_pass:
                with get_session() as session:
                    u_db = session.query(User).get(user["user_id"])
                    u_db.hashed_password = hash_password(new_pass)
                st.success("Mot de passe mis à jour avec succès !")
            else:
                st.error("Les mots de passe ne correspondent pas ou sont vides.")
    st.stop()

# ---------------------------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def _load_users(firm_id=None) -> pd.DataFrame:
    with get_session() as session:
        if firm_id:
            users_db = session.query(User).filter_by(firm_id=firm_id).all()
        else:
            users_db = session.query(User).all()
        firms_db = {f.firm_id: f.name for f in session.query(Firm).all()}
        return pd.DataFrame([{
            "user_id": u.user_id,
            "Utilisateur": u.username,
            "Nom": u.display_name,
            "Rôle": u.role,
            "Entreprise": firms_db.get(u.firm_id, "—"),
            "Limite/An": getattr(u, 'max_surveys_per_year', 1),
            "Nouveau mot de passe": ""
        } for u in users_db])

# ===========================================================================
# Define Tabs (Only Admin gets multiple tabs)
# ===========================================================================
if is_admin:
    tab1, tab2, tab3 = st.tabs(["Gestion", "Santé DB", "Test Hypothèses"])
else:
    # Hack to simulate tabs block for non-admins so we can just use tab1
    tab1 = st.container()
    tab2 = None
    tab3 = None

# ===========================================================================
# TAB 1: Gestion (Firm & User Management)
# ===========================================================================
with tab1:
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
                        session.add(User(
                            user_id=str(uuid.uuid4()), firm_id=new_firm_id, username=f"manager_{new_firm_name[:3].lower()}",
                            role="hr_manager", hashed_password=hash_password("password123"), display_name=f"Manager ({new_firm_name})"
                        ))
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
            if submitted_user and new_username:
                with get_session() as session:
                    target_firm_id = None
                    if is_admin and firm_name_sel != "Aucune":
                        target_firm_id = firm_map[firm_name_sel]
                    elif not is_admin:
                        target_firm_id = user["firm_id"]
                        
                    session.add(User(
                        user_id=str(uuid.uuid4()), firm_id=target_firm_id, username=new_username,
                        role=new_role, hashed_password=hash_password("ChangeMe"), display_name=new_display
                    ))
                st.success("Utilisateur créé avec succès. Le mot de passe par défaut est 'ChangeMe'.")
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
                    _load_users.clear()
                    st.success("Limite mise à jour.")
                    st.rerun()
            else:
                st.info("Aucun utilisateur disponible.")
                st.form_submit_button("Mettre à jour", disabled=True)

    st.divider()

    st.subheader("👤 Utilisateurs enregistrés")
    users_df = _load_users(None if is_admin else user["firm_id"])

    with get_session() as session:
        firm_map_inv = {f.name: f.firm_id for f in session.query(Firm).all()}
        firm_map_inv["—"] = None
        firm_names = list(f.name for f in session.query(Firm).all())

    disabled_cols = [] if is_admin else ["Rôle", "Entreprise"]

    with st.form("edit_users_form"):
        st.write("Modifiez les informations utilisateur ci-dessous :")
        edited_df = st.data_editor(
            users_df,
            column_config={
                "user_id": None,
                "Rôle": st.column_config.SelectboxColumn("Rôle", options=["admin", "hr_manager", "employee"]),
                "Entreprise": st.column_config.SelectboxColumn("Entreprise", options=["—"] + firm_names),
                "Limite/An": st.column_config.NumberColumn("Limite/An", min_value=1, step=1),
                "Nouveau mot de passe": st.column_config.TextColumn("Nouveau mot de passe (laisser vide sinon)")
            },
            disabled=disabled_cols,
            use_container_width=True,
            hide_index=True
        )
        
        if st.form_submit_button("Sauvegarder les modifications"):
            changes_made = False
            with get_session() as session:
                for i, row in edited_df.iterrows():
                    uid = row["user_id"]
                    new_username = row["Utilisateur"]
                    new_name = row["Nom"]
                    new_role = row["Rôle"]
                    new_firm = row["Entreprise"]
                    new_limit = row["Limite/An"]
                    new_pw = row["Nouveau mot de passe"]
                    
                    orig_row = users_df.iloc[i]
                    orig_username = orig_row["Utilisateur"]
                    orig_name = orig_row["Nom"]
                    orig_role = orig_row["Rôle"]
                    orig_firm = orig_row["Entreprise"]
                    orig_limit = orig_row["Limite/An"]
                    
                    if (new_username != orig_username or 
                        new_name != orig_name or 
                        new_role != orig_role or 
                        new_firm != orig_firm or 
                        new_limit != orig_limit or 
                        new_pw.strip() != ""):
                        
                        u_db = session.query(User).get(uid)
                        u_db.username = new_username
                        u_db.display_name = new_name
                        u_db.role = new_role
                        u_db.firm_id = firm_map_inv.get(new_firm)
                        u_db.max_surveys_per_year = new_limit
                        if new_pw.strip() != "":
                            u_db.hashed_password = hash_password(new_pw.strip())
                        changes_made = True
            
            if changes_made:
                _load_users.clear()
                st.success("Modifications sauvegardées avec succès.")
                st.rerun()
            else:
                st.info("Aucune modification détectée.")


# ===========================================================================
# TAB 2: Santé DB
# ===========================================================================
if is_admin:
    with tab2:
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
        
        if st.button("⬇️ Charger les données OFS globales", help="Ceci peut prendre du temps si la base de données est volumineuse"):
            ofs_full = _load_ofs_full()
            if not ofs_full.empty:
                is_synth = ofs_full["Source"].str.contains("synthetic").any()
                if is_synth:
                    st.warning("⚠️ Données OFS synthétiques (fallback actif — parser .px non résolu).")
                with st.expander(f"Voir les {len(ofs_full)} lignes OFS", expanded=True):
                    st.dataframe(ofs_full.round(2), use_container_width=True)
            else:
                st.info("Aucune donnée OFS chargée.")

# ===========================================================================
# TAB 3: Test Hypothèses (Kaggle Analytics)
# ===========================================================================
if is_admin:
    with tab3:
        st.subheader("🧪 Test d'Hypothèses et Analyse Comportementale")
        st.markdown(
            "Analyse de la théorie de contingence : corrélation entre "
            "le temps passé, l'intensité des clics, et les scores de performance des entreprises."
        )

        @st.cache_data(ttl=60)
        def _load_telemetry() -> pd.DataFrame:
            with get_session() as session:
                logs = session.query(ActivityLog).all()
                firms = session.query(Firm).all()
                firm_map = {f.firm_id: f.name for f in firms}
                
                # We need firm scores as well
                from db.models import SurveyResponse
                PILLARS = ["recrutement_avg", "competences_avg", "performance_avg",
                           "remuneration_avg", "qvt_avg", "droit_avg", "transverse_avg"]
                
                firm_scores = {}
                for f in firms:
                    resp = session.query(SurveyResponse).filter_by(firm_id=f.firm_id).all()
                    if resp:
                        df_resp = pd.DataFrame([{p: getattr(r, p) for p in PILLARS} for r in resp])
                        firm_scores[f.firm_id] = df_resp.mean().mean()

                if not logs:
                    return pd.DataFrame()
                
                data = []
                # Process logs per firm
                df_logs = pd.DataFrame([{
                    "firm_id": l.firm_id,
                    "action_type": l.action_type,
                    "action_value": l.action_value,
                    "timestamp": l.timestamp
                } for l in logs])

                for firm_id, group in df_logs.groupby("firm_id"):
                    # 1. Avg Survey Time
                    survey_logs = group[group["action_type"] == "survey_completion_time"]
                    avg_survey_time = survey_logs["action_value"].astype(float).mean() if not survey_logs.empty else 0
                    
                    # 2. Total Dashboard Pings
                    dash_pings = group[group["action_type"] == "dashboard_ping"]
                    # For simplicity, count of pings = proxy for visits
                    dashboard_visits = len(dash_pings)
                    
                    # 3. Total Link Clicks
                    click_logs = group[group["action_type"] == "link_click"]
                    total_clicks = len(click_logs)

                    score = firm_scores.get(firm_id, np.nan)
                    firm_name = firm_map.get(firm_id, "Inconnue")
                    
                    # Feature Engineering: Dashboard Intensity
                    # (Clicks per visit - simple proxy since time-on-page requires JS)
                    intensity = total_clicks / dashboard_visits if dashboard_visits > 0 else 0

                    data.append({
                        "Entreprise": firm_name,
                        "Temps moyen sondage (s)": avg_survey_time,
                        "Visites Dashboard": dashboard_visits,
                        "Clics Totaux": total_clicks,
                        "Intensité Dashboard (clics/visite)": intensity,
                        "Score Global (Performance)": score
                    })
                
                return pd.DataFrame(data)

        df_telemetry = _load_telemetry()

        if df_telemetry.empty:
            st.info("Aucune donnée de télémétrie enregistrée pour le moment. Naviguez sur les dashboards et remplissez des sondages pour générer des données.")
        else:
            # Histograms
            st.markdown("#### 1. Distributions")
            c1, c2, c3 = st.columns(3)
            with c1:
                fig1 = px.histogram(df_telemetry, x="Temps moyen sondage (s)", title="Temps Sondage", template="plotly_dark", nbins=10)
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = px.histogram(df_telemetry, x="Visites Dashboard", title="Visites", template="plotly_dark", nbins=10)
                st.plotly_chart(fig2, use_container_width=True)
            with c3:
                fig3 = px.histogram(df_telemetry, x="Clics Totaux", title="Clics", template="plotly_dark", nbins=10)
                st.plotly_chart(fig3, use_container_width=True)

            st.divider()

            # Scatter Plots with OLS Trendlines
            st.markdown("#### 2. Corrélations avec la Performance")
            
            c4, c5 = st.columns(2)
            with c4:
                fig4 = px.scatter(
                    df_telemetry, x="Temps moyen sondage (s)", y="Score Global (Performance)", 
                    color="Entreprise", trendline="ols",
                    title="Score vs Temps Sondage", template="plotly_dark"
                )
                st.plotly_chart(fig4, use_container_width=True)
            with c5:
                fig5 = px.scatter(
                    df_telemetry, x="Intensité Dashboard (clics/visite)", y="Score Global (Performance)", 
                    color="Entreprise", trendline="ols",
                    title="Score vs Intensité d'utilisation", template="plotly_dark"
                )
                st.plotly_chart(fig5, use_container_width=True)
            
            st.divider()
            
            # K-Means Clustering Preview
            st.markdown("#### 3. Clustering K-Means (Adaptation des entreprises)")
            st.markdown("Groupement automatique des entreprises selon leur **Intensité d'utilisation** et leur **Score global**.")
            
            df_cluster = df_telemetry.dropna(subset=["Intensité Dashboard (clics/visite)", "Score Global (Performance)"]).copy()
            if len(df_cluster) >= 3:
                from sklearn.cluster import KMeans
                X = df_cluster[["Intensité Dashboard (clics/visite)", "Score Global (Performance)"]]
                # Normalize
                X_norm = (X - X.mean()) / X.std()
                # K-Means with k=min(3, n_samples)
                k = min(3, len(X))
                kmeans = KMeans(n_clusters=k, random_state=42)
                df_cluster["Cluster ID"] = kmeans.fit_predict(X_norm).astype(str)
                
                fig_cluster = px.scatter(
                    df_cluster, x="Intensité Dashboard (clics/visite)", y="Score Global (Performance)",
                    color="Cluster ID", hover_data=["Entreprise"],
                    title=f"K-Means Clustering (k={k})", template="plotly_dark", size_max=15
                )
                fig_cluster.update_traces(marker=dict(size=12))
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.info("Pas assez de données pour effectuer un clustering K-Means. (Minimum 3 entreprises avec scores et télémétrie requises).")

if st.button("🔄 Vider le cache et rafraîchir", type="secondary"):
    st.cache_data.clear()
    st.rerun()
