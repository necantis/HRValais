"""
pages/4_Dashboard_MixedModels.py
Dashboard 3 — Mixed Models: statsmodels logistic & OLS regressions.
Maps 7 HR Valais pillars to IBM HR Analytics variables to predict attrition.
Access: hr_manager only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.auth import require_role, get_current_user
from utils.test_data import format_firm_name
from utils.ibm_data import load_ibm_dataset, IBM_DATASET_LABEL
from db.database import get_session
from db.models import SurveyResponse, Firm

require_role("hr_manager", "admin")
user = get_current_user()
is_admin = user["role"] == "admin"

st.title("🔬 Modèles mixtes — Analyse de la rétention")

# ---------------------------------------------------------------------------
# Sidebar data source selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏢 Source de données")
    if is_admin:
        with get_session() as session:
            firms = session.query(Firm).order_by(Firm.name).all()
            firm_map = {format_firm_name(f.name): f.firm_id for f in firms}
        source_options = [IBM_DATASET_LABEL] + list(firm_map.keys())
        selected_source = st.selectbox("Sélectionner la source", source_options, index=0)
    else:
        my_firm = format_firm_name(user.get("firm_name", "Mon entreprise"))
        source_options = [my_firm, IBM_DATASET_LABEL]
        selected_source = st.selectbox("Sélectionner la source", source_options, index=0)
    st.divider()

is_ibm_source = (selected_source == IBM_DATASET_LABEL)
selected_display_name = "Dataset démo IBM (Attrition - 1'470 profils)" if is_ibm_source else selected_source

st.caption(f"Régression logistique et OLS sur les données de **{selected_display_name}**")

st.markdown("""
<div class="caveat-box">
⚠️ <strong>Avertissement statistique</strong> : Les p-values présentées ci-dessous sont de simples 
<em>indicateurs</em>. Elles ne constituent pas des probabilités précises ni une causalité établie. 
L'inférence causale nécessite une conception expérimentale rigoureuse au-delà de la portée de cet outil.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pillar → IBM variable mapping description
# ---------------------------------------------------------------------------
with st.expander("ℹ️ Correspondance Piliers HR Valais ↔ Variables IBM HR Analytics"):
    st.markdown("""
| Pilier HR Valais | Variables IBM proxy | Logique |
|---|---|---|
| Recrutement | JobSatisfaction, EnvironmentSatisfaction | Satisfaction initiale liée au recrutement |
| Compétences | TrainingTimesLastYear, JobLevel | Développement et positionnement hiérarchique |
| Performance | PerformanceRating, JobInvolvement | Évaluation formelle et engagement |
| Rémunération | MonthlyIncome, PercentSalaryHike | Compensation financière |
| QVT | WorkLifeBalance, OverTime | Équilibre vie-travail |
| Droit | YearsAtCompany, StockOptionLevel | Ancienneté et avantages légaux |
| Transverse | RelationshipSatisfaction, NumCompaniesWorked | Cohésion et mobilité |

*Source : IBM HR Analytics Employee Attrition dataset (n=1 470)*
""")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
PILLARS = ["recrutement_avg", "competences_avg", "performance_avg",
           "remuneration_avg", "qvt_avg", "droit_avg", "transverse_avg"]
PILLAR_LABELS = ["Recrutement", "Compétences", "Performance",
                 "Rémunération", "QVT", "Droit", "Transverse"]

@st.cache_data(ttl=120, show_spinner="Chargement des données…")
def _load_data(firm_id: str) -> pd.DataFrame:
    with get_session() as session:
        rows = (
            session.query(SurveyResponse)
            .filter_by(firm_id=firm_id)
            .with_entities(
                SurveyResponse.attrition_flag,
                SurveyResponse.engagement_state,
                *[getattr(SurveyResponse, p) for p in PILLARS],
            )
            .all()
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["attrition_flag", "engagement_state"] + PILLARS)

    # Safe conversion of attrition_flag to avoid TypeError when values are None/NaN
    def _to_int(flag, state):
        if pd.notna(flag):
            return 1 if flag is True or str(flag).lower() in ("true", "1", "yes") else 0
        return 1 if state == "Resigned" else 0

    df["attrition_int"] = [_to_int(a, s) for a, s in zip(df["attrition_flag"], df["engagement_state"])]
    df["resigned_int"] = (df["engagement_state"] == "Resigned").astype(int)
    return df.dropna(subset=PILLARS + ["resigned_int"])

if is_ibm_source:
    df = load_ibm_dataset()
else:
    selected_firm_id = firm_map[selected_source] if is_admin else user.get("firm_id")
    df = _load_data(selected_firm_id)

if df.empty or len(df) < 30:
    st.warning(f"Données insuffisantes pour l'analyse de {selected_display_name} (minimum 30 réponses nécessaires).")
    st.stop()

st.info(f"📊 **{len(df)}** réponses analysées pour **{selected_display_name}**")

# ---------------------------------------------------------------------------
# Logistic Regression: Predict Attrition from Pillar Scores
# ---------------------------------------------------------------------------
st.subheader("1. Régression logistique — Prédire l'attrition (Resigned = 1)")

try:
    import statsmodels.formula.api as smf
    warnings.filterwarnings("ignore")

    formula_parts = " + ".join(PILLARS)
    logit_formula = f"resigned_int ~ {formula_parts}"
    logit_model = smf.logit(logit_formula, data=df).fit(disp=0, maxiter=200)

    coef_df = pd.DataFrame({
        "Pilier": PILLARS,
        "Label": PILLAR_LABELS,
        "Coefficient": logit_model.params[PILLARS].values,
        "P-value": logit_model.pvalues[PILLARS].values,
        "OR (Odds Ratio)": np.exp(logit_model.params[PILLARS].values),
    }).sort_values("P-value")

    coef_df["Significatif (p<0.05)"] = coef_df["P-value"].apply(
        lambda p: "✅" if p < 0.05 else "—"
    )
    coef_df["Interprétation"] = coef_df["Coefficient"].apply(
        lambda c: "↑ Risque attrition" if c > 0 else "↓ Risque attrition"
    )

    st.dataframe(
        coef_df[["Label", "Coefficient", "OR (Odds Ratio)", "P-value", "Significatif (p<0.05)", "Interprétation"]]
        .round(4),
        use_container_width=True,
    )

    # Odds ratio plot
    fig_or = go.Figure()
    fig_or.add_trace(go.Bar(
        x=coef_df["OR (Odds Ratio)"],
        y=coef_df["Label"],
        orientation="h",
        marker_color=["#EF4444" if v > 1 else "#22C55E" for v in coef_df["OR (Odds Ratio)"]],
    ))
    fig_or.add_vline(x=1.0, line_dash="dash", line_color="#9CA3AF", annotation_text="OR=1 (neutre)")
    fig_or.update_layout(
        title="Odds Ratios — Impact des piliers sur le risque d'attrition",
        xaxis_title="Odds Ratio",
        template="plotly_dark",
        height=350,
    )
    st.plotly_chart(fig_or, use_container_width=True)

    # Model stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Log-Vraisemblance", f"{logit_model.llf:.1f}")
    c2.metric("Pseudo-R² (McFadden)", f"{logit_model.prsquared:.3f}")
    c3.metric("Observations", int(logit_model.nobs))

except Exception as e:
    st.error(f"Erreur lors de la régression logistique : {e}")

# ---------------------------------------------------------------------------
# OLS per-pillar correlations
# ---------------------------------------------------------------------------
st.divider()
st.subheader("2. Corrélations OLS — Score pilier vs. score global")

try:
    corr_results = []
    overall = df[PILLARS].mean(axis=1)

    for pillar, label in zip(PILLARS, PILLAR_LABELS):
        ols_formula = f"overall_score ~ {pillar}"
        df_tmp = df[[pillar]].copy()
        df_tmp["overall_score"] = overall
        df_tmp.columns = ["pillar_score", "overall_score"]
        ols = smf.ols("overall_score ~ pillar_score", data=df_tmp).fit()
        corr_results.append({
            "Pilier": label,
            "Coefficient": ols.params.get("pillar_score", np.nan),
            "R²": ols.rsquared,
            "P-value": ols.pvalues.get("pillar_score", np.nan),
        })

    ols_df = pd.DataFrame(corr_results).sort_values("R²", ascending=False)
    fig_ols = px.bar(
        ols_df, x="Pilier", y="R²",
        color="P-value",
        color_continuous_scale="RdYlGn_r",
        title="R² par pilier vs. score global (OLS)",
        template="plotly_dark",
        labels={"R²": "R² (variance expliquée)"},
    )
    st.plotly_chart(fig_ols, use_container_width=True)
    st.dataframe(ols_df.round(4), use_container_width=True)

except Exception as e:
    st.error(f"Erreur OLS : {e}")

# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
st.divider()
st.subheader("3. Matrice de corrélation des piliers")
corr_matrix = df[PILLARS].rename(columns=dict(zip(PILLARS, PILLAR_LABELS))).corr()
fig_heat = px.imshow(
    corr_matrix,
    color_continuous_scale="RdBu",
    zmin=-1, zmax=1,
    title="Corrélation inter-piliers",
    template="plotly_dark",
)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("""
<div class="caveat-box">
📌 <strong>Rappel</strong> : Les corrélations observées sur des données d'enquête agrégées sont
sujettes au biais écologique et à la causalité inverse. Ces résultats guident l'exploration,
ils ne remplacent pas une analyse causale rigoureuse.
</div>
""", unsafe_allow_html=True)
