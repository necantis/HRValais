"""
pages/8_Advanced_Predictions.py
Advanced Causal Predictions — HR Valais
Access: hr_manager | admin only.

Pipeline:
  1. Load per-firm survey responses.
  2. Fit a baseline logistic model (intercept-only) and an extended logistic
     model (pillar scores as covariates) using statsmodels.
  3. Evaluate out-of-sample (OOS) predictive error via cross-validated log-loss.
  4. Run DAG-inspired conditional independence (CI) falsification tests using
     partial correlation residuals.
  5. Raise a retraining flag if OOS error > naive heuristic baseline, or if
     any CI test reveals a statistically significant violation.

Falsifiability constraints (per spec):
  - Prominent markdown caveat asking about the likelihood the baseline is
    correct vs. our extension being an illusion.
  - P-values explicitly labelled as "directional indicators rather than
    accurate probabilities".
  - Retraining flag triggers displayed in the UI.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from db.database import get_session
from db.models import SurveyResponse
from utils.auth import get_current_user, require_role

# ---------------------------------------------------------------------------
# RBAC guard
# ---------------------------------------------------------------------------
require_role("hr_manager", "admin")
user = get_current_user()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🧪 Prédictions avancées — Analyse causale")
st.caption("Pipeline de régression logistique avec falsifiabilité explicite · Accès RH Manager / Admin")

# ---------------------------------------------------------------------------
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FALSIFIABILITY CAVEAT — MANDATORY, PROMINENT                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 100%);
    border: 2px solid #9B59B6;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
">
<h3 style="color: #CE93D8; margin-top: 0;">🔬 Avertissement de Falsifiabilité — À lire avant toute interprétation</h3>

<p style="color: #E0E0E0; font-size: 0.95rem; line-height: 1.7;">
<strong style="color: #F48FB1;">Question centrale :</strong>
<em>Quelle est la probabilité actuelle que le modèle de base (intercept uniquement) soit correct,
par opposition à notre extension causale qui serait une <strong>illusion statistique</strong> ?</em>
</p>

<ul style="color: #BDBDBD; font-size: 0.9rem; line-height: 1.8;">
  <li>Le modèle étendu incorpore les scores des piliers RH comme covariables — mais une amélioration
      apparente de l'ajustement peut refléter un <strong>sur-ajustement aux données d'entraînement</strong>,
      et non une relation causale réelle.</li>
  <li>Les p-values ci-dessous sont affichées comme <strong>indicateurs directionnels plutôt que comme
      probabilités précises</strong>. Elles ne quantifient pas la causalité.</li>
  <li>L'inférence causale requiert un graphe acyclique dirigé (DAG) validé, des données
      longitudinales, et idéalement une conception expérimentale randomisée.</li>
  <li>Ce pipeline inclut des <strong>tests de falsification</strong> automatisés. Si les résultats
      dépassent les seuils définis, un drapeau de réentraînement est activé et consigné.</li>
</ul>

<p style="color: #FF8A65; font-size: 0.85rem; margin-bottom: 0;">
⚡ <strong>Interprétez avec prudence :</strong> Les décisions RH ne doivent jamais reposer
uniquement sur ces sorties. Consultez un statisticien avant toute action substantielle.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PILLARS = [
    "recrutement_avg", "competences_avg", "performance_avg",
    "remuneration_avg", "qvt_avg", "droit_avg", "transverse_avg",
]
PILLAR_LABELS = [
    "Recrutement", "Compétences", "Performance",
    "Rémunération", "QVT", "Droit", "Transverse",
]
LABEL_MAP = dict(zip(PILLARS, PILLAR_LABELS))

# DAG: assumed causal structure (partial list of expected conditional independences)
# Format: (A, B, conditioning_set) — we test whether A ⊥ B | conditioning_set
# If rejected, the DAG assumption is violated.
DAG_CI_TESTS: list[tuple[str, str, list[str]]] = [
    # Hypothesis: Recrutement → Compétences, so Recrutement ⊥ Performance | Compétences
    ("recrutement_avg", "performance_avg", ["competences_avg"]),
    # Hypothesis: QVT → Engagement, so QVT ⊥ Rémunération | transverse_avg
    ("qvt_avg", "remuneration_avg", ["transverse_avg"]),
    # Hypothesis: Droit → ancienneté, so Droit ⊥ Compétences | Performance
    ("droit_avg", "competences_avg", ["performance_avg"]),
]

RETRAINING_LOG_KEY = "adv_pred_retraining_flags"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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
    df["y"] = (df["engagement_state"] == "Resigned").astype(int)
    return df.dropna(subset=PILLARS + ["y"])


df_raw = _load_data(user["firm_id"])

if df_raw.empty or len(df_raw) < 40:
    st.warning(
        "⚠️ Données insuffisantes pour ce pipeline (minimum 40 réponses avec scores piliers complets)."
    )
    st.stop()

df = df_raw.copy()

st.info(
    f"📊 **{len(df)} réponses** analysées pour **{user.get('firm_name', 'votre entreprise')}**  "
    f"· Prévalence attrition : **{df['y'].mean():.1%}**"
)

# ---------------------------------------------------------------------------
# Section 1 — Baseline vs. Extended Model Fit
# ---------------------------------------------------------------------------
st.divider()
st.subheader("1. Modèle de base vs. modèle étendu")

col_info, col_warn = st.columns([2, 1])
with col_info:
    st.markdown("""
**Modèle de base** : intercept uniquement — prédit la prévalence marginale.  
**Modèle étendu** : régression logistique avec les 7 piliers RH comme covariables.
""")
with col_warn:
    st.markdown("""
<div style="background:#1C2333;border-left:4px solid #F6AD55;padding:8px 12px;border-radius:4px;font-size:0.82rem;color:#F6E05E;">
⚠️ Les p-values sont des <strong>indicateurs directionnels</strong>,<br>
pas des probabilités précises.
</div>
""", unsafe_allow_html=True)

warnings.filterwarnings("ignore")

try:
    # --- Baseline model (intercept only) ---
    X_base = sm.add_constant(pd.DataFrame({"const": np.ones(len(df))}))
    baseline_model = sm.Logit(df["y"], X_base["const"].to_frame()).fit(
        disp=0, maxiter=200
    )

    # --- Extended model ---
    formula = "y ~ " + " + ".join(PILLARS)
    ext_model = smf.logit(formula, data=df).fit(disp=0, maxiter=200)

    # --- Likelihood-ratio test (baseline vs. extended) ---
    lr_stat = -2 * (baseline_model.llf - ext_model.llf)
    lr_df = len(PILLARS)
    lr_p = stats.chi2.sf(lr_stat, df=lr_df)

    # Display model comparison
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("LL baseline", f"{baseline_model.llf:.1f}")
    col_b.metric("LL modèle étendu", f"{ext_model.llf:.1f}")
    col_c.metric("Pseudo-R² (McFadden)", f"{ext_model.prsquared:.3f}")
    col_d.metric(
        "Test LR p-value ★",
        f"{lr_p:.4f}",
        delta="Modèle étendu meilleur" if lr_p < 0.05 else "Différence non significative",
        delta_color="normal" if lr_p < 0.05 else "off",
    )

    st.caption(
        "★ La p-value du test du rapport de vraisemblance (LR) est un **indicateur directionnel** "
        "de si le modèle étendu dépasse le modèle de base. Elle n'établit pas de causalité."
    )

    # --- Coefficient table ---
    st.markdown("#### Coefficients du modèle étendu")
    coef_df = pd.DataFrame({
        "Pilier": PILLAR_LABELS,
        "Colonne": PILLARS,
        "Coefficient (log-odds)": ext_model.params[PILLARS].values,
        "Odds Ratio": np.exp(ext_model.params[PILLARS].values),
        "P-value [directionnel]": ext_model.pvalues[PILLARS].values,
        "IC 95% inf": np.exp(ext_model.conf_int().loc[PILLARS, 0].values),
        "IC 95% sup": np.exp(ext_model.conf_int().loc[PILLARS, 1].values),
    }).sort_values("P-value [directionnel]")

    coef_df["Signal"] = coef_df["P-value [directionnel]"].apply(
        lambda p: "🔴 Fort signal" if p < 0.01
        else ("🟡 Signal modéré" if p < 0.05 else "⚪ Signal faible")
    )
    coef_df["Direction"] = coef_df["Coefficient (log-odds)"].apply(
        lambda c: "↑ Risque attrition" if c > 0 else "↓ Risque attrition"
    )

    st.dataframe(
        coef_df[["Pilier", "Coefficient (log-odds)", "Odds Ratio",
                 "IC 95% inf", "IC 95% sup", "P-value [directionnel]",
                 "Signal", "Direction"]].round(4),
        use_container_width=True,
        hide_index=True,
    )

    # Odds ratio forest plot
    fig_or = go.Figure()
    colors = ["#EF4444" if v > 1 else "#22C55E" for v in coef_df["Odds Ratio"]]
    fig_or.add_trace(go.Scatter(
        x=coef_df["Odds Ratio"],
        y=coef_df["Pilier"],
        mode="markers",
        marker=dict(size=14, color=colors, symbol="diamond"),
        error_x=dict(
            type="data",
            symmetric=False,
            array=(coef_df["IC 95% sup"] - coef_df["Odds Ratio"]).tolist(),
            arrayminus=(coef_df["Odds Ratio"] - coef_df["IC 95% inf"]).tolist(),
            color="#9CA3AF",
            thickness=2,
        ),
        name="Odds Ratio",
    ))
    fig_or.add_vline(x=1.0, line_dash="dash", line_color="#9CA3AF",
                     annotation_text="OR=1 (neutre)")
    fig_or.update_layout(
        title="Forest plot — Odds Ratios avec IC 95%",
        xaxis_title="Odds Ratio (indicateur directionnel)",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_or, use_container_width=True)

except Exception as exc:
    st.error(f"Erreur lors de l'ajustement des modèles : {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Section 2 — Out-of-Sample (OOS) Predictive Error vs. Naive Baseline
# ---------------------------------------------------------------------------
st.divider()
st.subheader("2. Erreur prédictive hors-échantillon (OOS) vs. heuristique naïve")

st.markdown("""
L'erreur **OOS** mesure la performance du modèle sur des données non vues via validation croisée.  
La **heuristique naïve** prédit toujours la prévalence marginale (probabilité constante).
Si l'erreur OOS du modèle étendu ≥ heuristique naïve → **drapeau de réentraînement activé**.
""")

try:
    X_feat = df[PILLARS].values
    y_arr = df["y"].values
    prevalence = y_arr.mean()

    # Naive heuristic log-loss
    naive_proba = np.full(len(y_arr), prevalence)
    naive_logloss = log_loss(y_arr, naive_proba)

    # Cross-validated log-loss for extended model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oos_losses = []

    for train_idx, test_idx in skf.split(X_feat, y_arr):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()
        try:
            fold_model = smf.logit(formula, data=df_train).fit(disp=0, maxiter=200)
            fold_proba = fold_model.predict(df_test)
            fold_proba = np.clip(fold_proba, 1e-7, 1 - 1e-7)
            oos_losses.append(log_loss(df_test["y"].values, fold_proba))
        except Exception:
            oos_losses.append(naive_logloss)  # fallback on convergence failure

    mean_oos_loss = float(np.mean(oos_losses))
    oos_better = mean_oos_loss < naive_logloss
    oos_improvement_pct = (naive_logloss - mean_oos_loss) / naive_logloss * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Log-loss naïf (heuristique)", f"{naive_logloss:.4f}")
    col2.metric(
        "Log-loss OOS moyen (5-fold CV)",
        f"{mean_oos_loss:.4f}",
        delta=f"{oos_improvement_pct:+.1f}% vs naïf",
        delta_color="normal" if oos_better else "inverse",
    )
    col3.metric(
        "Verdict OOS",
        "✅ Modèle utile" if oos_better else "🚨 Drapeau réentraînement",
        delta="Dépasse l'heuristique" if oos_better else "En-dessous de l'heuristique",
        delta_color="normal" if oos_better else "inverse",
    )

    # Fold-by-fold chart
    fold_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(oos_losses))],
        "Log-loss OOS": oos_losses,
        "Heuristique naïve": [naive_logloss] * len(oos_losses),
    })
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(
        x=fold_df["Fold"], y=fold_df["Log-loss OOS"],
        name="Log-loss OOS",
        marker_color=["#22C55E" if v < naive_logloss else "#EF4444" for v in oos_losses],
    ))
    fig_cv.add_hline(
        y=naive_logloss, line_dash="dot", line_color="#F6AD55",
        annotation_text="Heuristique naïve", annotation_position="top right",
    )
    fig_cv.update_layout(
        title="Log-loss par fold — Modèle étendu vs. heuristique naïve",
        yaxis_title="Log-loss (↓ meilleur)",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    # Store retraining flag
    oos_flag = not oos_better

except Exception as exc:
    st.error(f"Erreur lors de l'évaluation OOS : {exc}")
    oos_flag = False
    mean_oos_loss = None

# ---------------------------------------------------------------------------
# Section 3 — DAG Falsification Tests (Conditional Independence)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("3. Tests de falsification du DAG — Indépendances conditionnelles")

st.markdown("""
Le DAG postule des **indépendances conditionnelles** entre certaines paires de piliers.  
Nous testons ces hypothèses via corrélation partielle des résidus OLS.  
Un **test significatif (p < 0.05)** révèle une violation de l'indépendance et constitue
un signal de mauvaise spécification causale.
""")

ci_results = []
ci_flag = False

for (var_a, var_b, cond_set) in DAG_CI_TESTS:
    try:
        # Partial correlation: residualise A and B on the conditioning set
        # then test if the residuals are correlated.
        if cond_set:
            reg_a = smf.ols(f"{var_a} ~ " + " + ".join(cond_set), data=df).fit()
            reg_b = smf.ols(f"{var_b} ~ " + " + ".join(cond_set), data=df).fit()
            res_a = reg_a.resid.values
            res_b = reg_b.resid.values
        else:
            res_a = df[var_a].values - df[var_a].mean()
            res_b = df[var_b].values - df[var_b].mean()

        r, p_val = stats.pearsonr(res_a, res_b)
        violation = p_val < 0.05

        cond_label = (
            " | " + ", ".join([LABEL_MAP.get(c, c) for c in cond_set])
            if cond_set else ""
        )
        ci_results.append({
            "Hypothèse d'indépendance":
                f"{LABEL_MAP.get(var_a, var_a)} ⊥ {LABEL_MAP.get(var_b, var_b)}{cond_label}",
            "Corrélation partielle r": round(r, 4),
            "P-value [directionnel]": round(p_val, 4),
            "Violation DAG": "🔴 Violation" if violation else "✅ Respecté",
        })
        if violation:
            ci_flag = True

    except Exception as exc:
        ci_results.append({
            "Hypothèse d'indépendance": f"{var_a} ⊥ {var_b} | ...",
            "Corrélation partielle r": None,
            "P-value [directionnel]": None,
            "Violation DAG": f"⚠️ Erreur : {exc}",
        })

ci_df = pd.DataFrame(ci_results)
st.dataframe(ci_df, use_container_width=True, hide_index=True)

if ci_flag:
    st.error(
        "🚨 **Violation(s) d'indépendance conditionnelle détectée(s).** "
        "Le DAG actuel est mal spécifié. Le modèle causal est flaggé pour révision."
    )
else:
    st.success(
        "✅ Aucune violation d'indépendance conditionnelle détectée. "
        "Le DAG est cohérent avec les données observées."
    )

# ---------------------------------------------------------------------------
# Section 4 — Retraining Trigger Summary
# ---------------------------------------------------------------------------
st.divider()
st.subheader("4. Tableau de bord des drapeaux de réentraînement")

flags = []

if "oos_flag" in dir() and oos_flag:
    flags.append({
        "Déclencheur": "Erreur OOS ≥ heuristique naïve",
        "Valeur": f"OOS log-loss = {mean_oos_loss:.4f} ≥ {naive_logloss:.4f}",
        "Action": "Réentraîner le modèle étendu avec nouvelles données",
        "Statut": "🔴 Actif",
    })

if ci_flag:
    violations = [r["Hypothèse d'indépendance"] for r in ci_results
                  if "Violation" in str(r.get("Violation DAG", ""))]
    flags.append({
        "Déclencheur": "Violation(s) du DAG (indépendance conditionnelle)",
        "Valeur": "; ".join(violations),
        "Action": "Réviser la structure causale du DAG avant réentraînement",
        "Statut": "🔴 Actif",
    })

if not flags:
    flags.append({
        "Déclencheur": "Aucun",
        "Valeur": "Tous les seuils sont respectés",
        "Action": "—",
        "Statut": "✅ Aucun drapeau",
    })

flags_df = pd.DataFrame(flags)

# Persist flags to session state for audit trail
if RETRAINING_LOG_KEY not in st.session_state:
    st.session_state[RETRAINING_LOG_KEY] = []

active_flags = [f for f in flags if "Actif" in f["Statut"]]
if active_flags:
    import datetime
    for f in active_flags:
        st.session_state[RETRAINING_LOG_KEY].append({
            **f,
            "timestamp": datetime.datetime.now().isoformat(),
            "firm_id": user.get("firm_id"),
            "username": user.get("username"),
        })

# Display flag table
st.dataframe(flags_df, use_container_width=True, hide_index=True)

if active_flags:
    st.warning(
        f"⚠️ **{len(active_flags)} drapeau(x) de réentraînement actif(s).**  "
        "Ces événements ont été consignés dans le journal de session."
    )

    with st.expander("📋 Journal de réentraînement (session courante)"):
        log_df = pd.DataFrame(st.session_state[RETRAINING_LOG_KEY])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
else:
    st.success("✅ Le modèle causal ne nécessite pas de réentraînement immédiat.")

# ---------------------------------------------------------------------------
# Section 5 — Model Calibration Plot
# ---------------------------------------------------------------------------
st.divider()
st.subheader("5. Calibration du modèle — Probabilités prédites vs. observées")

st.markdown("""
Un modèle bien calibré produit des probabilités prédites proches des fréquences observées.
Les déviations indiquent un sous- ou sur-ajustement systématique.
""")

try:
    pred_proba = ext_model.predict(df)
    cal_df = pd.DataFrame({"pred": pred_proba, "obs": df["y"].values})
    cal_df["decile"] = pd.qcut(cal_df["pred"], q=10, labels=False, duplicates="drop")
    cal_summary = cal_df.groupby("decile").agg(
        mean_pred=("pred", "mean"),
        mean_obs=("obs", "mean"),
        n=("obs", "count"),
    ).reset_index()

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(dash="dash", color="#9CA3AF"),
        name="Calibration parfaite",
    ))
    fig_cal.add_trace(go.Scatter(
        x=cal_summary["mean_pred"],
        y=cal_summary["mean_obs"],
        mode="markers+lines",
        marker=dict(size=10, color="#60A5FA"),
        line=dict(color="#60A5FA"),
        name="Modèle étendu",
    ))
    fig_cal.update_layout(
        title="Courbe de calibration (déciles de probabilité prédite)",
        xaxis_title="Probabilité prédite moyenne",
        yaxis_title="Proportion observée (attrition)",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_cal, use_container_width=True)

except Exception as exc:
    st.warning(f"Impossible de tracer la courbe de calibration : {exc}")

# ---------------------------------------------------------------------------
# Final epistemic caveat
# ---------------------------------------------------------------------------
st.divider()
st.markdown("""
<div style="
    background: #0D1117;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.85rem;
    color: #8B949E;
    line-height: 1.7;
">
<strong style="color:#C9D1D9;">📌 Note épistémique finale</strong><br>
Ce pipeline constitue un <em>placeholder analytique</em> destiné à guider l'exploration,
non à établir des faits causaux. Les modèles de régression logistique supposent une relation
linéaire entre les log-odds et les covariables, une séparabilité parfaite est absente,
et l'absence de variables confondantes non mesurées — toutes des hypothèses
<strong>non testables</strong> sur ces données d'enquête agrégées.<br><br>
Les p-values affichées sont des <strong>indicateurs directionnels plutôt que des probabilités précises</strong>.
Elles ne doivent pas être interprétées comme confirmant ou infirmant une hypothèse causale spécifique.
</div>
""", unsafe_allow_html=True)
