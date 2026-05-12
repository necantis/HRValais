"""
pages/1_Survey.py
Employee survey screen — 32 Likert items across 7 HR Valais dimensions.
Scale: 0 = Je ne sais pas | 1 = Faible | 2 = Partiel | 3 = Avancé | 4 = Optimal
Access: employee role only.
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.auth import require_role, get_current_user
from utils.pdf_generator import generate_survey_pdf, SURVEY_STRUCTURE, URL_MAPPING

require_role("employee", "hr_manager", "admin")
user = get_current_user()

# ---------------------------------------------------------------------------
# Submission limit check
# ---------------------------------------------------------------------------
from sqlalchemy import extract
from db.database import get_session
from db.models import SurveyResponse

current_year = datetime.utcnow().year
with get_session() as session:
    submissions_this_year = session.query(SurveyResponse).filter(
        SurveyResponse.user_id == user["user_id"],
        extract('year', SurveyResponse.timestamp) == current_year
    ).count()

max_surveys = user.get("max_surveys_per_year", 1)
if submissions_this_year >= max_surveys:
    st.warning(f"Vous avez déjà soumis {submissions_this_year} sondage(s) cette année. La limite est de {max_surveys}.")
    st.stop()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("📋 Sondage HR Valais — Fiches pratiques")
st.caption(f"Bonjour {user['display_name']} · {datetime.now().strftime('%d %B %Y')}")

# ---------------------------------------------------------------------------
# Inline CSS — clean radio groups + fiche link badges
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Compact horizontal radio buttons */
div[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: row;
    gap: 8px;
    flex-wrap: wrap;
}
div[data-testid="stRadio"] label {
    padding: 4px 14px;
    border-radius: 20px;
    border: 1.5px solid #2D3748;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 500;
    transition: background 0.18s, border-color 0.18s;
    white-space: nowrap;
}
div[data-testid="stRadio"] label:hover {
    border-color: #4F8EF7;
    background: #1a2744;
}
/* Fiche badge link */
.fiche-link {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    color: #63B3ED;
    text-decoration: none;
    border: 1px solid #2D3748;
    border-radius: 6px;
    padding: 2px 8px;
    margin-top: 2px;
    margin-bottom: 8px;
    transition: background 0.15s;
}
.fiche-link:hover { background: #1A365D; }
/* Question label styling */
.q-label {
    font-size: 0.92rem;
    font-weight: 500;
    color: #E2E8F0;
    line-height: 1.45;
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PDF viewer (optional)
# ---------------------------------------------------------------------------
pdf_path = generate_survey_pdf()

with st.expander("📄 Consulter les Fiches pratiques HR Valais (PDF)", expanded=False):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    import base64
    b64 = base64.b64encode(pdf_bytes).decode()
    components.html(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="600px" style="border:none;border-radius:8px;"></iframe>',
        height=620,
    )
    st.download_button(
        "⬇️ Télécharger le PDF",
        data=pdf_bytes,
        file_name="HR_Valais_Fiches_pratiques.pdf",
        mime="application/pdf",
    )

st.divider()

if "missing_questions" not in st.session_state:
    st.session_state["missing_questions"] = []

if st.session_state["missing_questions"]:
    st.error(f"❌ Impossible de soumettre. Vous avez oublié de répondre à **{len(st.session_state['missing_questions'])}** question(s). Elles sont surlignées en rouge ci-dessous.")

st.subheader("Répondez au sondage annuel")
st.markdown(
    "Évaluez chaque affirmation sur une échelle de **1** (faible) à **4** (optimal). "
    "Choisissez **0 — Je ne sais pas** si vous ne pouvez pas évaluer l'affirmation."
)

# ---------------------------------------------------------------------------
# Scale definition
# ---------------------------------------------------------------------------
SCALE_OPTIONS = [
    "0 — Je ne sais pas",
    "1 — Faible",
    "2 — Partiel",
    "3 — Avancé",
    "4 — Optimal",
]
SCALE_VALUES: dict[str, int] = {opt: int(opt[0]) for opt in SCALE_OPTIONS}
DEFAULT_OPTION = "0 — Je ne sais pas"

PILLAR_ICONS = ["🤝", "🎓", "🏆", "💶", "🌿", "⚖️", "🌐"]

# ---------------------------------------------------------------------------
# Survey form
# ---------------------------------------------------------------------------
answers: dict[str, int] = {}

with st.form("survey_form"):
    q_num = 1
    for idx, (pillar, questions) in enumerate(SURVEY_STRUCTURE):
        icon = PILLAR_ICONS[idx]
        with st.expander(f"{icon} Pilier {idx+1} : {pillar}", expanded=True):
            for q_text in questions:
                key = f"q{q_num}"
                fiche_url = URL_MAPPING.get(q_text, "#")

                is_missing = key in st.session_state["missing_questions"]
                q_color = "#FC8181" if is_missing else "#E2E8F0"

                # Question label
                st.markdown(
                    f'<p class="q-label" style="color: {q_color}; font-weight: {"700" if is_missing else "500"};">{q_num}. {q_text}</p>',
                    unsafe_allow_html=True,
                )

                # Horizontal radio — label hidden (question already shown above)
                val = st.radio(
                    label=f"_{key}",        # hidden by label_visibility
                    options=SCALE_OPTIONS,
                    index=None,            # Force user to explicitly choose
                    horizontal=True,
                    key=key,
                    label_visibility="collapsed",
                )
                if val is not None:
                    answers[key] = SCALE_VALUES[val]
                q_num += 1

    st.divider()
    free_text = st.text_area(
        "💬 Commentaire libre (optionnel)",
        placeholder="Partagez vos remarques ou suggestions…",
        max_chars=1000,
    )

    if st.session_state["missing_questions"]:
        st.error("⚠️ Il manque des réponses. Veuillez corriger les questions en rouge ci-dessus avant de soumettre.")

    submitted = st.form_submit_button("✅ Soumettre mes réponses", use_container_width=True)

# ---------------------------------------------------------------------------
# On submit — compute averages (excluding 0) and persist
# ---------------------------------------------------------------------------
if submitted:
    # Validation: Ensure all 33 questions are answered
    missing = [f"q{i}" for i in range(1, 34) if f"q{i}" not in answers]
    if missing:
        st.session_state["missing_questions"] = missing
        st.rerun()
    else:
        st.session_state["missing_questions"] = []
        
    from db.database import get_session
    from db.models import SurveyResponse

    def _pil_avg(*keys: str) -> float | None:
        """Average of non-zero answers; None if all are 0 (Je ne sais pas)."""
        vals = [answers[k] for k in keys if answers.get(k, 0) != 0]
        return float(np.mean(vals)) if vals else None

    # Map question numbers to DB columns
    # Pillar 1: q1–q6   (Recrutement)
    # Pillar 2: q7–q10  (Compétences)
    # Pillar 3: q11–q14 (Performance)
    # Pillar 4: q15–q18 (Rémunération)
    # Pillar 5: q19–q24 (QVT)
    # Pillar 6: q25–q29 (Droit)
    # Pillar 7: q30–q33 (Transverses)

    response = SurveyResponse(
        response_id=str(uuid.uuid4()),
        user_id=user["user_id"],
        firm_id=user["firm_id"] or "00000000-0000-0000-0000-000000000000",
        timestamp=datetime.utcnow(),

        # Pillar 1 — Recrutement
        recrutement_q1=answers["q1"],
        recrutement_q2=answers["q2"],
        recrutement_q3=answers["q3"],
        recrutement_q4=answers["q4"],
        recrutement_q5=answers["q5"],
        recrutement_q6=answers["q6"],

        # Pillar 2 — Compétences
        competences_q7=answers["q7"],
        competences_q8=answers["q8"],
        competences_q9=answers["q9"],
        competences_q10=answers["q10"],

        # Pillar 3 — Performance
        performance_q11=answers["q11"],
        performance_q12=answers["q12"],
        performance_q13=answers["q13"],
        performance_q14=answers["q14"],

        # Pillar 4 — Rémunération
        remuneration_q15=answers["q15"],
        remuneration_q16=answers["q16"],
        remuneration_q17=answers["q17"],
        remuneration_q18=answers["q18"],

        # Pillar 5 — QVT
        qvt_q19=answers["q19"],
        qvt_q20=answers["q20"],
        qvt_q21=answers["q21"],
        qvt_q22=answers["q22"],
        qvt_q23=answers["q23"],
        qvt_q24=answers["q24"],

        # Pillar 6 — Droit
        droit_q25=answers["q25"],
        droit_q26=answers["q26"],
        droit_q27=answers["q27"],
        droit_q28=answers["q28"],
        droit_q29=answers["q29"],

        # Pillar 7 — Transverses
        transverse_q30=answers["q30"],
        transverse_q31=answers["q31"],
        transverse_q32=answers["q32"],
        transverse_q33=answers["q33"],

        # Averages (exclude Je ne sais pas = 0)
        recrutement_avg=_pil_avg("q1","q2","q3","q4","q5","q6"),
        competences_avg=_pil_avg("q7","q8","q9","q10"),
        performance_avg=_pil_avg("q11","q12","q13","q14"),
        remuneration_avg=_pil_avg("q15","q16","q17","q18"),
        qvt_avg=_pil_avg("q19","q20","q21","q22","q23","q24"),
        droit_avg=_pil_avg("q25","q26","q27","q28","q29"),
        transverse_avg=_pil_avg("q30","q31","q32","q33"),

        free_text_feedback=free_text or None,
    )

    with get_session() as session:
        session.add(response)
        session.commit()

    st.success("🎉 Merci ! Vos réponses ont été enregistrées avec succès.")
    st.balloons()
