"""
pages/1_Survey.py
Employee survey screen — 21 Likert items across 7 HR Valais pillars.
Access: employee role only.
"""

import sys
from pathlib import Path
import uuid
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.auth import require_role, get_current_user
from utils.pdf_generator import generate_survey_pdf, SURVEY_STRUCTURE

require_role("employee")
user = get_current_user()

st.title("📋 Sondage HR Valais — Fiches pratiques")
st.caption(f"Bonjour {user['display_name']} · {datetime.now().strftime('%d %B %Y')}")

# ---------------------------------------------------------------------------
# Generate and embed the PDF
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
st.subheader("Répondez au sondage annuel")
st.markdown(
    "Évaluez chaque affirmation sur une échelle de **1** (pas du tout d'accord) à **5** (tout à fait d'accord)."
)

PILLAR_ICONS = ["🤝", "🎓", "🏆", "💶", "🌿", "⚖️", "🌐"]
LIKERT = {"1 — Pas du tout": 1, "2 — Plutôt non": 2, "3 — Neutre": 3,
          "4 — Plutôt oui": 4, "5 — Tout à fait": 5}
LIKERT_OPTIONS = list(LIKERT.keys())

answers: dict[str, int] = {}

with st.form("survey_form"):
    q_num = 1
    for idx, (pillar, questions) in enumerate(SURVEY_STRUCTURE):
        icon = PILLAR_ICONS[idx]
        with st.expander(f"{icon} Pilier {idx+1} : {pillar}", expanded=True):
            for q_text in questions:
                key = f"q{q_num}"
                val = st.select_slider(
                    label=q_text,
                    options=LIKERT_OPTIONS,
                    value="3 — Neutre",
                    key=key,
                )
                answers[key] = LIKERT[val]
                q_num += 1

    free_text = st.text_area(
        "💬 Commentaire libre (optionnel)",
        placeholder="Partagez vos remarques ou suggestions…",
        max_chars=1000,
    )

    submitted = st.form_submit_button("✅ Soumettre mes réponses", use_container_width=True)

if submitted:
    from db.database import get_session
    from db.models import SurveyResponse
    import numpy as np

    def pil_avg(*keys): return float(np.mean([answers[k] for k in keys]))

    response = SurveyResponse(
        response_id=str(uuid.uuid4()),
        user_id=user["user_id"],
        firm_id=user["firm_id"] or "00000000-0000-0000-0000-000000000000",
        timestamp=datetime.utcnow(),
        recrutement_q1=answers["q1"], recrutement_q2=answers["q2"], recrutement_q3=answers["q3"],
        competences_q4=answers["q4"], competences_q5=answers["q5"], competences_q6=answers["q6"],
        performance_q7=answers["q7"], performance_q8=answers["q8"], performance_q9=answers["q9"],
        remuneration_q10=answers["q10"], remuneration_q11=answers["q11"], remuneration_q12=answers["q12"],
        qvt_q13=answers["q13"], qvt_q14=answers["q14"], qvt_q15=answers["q15"],
        droit_q16=answers["q16"], droit_q17=answers["q17"], droit_q18=answers["q18"],
        transverse_q19=answers["q19"], transverse_q20=answers["q20"], transverse_q21=answers["q21"],
        recrutement_avg=pil_avg("q1","q2","q3"),
        competences_avg=pil_avg("q4","q5","q6"),
        performance_avg=pil_avg("q7","q8","q9"),
        remuneration_avg=pil_avg("q10","q11","q12"),
        qvt_avg=pil_avg("q13","q14","q15"),
        droit_avg=pil_avg("q16","q17","q18"),
        transverse_avg=pil_avg("q19","q20","q21"),
        free_text_feedback=free_text or None,
    )

    with get_session() as session:
        session.add(response)
        session.commit()

    st.success("🎉 Merci ! Vos réponses ont été enregistrées avec succès.")
    st.balloons()
