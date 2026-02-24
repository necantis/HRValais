"""
pages/6_Upload.py
HR Manager file upload — monthly CSV ingestion via st.chat_input(accept_file=True).
Access: hr_manager only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.auth import require_role, get_current_user
from db.database import get_session
from db.models import MonthlyUpload, SurveyResponse

require_role("hr_manager")
user = get_current_user()

st.title("📤 Importer des données RH mensuelles")
st.caption(f"Entreprise : **{user['firm_name']}**")

st.markdown("""
Déposez un fichier CSV mensuel contenant les scores de piliers HR Valais.

**Format attendu** : colonnes `recrutement_avg`, `competences_avg`, `performance_avg`,
`remuneration_avg`, `qvt_avg`, `droit_avg`, `transverse_avg`  
*(optionnel : `month_index`, `age`, `position`, `gender`, `engagement_state`)*
""")

# ---------------------------------------------------------------------------
# Recent uploads history
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def _load_uploads(firm_id: str) -> pd.DataFrame:
    with get_session() as session:
        rows = (
            session.query(MonthlyUpload)
            .filter_by(firm_id=firm_id)
            .order_by(MonthlyUpload.uploaded_at.desc())
            .limit(10)
            .all()
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "Fichier": r.filename,
        "Date import": r.uploaded_at.strftime("%Y-%m-%d %H:%M"),
        "Lignes": r.row_count,
        "ID": r.upload_id[:8] + "…",
    } for r in rows])

history_df = _load_uploads(user["firm_id"])
if not history_df.empty:
    with st.expander("📋 Historique des imports récents", expanded=False):
        st.dataframe(history_df, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Chat-style file upload
# ---------------------------------------------------------------------------
msg = st.chat_input("Tapez un message ou déposez un fichier CSV ici…", accept_file=True)

if msg is not None:
    # Handle text-only messages
    if msg.text and not msg.files:
        with st.chat_message("assistant"):
            st.markdown("Pour importer des données, veuillez joindre un fichier CSV à votre message.")

    elif msg.files:
        uploaded_file = msg.files[0]
        with st.chat_message("user"):
            st.markdown(f"📎 Fichier reçu : **{uploaded_file.name}**")

        with st.chat_message("assistant"):
            try:
                raw_bytes = uploaded_file.read()
                df = pd.read_csv(io.BytesIO(raw_bytes))

                PILLAR_COLS = ["recrutement_avg", "competences_avg", "performance_avg",
                               "remuneration_avg", "qvt_avg", "droit_avg", "transverse_avg"]
                missing = [c for c in PILLAR_COLS if c not in df.columns]

                if missing:
                    st.error(f"❌ Colonnes manquantes : `{', '.join(missing)}`")
                    st.markdown("Veuillez corriger le fichier et réessayer.")
                else:
                    # Show preview
                    st.success(f"✅ Fichier valide — {len(df)} lignes détectées.")
                    st.dataframe(df[PILLAR_COLS].describe().round(2), use_container_width=True)

                    # Save as MonthlyUpload
                    with get_session() as session:
                        upload = MonthlyUpload(
                            upload_id=str(uuid.uuid4()),
                            firm_id=user["firm_id"],
                            uploaded_at=datetime.utcnow(),
                            filename=uploaded_file.name,
                            row_count=len(df),
                            raw_csv=raw_bytes.decode("utf-8", errors="replace"),
                        )
                        session.add(upload)

                        # Append rows to survey_responses
                        PSEUDO_HR = "00000000-0000-0000-0000-000000000099"
                        OPTIONAL = ["month_index", "age", "position", "gender", "engagement_state"]
                        added, skipped = 0, 0
                        for _, row in df.iterrows():
                            try:
                                import numpy as np
                                def col(c):
                                    return float(row[c]) if c in df.columns and pd.notna(row[c]) else None
                                r_avg = col("recrutement_avg")
                                c_avg = col("competences_avg")
                                p_avg = col("performance_avg")
                                rem_avg = col("remuneration_avg")
                                q_avg = col("qvt_avg")
                                d_avg = col("droit_avg")
                                t_avg = col("transverse_avg")
                                if any(v is None for v in [r_avg, c_avg, p_avg, rem_avg, q_avg, d_avg, t_avg]):
                                    skipped += 1
                                    continue

                                def jit(v):
                                    return max(1.0, min(5.0, v + np.random.normal(0, 0.2)))

                                session.add(SurveyResponse(
                                    response_id=str(uuid.uuid4()),
                                    user_id=PSEUDO_HR,
                                    firm_id=user["firm_id"],
                                    timestamp=datetime.utcnow(),
                                    month_index=int(row["month_index"]) if "month_index" in df.columns else None,
                                    age=int(row["age"]) if "age" in df.columns and pd.notna(row.get("age")) else None,
                                    position=str(row["position"]) if "position" in df.columns else None,
                                    gender=str(row["gender"]) if "gender" in df.columns else None,
                                    engagement_state=str(row["engagement_state"]) if "engagement_state" in df.columns else None,
                                    recrutement_avg=r_avg, competences_avg=c_avg,
                                    performance_avg=p_avg, remuneration_avg=rem_avg,
                                    qvt_avg=q_avg, droit_avg=d_avg, transverse_avg=t_avg,
                                    recrutement_q1=jit(r_avg), recrutement_q2=jit(r_avg), recrutement_q3=jit(r_avg),
                                    competences_q4=jit(c_avg), competences_q5=jit(c_avg), competences_q6=jit(c_avg),
                                    performance_q7=jit(p_avg), performance_q8=jit(p_avg), performance_q9=jit(p_avg),
                                    remuneration_q10=jit(rem_avg), remuneration_q11=jit(rem_avg), remuneration_q12=jit(rem_avg),
                                    qvt_q13=jit(q_avg), qvt_q14=jit(q_avg), qvt_q15=jit(q_avg),
                                    droit_q16=jit(d_avg), droit_q17=jit(d_avg), droit_q18=jit(d_avg),
                                    transverse_q19=jit(t_avg), transverse_q20=jit(t_avg), transverse_q21=jit(t_avg),
                                ))
                                added += 1
                            except Exception:
                                skipped += 1

                        session.commit()

                    st.markdown(
                        f"📥 Import terminé : **{added}** lignes ajoutées, **{skipped}** ignorées."
                    )
                    _load_uploads.clear()

            except Exception as e:
                st.error(f"Erreur lors du traitement : {e}")
