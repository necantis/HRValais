"""
scratch/validate_survey.py  — quick structural validation
"""
import sys
sys.path.insert(0, '.')

from utils.pdf_generator import SURVEY_STRUCTURE, URL_MAPPING

total_q = sum(len(qs) for _, qs in SURVEY_STRUCTURE)
print(f"Total questions: {total_q}")

for idx, (pillar, qs) in enumerate(SURVEY_STRUCTURE, 1):
    print(f"  Pillar {idx} ({pillar}): {len(qs)} questions")

missing = [q for _, qs in SURVEY_STRUCTURE for q in qs if q not in URL_MAPPING]
if missing:
    print("MISSING URL mappings:")
    for m in missing:
        print(f"  - {m}")
else:
    print("All questions have URL mappings. OK")

# DB column check
import sqlite3
conn = sqlite3.connect('hr_valais_prototype.db')
cur = conn.cursor()
cur.execute('PRAGMA table_info(survey_responses)')
cols = {row[1] for row in cur.fetchall()}
conn.close()

q_cols = sorted(c for c in cols if any(
    prefix in c for prefix in
    ['recrutement_q','competences_q','performance_q','remuneration_q','qvt_q','droit_q','transverse_q']
))
print(f"\nDB question columns ({len(q_cols)}): {q_cols}")
print("PASS" if len(q_cols) >= total_q else "FAIL: column count mismatch")
