"""Smoke test: verify DB seed and all module imports."""
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

print("=== 1. Testing imports ===")
from db.database import init_db, get_session
from db.models import Firm, User, SurveyResponse, OFSMacroData
from utils.auth import hash_password, check_password
from utils.ofs_parser import load_ofs_data
from utils.kanon import kanonymize
from utils.pdf_generator import generate_survey_pdf
print("All imports OK ✅")

print("\n=== 2. DB init + seed ===")
init_db()

print("\n=== 3. Row counts ===")
s = get_session()
print(f"  Firms:     {s.query(Firm).count()}")
print(f"  Users:     {s.query(User).count()}")
print(f"  Responses: {s.query(SurveyResponse).count()}")
print(f"  OFS rows:  {s.query(OFSMacroData).count()}")

print("\n=== 4. User roles ===")
for u in s.query(User).all():
    fid = u.firm_id[:8] if u.firm_id else "(none)"
    print(f"  {u.username:<20} role={u.role:<12} firm={fid}")
s.close()

print("\n=== 5. Auth smoke test ===")
# Test password hashing
h = hash_password("password123")
assert check_password("password123", h), "Password check failed!"
assert not check_password("wrong", h), "False positive password!"
print("  bcrypt hash/verify OK ✅")

print("\n=== 6. PDF generation ===")
pdf_path = generate_survey_pdf()
print(f"  PDF generated: {pdf_path} ({pdf_path.stat().st_size // 1024} KB) ✅")

print("\n=== ALL SMOKE TESTS PASSED ✅ ===")
