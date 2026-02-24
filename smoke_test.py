import sys, traceback, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

try:
    from db.database import init_db, get_session
    from db.models import Firm, User, SurveyResponse, OFSMacroData
    print('IMPORTS OK')
    init_db()
    with get_session() as s:
        print('Firms:', s.query(Firm).count())
        print('Users:', s.query(User).count())
        print('Responses:', s.query(SurveyResponse).count())
        print('OFS:', s.query(OFSMacroData).count())
    print('SEED OK')
    from utils.auth import hash_password, check_password
    h = hash_password('password123')
    assert check_password('password123', h)
    assert not check_password('wrong', h)
    print('AUTH OK')
    from utils.pdf_generator import generate_survey_pdf
    p = generate_survey_pdf()
    print('PDF OK:', p)
    print('ALL TESTS PASSED')
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
