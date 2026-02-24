"""
app.py — HR Valais Multi-Tenant Streamlit Application entry point.

Responsibilities:
  1. Initialise the SQLite DB (once, idempotent).
  2. Render a login screen if the user is not authenticated.
  3. Route authenticated users to their role-specific pages via st.navigation.
  4. Render a sidebar with role badge, firm name, and logout button.
"""

import logging
import sys
from pathlib import Path

import streamlit as st

# Make project root importable regardless of working directory
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Valais",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB init (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initialising database…")
def _init_db():
    from db.database import init_db
    init_db()
    return True

_init_db()

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
from utils.auth import get_current_user, login, logout, get_role  # noqa: E402

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar style */
section[data-testid="stSidebar"] { background: #1C2333; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

/* Role badge */
.role-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-employee   { background: #2D3748; color: #A0AEC0 !important; }
.badge-hr_manager { background: #1A365D; color: #63B3ED !important; }
.badge-admin      { background: #1C4532; color: #68D391 !important; }

/* Card-style metric boxes */
div[data-testid="metric-container"] {
    background: #1C2333;
    border: 1px solid #2D3748;
    border-radius: 10px;
    padding: 12px 16px;
}

/* Warning caveat boxes */
.caveat-box {
    background: #2D1F0E;
    border-left: 4px solid #F6AD55;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 0.88rem;
    color: #F6E05E;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Login screen
# ---------------------------------------------------------------------------
def _render_login():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Flag_of_Switzerland_%28Pantone%29.svg/240px-Flag_of_Switzerland_%28Pantone%29.svg.png", width=60)
        st.markdown("## 🏔️ HR Valais")
        st.markdown("*Plateforme RH multi-tenant — Prototype*")
        st.divider()

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Nom d'utilisateur", placeholder="ex: hr_manager_a")
            password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button("Connexion →", use_container_width=True)

        if submitted:
            ok, err = login(username, password)
            if ok:
                st.rerun()
            else:
                st.error(f"❌ {err}")

        with st.expander("🔑 Comptes de démonstration"):
            st.markdown("""
| Utilisateur | Mot de passe | Rôle |
|-------------|--------------|------|
| `employee1` | `password123` | Employé (Firm A) |
| `employee2` | `password123` | Employé (Firm A) |
| `hr_manager_a` | `password123` | RH Manager (Firm A) |
| `hr_manager_b` | `password123` | RH Manager (Firm B) |
| `admin` | `admin_secure_2026` | Administrateur |
""")


# ---------------------------------------------------------------------------
# Sidebar (shown when authenticated)
# ---------------------------------------------------------------------------
def _render_sidebar(user: dict):
    with st.sidebar:
        st.markdown("### 🏔️ HR Valais")
        st.divider()

        role = user["role"]
        badge_cls = f"badge-{role}"
        role_labels = {
            "employee": "Employé",
            "hr_manager": "RH Manager",
            "admin": "Administrateur",
        }
        st.markdown(
            f'<span class="role-badge {badge_cls}">{role_labels.get(role, role)}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**{user['display_name']}**")
        if user.get("firm_name"):
            st.caption(f"🏢 {user['firm_name']}")

        st.divider()
        if st.button("🚪 Déconnexion", use_container_width=True):
            logout()
            st.rerun()


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
def _get_pages_for_role(role: str) -> list:
    """Return the st.Page objects allowed for this role."""
    survey_page = st.Page("pages/1_Survey.py", title="📋 Sondage RH", icon="📋")
    dash1_page  = st.Page("pages/2_Dashboard_Internal.py",   title="📊 Tableau de bord interne",  icon="📊")
    dash2_page  = st.Page("pages/3_Dashboard_Benchmarking.py", title="📈 Benchmarking OFS",        icon="📈")
    dash3_page  = st.Page("pages/4_Dashboard_MixedModels.py",  title="🔬 Modèles mixtes",          icon="🔬")
    dash4_page  = st.Page("pages/5_Dashboard_Timeseries.py",   title="⏱️ Séries temporelles",      icon="⏱️")
    upload_page = st.Page("pages/6_Upload.py",  title="📤 Importer des données", icon="📤")
    admin_page  = st.Page("pages/7_Admin.py",   title="⚙️ Administration",       icon="⚙️")

    if role == "employee":
        return [survey_page]
    elif role == "hr_manager":
        return [dash1_page, dash2_page, dash3_page, dash4_page, upload_page]
    elif role == "admin":
        return [admin_page, dash2_page]
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
user = get_current_user()

if user is None:
    _render_login()
else:
    _render_sidebar(user)
    pages = _get_pages_for_role(user["role"])
    if pages:
        nav = st.navigation(pages)
        nav.run()
    else:
        st.error("Aucune page disponible pour ce rôle.")
