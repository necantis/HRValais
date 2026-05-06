import streamlit as st

st.set_page_config(page_title="Secure HR App", layout="wide")

# Hardcoded static dictionary for RBAC to save time
USERS = {
    "Employee1_firmA": {"password": "password123", "role": "Employee", "name": "Alice (Firm A)", "firm_name": "Alpina Services SA"},
    "Employee1_firmB": {"password": "password123", "role": "Employee", "name": "Bob (Firm B)", "firm_name": "Rhône Industrie Sàrl"},
    "Manager1_firmA": {"password": "password123", "role": "HR Manager", "name": "Chloé (Firm A)", "firm_name": "Alpina Services SA"},
    "Manager1_firmB": {"password": "password123", "role": "HR Manager", "name": "David (Firm B)", "firm_name": "Rhône Industrie Sàrl"},
    "admin1": {"password": "password123", "role": "Admin", "name": "Charlie Admin", "firm_name": "HR Valais (Platform)"}
}

def login():
    st.title("Login")
    
    st.info("""
    **Test Accounts:**
    - **Firm A:** `Employee1_firmA` / `password123` | `Manager1_firmA` / `password123`
    - **Firm B:** `Employee1_firmB` / `password123` | `Manager1_firmB` / `password123`
    - **Admin:** `admin1` / `password123`
    """)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username in USERS and USERS[username]["password"] == password:
                st.session_state["authenticated"] = True
                role = USERS[username]["role"]
                st.session_state["role"] = role
                st.session_state["name"] = USERS[username]["name"]
                
                # Create a mock hrv_user for compatibility with the old dashboards
                mapped_role = "hr_manager" if role == "HR Manager" else role.lower()
                target_firm_name = USERS[username].get("firm_name")
                # Fetch the real firm ID and ensure the user exists in the prototype DB
                real_firm_id = "00000000-0000-0000-0000-000000000001"
                real_user_id = "00000000-0000-0000-0000-000000000000"
                
                try:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent))
                    from db.database import get_session
                    from db.models import Firm, User
                    import uuid
                    
                    with get_session() as session:
                        # 1. Get real firm_id
                        firm = session.query(Firm).filter_by(name=target_firm_name).first()
                        if firm:
                            real_firm_id = firm.firm_id
                            
                        # 2. Get or create real user_id to satisfy Foreign Key constraints
                        user_db = session.query(User).filter_by(username=username).first()
                        if not user_db:
                            real_user_id = str(uuid.uuid4())
                            user_db = User(
                                user_id=real_user_id,
                                firm_id=real_firm_id,
                                username=username,
                                role=mapped_role,
                                hashed_password="mock",
                                display_name=USERS[username]["name"]
                            )
                            session.add(user_db)
                            session.commit()
                        else:
                            real_user_id = user_db.user_id
                except Exception as e:
                    print(f"Error initializing user: {e}")

                st.session_state["hrv_user"] = {
                    "user_id": real_user_id,
                    "username": username,
                    "display_name": USERS[username]["name"],
                    "role": mapped_role,
                    "firm_id": real_firm_id,
                    "firm_name": target_firm_name
                }
                st.rerun()
            else:
                st.error("Invalid username or password")

def logout():
    st.session_state.clear()
    st.rerun()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
else:
    st.sidebar.write(f"Logged in as: **{st.session_state['name']}** ({st.session_state['role']})")
    if st.sidebar.button("Logout"):
        logout()
        
    role = st.session_state["role"]
    
    # Define pages
    survey_page = st.Page("pages/1_Survey.py", title="📋 Sondage RH", icon="📋")
    demo_page = st.Page("pages/demo_app.py", title="Demo App (Dashboard)", icon="📊")
    
    internal_dashboard = st.Page("pages/2_Dashboard_Internal.py", title="Tableau de bord interne", icon="📊")
    benchmarking = st.Page("pages/3_Dashboard_Benchmarking.py", title="Benchmarking OFS", icon="📈")
    mixed_models = st.Page("pages/4_Dashboard_MixedModels.py", title="Modèles mixtes", icon="🧠")
    timeseries = st.Page("pages/5_Dashboard_Timeseries.py", title="Séries temporelles", icon="⏱️")
    upload_page = st.Page("pages/6_Upload.py", title="Import Données", icon="📤")
    admin_page = st.Page("pages/7_Admin.py", title="Administration", icon="⚙️")
    
    # Route based on role
    if role == "Employee":
        pg = st.navigation([survey_page])
    elif role == "HR Manager":
        pg = st.navigation([survey_page, demo_page, internal_dashboard, benchmarking, mixed_models, timeseries, upload_page])
    elif role == "Admin":
        pg = st.navigation([survey_page, demo_page, internal_dashboard, benchmarking, mixed_models, timeseries, upload_page, admin_page])
    else:
        pg = st.navigation([])
        
    pg.run()
