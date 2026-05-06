import streamlit as st

st.set_page_config(page_title="Secure HR App", layout="wide")

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
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from utils.auth import login as auth_login
            
            success, msg = auth_login(username, password)
            if success:
                st.session_state["authenticated"] = True
                user = st.session_state["hrv_user"]
                # map hr_manager -> HR Manager for compatibility with sidebar text/routing
                raw_role = user["role"]
                mapped_role = "HR Manager" if raw_role == "hr_manager" else raw_role.capitalize()
                st.session_state["role"] = mapped_role
                st.session_state["name"] = user["display_name"]
                st.rerun()
            else:
                st.error(msg)

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
