import streamlit as st

st.set_page_config(page_title="Secure HR App", layout="wide")

# Hardcoded static dictionary for RBAC to save time
USERS = {
    "employee1": {"password": "password123", "role": "Employee", "name": "Alice Employee"},
    "manager1": {"password": "password123", "role": "HR Manager", "name": "Bob Manager"},
    "admin1": {"password": "password123", "role": "Admin", "name": "Charlie Admin"}
}

def login():
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username in USERS and USERS[username]["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["role"] = USERS[username]["role"]
                st.session_state["name"] = USERS[username]["name"]
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
    live_page = st.Page("pages/live_app.py", title="Live App (Survey)", icon="📝")
    demo_page = st.Page("pages/demo_app.py", title="Demo App (Dashboard)", icon="📊")
    
    # Route based on role
    if role == "Employee":
        pg = st.navigation([live_page])
    elif role in ["HR Manager", "Admin"]:
        pg = st.navigation([live_page, demo_page])
    else:
        pg = st.navigation([])
        
    pg.run()
