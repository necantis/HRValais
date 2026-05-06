import streamlit as st
import sqlite3
import pandas as pd

st.title("Employee Survey")

@st.cache_resource(check_same_thread=False)
def get_db_connection():
    conn = sqlite3.connect('hr_valais_live.db', check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    # Initialize table if it doesn't exist
    conn.execute('''
        CREATE TABLE IF NOT EXISTS survey_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            q1_rating INTEGER,
            q2_rating INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

conn = get_db_connection()

st.write("Please fill out the following survey. Scale: 1 (Poor) to 4 (Excellent). Select 0 for 'Je ne sais pas'.")

with st.form("survey_form", clear_on_submit=True):
    q1 = st.radio("How satisfied are you with the company culture?", options=[0, 1, 2, 3, 4], index=0, format_func=lambda x: "0 (Je ne sais pas)" if x == 0 else str(x))
    q2 = st.radio("How would you rate your work-life balance?", options=[0, 1, 2, 3, 4], index=0, format_func=lambda x: "0 (Je ne sais pas)" if x == 0 else str(x))
    
    submitted = st.form_submit_button("Submit Survey")
    
    if submitted:
        # Batch database write inside form submission using parameterized query
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO survey_responses (user_name, q1_rating, q2_rating) VALUES (?, ?, ?)", 
            (st.session_state.get('name', 'Anonymous'), q1, q2)
        )
        conn.commit()
        st.success("Survey submitted successfully! Thank you.")

st.divider()
st.subheader("Live Survey Statistics")

# Read data excluding 0 from average calculations in SQL
# We use NULLIF(column, 0) so that 0 becomes NULL and is ignored by AVG()
stats_query = """
    SELECT 
        COUNT(*) as total_responses,
        AVG(NULLIF(q1_rating, 0)) as avg_q1,
        AVG(NULLIF(q2_rating, 0)) as avg_q2
    FROM survey_responses
"""

try:
    df_stats = pd.read_sql_query(stats_query, conn)
    st.write(f"Total responses: {df_stats['total_responses'][0]}")
    st.write(f"Average Q1 Rating: {df_stats['avg_q1'][0]:.2f}" if pd.notna(df_stats['avg_q1'][0]) else "Average Q1 Rating: N/A")
    st.write(f"Average Q2 Rating: {df_stats['avg_q2'][0]:.2f}" if pd.notna(df_stats['avg_q2'][0]) else "Average Q2 Rating: N/A")
except Exception as e:
    st.error(f"Error fetching stats: {e}")

