import streamlit as st
import sqlite3
import pandas as pd
import sys
from pathlib import Path

# Load the full survey structure
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pdf_generator import SURVEY_STRUCTURE

st.title("Employee Survey")

# Flatten questions to know the total count
ALL_QUESTIONS = []
for pillar, qs in SURVEY_STRUCTURE:
    ALL_QUESTIONS.extend(qs)

@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect('hr_valais_live.db', check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    # We create a new table 'survey_responses_full' to accommodate all 33 questions, 
    # preventing schema issues with the old table.
    columns = ", ".join([f"q{i+1}_rating INTEGER" for i in range(len(ALL_QUESTIONS))])
    conn.execute(f'''
        CREATE TABLE IF NOT EXISTS survey_responses_full (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            {columns},
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

conn = get_db_connection()

st.write("Please fill out the following survey. Scale: 1 (Faible) to 4 (Optimal). Select 0 for 'Je ne sais pas'.")

with st.form("survey_form", clear_on_submit=True):
    responses = {}
    
    q_index = 1
    for pillar, questions in SURVEY_STRUCTURE:
        st.subheader(pillar)
        for q in questions:
            responses[f"q{q_index}"] = st.radio(
                f"{q_index}. {q}", 
                options=[0, 1, 2, 3, 4], 
                index=0, 
                format_func=lambda x: "0 (Je ne sais pas)" if x == 0 else str(x),
                key=f"radio_q{q_index}",
                horizontal=True
            )
            q_index += 1
            
    submitted = st.form_submit_button("Submit Survey", use_container_width=True)
    
    if submitted:
        cursor = conn.cursor()
        
        # Prepare dynamic INSERT statement
        cols = ["user_name"] + [f"q{i+1}_rating" for i in range(len(ALL_QUESTIONS))]
        placeholders = ", ".join(["?"] * len(cols))
        
        values = [st.session_state.get('name', 'Anonymous')] + [responses[f"q{i+1}"] for i in range(len(ALL_QUESTIONS))]
        
        cursor.execute(f"INSERT INTO survey_responses_full ({', '.join(cols)}) VALUES ({placeholders})", values)
        conn.commit()
        st.success("🎉 Survey submitted successfully! Thank you.")

st.divider()
st.subheader("Live Survey Statistics")

# Dynamically generate the averages query
avg_queries = ",\n        ".join([f"AVG(NULLIF(q{i+1}_rating, 0)) as avg_q{i+1}" for i in range(len(ALL_QUESTIONS))])
stats_query = f"""
    SELECT 
        COUNT(*) as total_responses,
        {avg_queries}
    FROM survey_responses_full
"""

try:
    df_stats = pd.read_sql_query(stats_query, conn)
    st.write(f"**Total responses:** {df_stats['total_responses'][0]}")
    
    # Render averages in an expander
    with st.expander("📊 View Average Scores per Question"):
        avg_data = []
        q_idx = 1
        for pillar, questions in SURVEY_STRUCTURE:
            for q in questions:
                avg_val = df_stats[f'avg_q{q_idx}'][0]
                avg_data.append({
                    "Pillar": pillar,
                    "Question": q,
                    "Average Score": round(avg_val, 2) if pd.notna(avg_val) else None
                })
                q_idx += 1
                
        df_display = pd.DataFrame(avg_data)
        st.dataframe(df_display, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching stats: {e}")
