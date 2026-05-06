import streamlit as st
import pandas as pd
import os

st.title("Demo App (IBM Attrition Dashboard)")

@st.cache_data(max_entries=1)
def load_data():
    # Use relative path assuming we run from root directory
    csv_path = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)

df = load_data()

if df.empty:
    st.error("Data file not found. Please ensure data/WA_Fn-UseC_-HR-Employee-Attrition.csv exists.")
else:
    st.write("### Raw Data Preview")
    st.dataframe(df.head())
    
    st.write("### Calculated Metrics")
    
    # In the IBM attrition dataset, standard rating columns (like EnvironmentSatisfaction, JobSatisfaction) 
    # use a 1-4 scale. If there are any 0s (for "Je ne sais pas" or missing data), we MUST exclude them 
    # from the denominator/averages.
    
    # Example metric calculations excluding 0s
    if 'JobSatisfaction' in df.columns:
        # Filter out 0s before calculating mean
        valid_satisfaction = df[df['JobSatisfaction'] != 0]['JobSatisfaction']
        avg_job_satisfaction = valid_satisfaction.mean() if not valid_satisfaction.empty else 0
        st.metric("Avg Job Satisfaction", f"{avg_job_satisfaction:.2f}")
    
    if 'EnvironmentSatisfaction' in df.columns:
        valid_env = df[df['EnvironmentSatisfaction'] != 0]['EnvironmentSatisfaction']
        avg_env_satisfaction = valid_env.mean() if not valid_env.empty else 0
        st.metric("Avg Environment Satisfaction", f"{avg_env_satisfaction:.2f}")
