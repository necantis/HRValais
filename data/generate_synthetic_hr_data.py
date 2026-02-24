import pandas as pd
import numpy as np
from faker import Faker
import uuid

fake = Faker('fr_CH') # Using Swiss French locale for Valais
Faker.seed(42)
np.random.seed(42)

# --- Configuration ---
NUM_EMPLOYEES = 200
MONTHS = 36
FIRM_ID = str(uuid.uuid4())

# Core states for the Markov Chain
STATES = ["Highly Engaged", "Content", "Passively Looking", "Resigned"]

# Reference Class Transition Matrix (Calibrated to an estimated baseline)
# Probabilities of moving from Row(State A) to Column(State B)
TRANSITION_MATRIX = [
    [0.85, 0.10, 0.04, 0.01], # From Highly Engaged
    [0.15, 0.70, 0.10, 0.05], # From Content
    [0.05, 0.15, 0.60, 0.20], # From Passively Looking
    [0.00, 0.00, 0.00, 1.00]  # From Resigned (Terminal state)
]

data = []

for _ in range(NUM_EMPLOYEES):
    emp_id = str(uuid.uuid4())
    # Assign baseline static demographic data (mapping to OFS variables)
    age = np.random.randint(20, 65)
    position = np.random.choice(["Execution", "Upper Management"], p=[0.8, 0.2])
    gender = np.random.choice(["Male", "Female"])
    
    current_state = np.random.choice(STATES, p=[0.5, 0.3, 0.2, 0.0]) # Initial distribution
    
    for month in range(1, MONTHS + 1):
        # Generate the 7 HR Valais Pillar Scores (1-5) influenced by the current state
        state_modifier = {"Highly Engaged": 1, "Content": 0, "Passively Looking": -1, "Resigned": -2}[current_state]
        
        # Ensure scores stay within the 1-5 Likert range
        scores = {
            "recrutement_avg": max(1, min(5, np.random.normal(4 + state_modifier, 0.5))),
            "competences_avg": max(1, min(5, np.random.normal(3.5 + state_modifier, 0.8))),
            "performance_avg": max(1, min(5, np.random.normal(3.8 + state_modifier, 0.6))),
            "remuneration_avg": max(1, min(5, np.random.normal(3.2 + state_modifier, 0.9))),
            "qvt_avg": max(1, min(5, np.random.normal(4 + state_modifier, 0.7))),
            "droit_avg": max(1, min(5, np.random.normal(4.2 + state_modifier, 0.4))),
            "transverse_avg": max(1, min(5, np.random.normal(3.5 + state_modifier, 0.8)))
        }
        
        data.append({
            "month_index": month,
            "employee_id": emp_id,
            "firm_id": FIRM_ID,
            "age": age,
            "position": position,
            "gender": gender,
            "state": current_state,
            **scores
        })
        
        # Calculate next state if the employee hasn't resigned
        if current_state != "Resigned":
            current_state_idx = STATES.index(current_state)
            current_state = np.random.choice(STATES, p=TRANSITION_MATRIX[current_state_idx])

df = pd.DataFrame(data)
df.to_csv("data/synthetic_longitudinal_hr.csv", index=False)
print("✅ synthetic_longitudinal_hr.csv generated successfully.")