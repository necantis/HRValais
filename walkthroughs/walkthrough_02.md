# Secure Streamlit Application Scaffolding

We have successfully scaffolded the new multipage architecture using Streamlit's latest `st.navigation` and `st.Page` mechanics.

## Implementation Details

### Core Application Entry (`app.py`)
- We replaced the previous architecture with a streamlined `app.py`.
- **Role-Based Access Control (RBAC)**: A static, hardcoded dictionary (`USERS`) handles authentication in `st.session_state` without latency. Roles defined include Employee, HR Manager, and Admin.
- **Routing**: `st.navigation()` is dynamically populated. Employees only see the live survey app, whereas managers and admins also see the demo data dashboard.

### Live App (`pages/live_app.py`)
- **Database Architecture**: Implemented a local SQLite connection to `hr_valais_live.db`.
- **Caching & Tuning**: Wrapped the connection in `@st.cache_resource(check_same_thread=False)` to prevent concurrency locks across sessions, immediately executing `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;` for high-throughput disk writes.
- **Data Integrity**: 
  - Submissions are wrapped in an `st.form` to ensure grouped logic.
  - Inserts use robust parameterized SQL tuples.
  - Survey metrics explicitly enforce the data constraint: using `AVG(NULLIF(q_rating, 0))` prevents answers of `0` ("Je ne sais pas") from dragging down the average.

### Demo App (`pages/demo_app.py`)
- **Data Loading**: Pandas reads the IBM Attrition CSV directly, cached efficiently via `@st.cache_data(max_entries=1)` to avoid memory leaks (and without `.clear()`).
- **Data Constraints**: Implemented pandas masks `df[df['Col'] != 0]` to actively strip out `0` from calculations before calling `.mean()`.

## Verification Status
- [x] Application successfully routes depending on the user credentials used.
- [x] Streamlit data schemas follow the explicitly requested handling of 0.
