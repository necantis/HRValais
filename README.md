# HR Valais Platform

**HR Valais** is a multi-tenant, Role-Based Access Control (RBAC) survey and behavioral analytics platform built with Streamlit and PostgreSQL (Neon Cloud). It allows companies to assess their HR practices across 7 dimensions (Pillars) and provides managers and global administrators with advanced interactive dashboards, benchmarking, and hypothesis-testing capabilities.

---

## 🤖 Context for AI Coding Agents

This section is written specifically for future AI agents working on this repository to help you get up to speed quickly and avoid common pitfalls.

### Tech Stack & Architecture
- **Frontend / Routing**: Streamlit (`app.py` handles the main routing and session initialization).
- **Database ORM**: SQLAlchemy (`db/models.py`, `db/database.py`).
- **Production Database**: Hosted on Neon Cloud PostgreSQL (Requires `DATABASE_URL` in `.streamlit/secrets.toml`).
- **Development Database**: Falls back to local SQLite if no `DATABASE_URL` is found.
- **Visualizations**: Plotly Express & Plotly Graph Objects (`px` and `go`).

### Directory Structure & Key Files
- `app.py`: The entry point. Handles `st.navigation()`, login forms, role resolution, and the **URL Redirect Telemetry Interceptor**.
- `db/database.py`: Handles engine creation and sessions (prioritizes Neon, falls back to SQLite).
- `db/models.py`: Contains the schema (`Firm`, `User`, `SurveyResponse`, `ActivityLog`, `OFSMacroData`, `MonthlyUpload`).
- `db/seed.py`: Used to initialize the database with mock firms and users.
- `utils/auth.py`: Session-state-based authentication. Stores current user in `st.session_state["hrv_user"]`.
- `utils/pdf_generator.py`: Contains `SURVEY_STRUCTURE` (the 33 questions mapped to 7 pillars) and `URL_MAPPING` (links to HR Valais Fiches Pratiques).

### Pages (Streamlit Routing)
- `pages/1_Survey.py`: The employee interface. **Crucial detail**: The survey enforces explicit answers (`index=None` on radio buttons). Missing questions are stored in `st.session_state["missing_questions"]` and highlighted in red upon validation failure.
- `pages/2_Dashboard_Internal.py`: The internal dashboard. Accessible by `hr_manager` (sees their firm + benchmark) and `admin` (sees all firms overlaid, boxplots, and violin plots).
- `pages/7_Admin.py`: The global admin panel. Uses `st.tabs` for layout. Contains user management, database health, and the **Test Hypothèses** tab.
- *Other Dashboard Pages (`3_...` to `6_...`)*: Various statistical and external data imports (Benchmarking, Mixed Models, Timeseries).

### The 7 HR Dimensions (Pillars)
The survey consists of exactly 33 questions distributed as follows:
1. Recrutement (6 questions)
2. Gestion des compétences (4 questions)
3. Évaluation & Performance (4 questions)
4. Rémunération (4 questions)
5. Qualité de vie au travail (QVT) (6 questions)
6. Droit du travail (5 questions)
7. Thématiques transverses (4 questions)

**Data Processing Rule**: When calculating averages for dimensions, `0` ("Je ne sais pas") is always excluded/replaced with `np.nan` to prevent statistical skewing.

### Telemetry & Analytics (Contingency Theory)
The platform acts as a behavioral analytics tool to test contingency theory (how firms adapt to environments).
- **ActivityLog**: We track `survey_completion_time`, `dashboard_ping`, and `link_click` in the database.
- **Silent URL Tracking**: We do *not* use external JS trackers. Instead, `app.py` intercepts `/?redirect_url=...` query parameters, logs the click to the `ActivityLog` table, and then performs an HTML meta refresh to the destination URL. If you add external links to the dashboard, **always wrap them in the `/?redirect_url=` pattern**.
- **K-Means Clustering**: Located in `pages/7_Admin.py`, it automatically clusters firms based on "Dashboard Intensity" (clicks per visit) and "Global Score".

### ⚠️ Common Pitfalls & Rules for Agents
1. **Never use `index=0` for surveys**: It ruins data integrity. Always use `index=None` and enforce explicit validation.
2. **Heavy Data Loading**: Never put heavy `SELECT *` queries (like loading the entire `OFSMacroData` table) directly in the rendering path of a page or tab. Always wrap them in an `st.button` or use pagination, otherwise the Streamlit page will freeze.
3. **Session State vs Reruns**: Streamlit reruns the whole script on every interaction. If you need to persist a state (like highlighting missing survey questions), use `st.session_state` and `st.rerun()`.
4. **Role Checks**: Always put `require_role(...)` from `utils.auth` at the very top of page scripts.
5. **File Encoding for Streamlit Cloud**: Streamlit Cloud runs on Linux. If you modify files via PowerShell (`Add-Content`), ensure you do not introduce UTF-16 BOM or mixed line endings, as this will crash Python's `import` mechanism in the cloud. Stick to UTF-8.
6. **Editable Dataframes**: We use `st.data_editor` in `pages/7_Admin.py` to allow direct inline editing of users. Do not use `disabled` for fields you want to map back to the database, use `st.column_config` elements (like `SelectboxColumn`) to constrain inputs instead.
7. **Append-Only Submissions**: Survey responses are strictly append-only. Each submission generates a new `UUID4`, meaning multiple responses from the exact same account (e.g., anonymous shared accounts) will **never** overwrite each other. They are limited purely by the `max_surveys_per_year` value.