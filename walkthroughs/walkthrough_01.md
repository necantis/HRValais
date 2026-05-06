# HR Valais — Walkthrough

## What Was Built

A complete multi-tenant Streamlit application for HR Valais, fully deployable on Streamlit Cloud.

## App Structure

```
HRValais/
├── app.py                          # Entry point — login + role-based routing
├── pages/
│   ├── 1_Survey.py                 # Employee: 21-question Likert survey (7 pillars)
│   ├── 2_Dashboard_Internal.py     # HR Manager: k-anonymized internal radar/trends
│   ├── 3_Dashboard_Benchmarking.py # HR Manager: OFS wage/turnover benchmarking
│   ├── 4_Dashboard_MixedModels.py  # HR Manager: logistic + OLS regression (statsmodels)
│   ├── 5_Dashboard_Timeseries.py   # HR Manager: Markov/Bayesian/Ensemble time-series
│   ├── 6_Upload.py                 # HR Manager: monthly CSV upload via chat_input
│   └── 7_Admin.py                  # Admin: cross-firm heatmap + DB health
├── db/
│   ├── models.py                   # SQLAlchemy ORM models
│   ├── database.py                 # Engine, contextmanager session, init_db()
│   └── seed.py                     # Idempotent cold-start seeder
└── utils/
    ├── auth.py                     # bcrypt + session-state RBAC
    ├── ofs_parser.py               # Pure-Python PC-Axis (.px) parser + synthetic fallback
    ├── kanon.py                    # k-anonymization (k=5 suppression)
    └── pdf_generator.py            # fpdf2 survey PDF generator
```

## Demo Credentials

| Username | Password | Role | Firm |
|---|---|---|---|
| `employee1` | `password123` | Employee | Alpina Services SA |
| `employee2` | `password123` | Employee | Alpina Services SA |
| `hr_manager_a` | `password123` | HR Manager | Alpina Services SA |
| `hr_manager_b` | `password123` | HR Manager | Rhône Industrie Sàrl |
| `admin` | `admin_secure_2026` | Admin | Platform |

## Smoke Test Results (local)

| Check | Result |
|---|---|
| Firms seeded | 3 |
| Users seeded | 5 |
| IBM survey responses | 1 470 |
| Synthetic longitudinal responses | Skipped (count > 10) |
| OFS macro-data rows | 767 520 (synthetic fallback, .px parsed partially) |
| Auth (bcrypt) | Pass |
| PDF generation | Pass |
| Syntax errors across 15 files | 0 |

## Bugs Fixed During Build

1. **[get_session()](file:///c:/Users/riccardo.bonazzi/Documents/GitHub/HRValais/db/database.py#54-72) context manager** — converted from bare `sessionmaker()` call to a `@contextmanager` that commits on success and rolls back on exception.
2. **OFS parser STUB/HEADING** — dimension names in [.px](file:///c:/Users/riccardo.bonazzi/Documents/GitHub/HRValais/data/px-x-0304010000_206.px) files are comma-separated and quoted (`"Jahr","Sektor"`); fixed by stripping surrounding quotes before lookup.
3. **`year=int(...)` ValueError** — OFS synthetic fallback returns `"Unknown"` for year; replaced with [safe_year()](file:///c:/Users/riccardo.bonazzi/Documents/GitHub/HRValais/db/seed.py#227-232) / [safe_float()](file:///c:/Users/riccardo.bonazzi/Documents/GitHub/HRValais/db/seed.py#233-238) helpers.
4. **FK constraint on bulk seed** — pseudo-employee UUIDs not in [User](file:///c:/Users/riccardo.bonazzi/Documents/GitHub/HRValais/db/models.py#35-47) table; disabled FK checks (`PRAGMA foreign_keys=OFF`) during seed, re-enabled after commit.
5. **PDF em-dash encoding** — Helvetica in fpdf2 is Latin-1 only; replaced `—` and `©` with ASCII equivalents.
6. **PowerShell `&&` operator** — used `;` instead throughout.

## Dashboard Features

### Dashboard 1 — Internal (k-anonymized)
- Radar chart of 7 pillar scores
- Trend lines over month_index
- Breakdown by position and engagement state
- k=5 suppression applied; groups with N<5 hidden

### Dashboard 2 — OFS Benchmarking
- Gross monthly wage by position / age / gender
- Gender pay gap calculation
- OFS turnover rate as reference class
- Synthetic data warning banner when .px parse fails

### Dashboard 3 — Mixed Models
- Logistic regression: pillar scores → P(Resigned)
- Odds ratio bar chart
- OLS R² per pillar vs. overall score
- Correlation heatmap
- Prominent p-value caveat boxes

### Dashboard 4 — Time-Series + Falsifiability
- **Markov Chain** with OFS-calibrated prior; stationarity Chi² test
- **Bayesian** analytical Dirichlet posterior; 3-prior sensitivity analysis
- **Ensemble** (weighted Markov + Bayesian)
- **RLlib stub** (reserved slot; GPU dependency)
- Type B subjective probability slider
- Falsifiability monitor: RMSE vs naïve baseline, partial-correlation DAG test, remaining-data estimate, retraining flag

### Upload (HR Manager)
- `st.chat_input(accept_file=True)` for monthly CSV
- Schema validation (7 pillar avg columns required)
- Appends to `survey_responses` + records in `monthly_uploads`

### Admin Dashboard
- DB health metrics (row counts, file size)
- Cross-firm pillar heatmap (SHA-256 anonymized firm IDs)
- OFS data inspector
- User list

## Streamlit Cloud Notes

- The [safe_year](file:///c:/Users/riccardo.bonazzi/Documents/GitHub/HRValais/db/seed.py#227-232) fix is in commit `dd54411` (pushed Feb 24 2026)
- If Cloud still shows the old error, click **"Reboot app"** in the Manage App panel to force a fresh deploy from the latest commit
