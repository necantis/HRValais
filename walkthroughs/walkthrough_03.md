# Administration & Survey Walkthrough

This document outlines the modifications made to achieve the requested functionality.

## Modifications Made

### 1. Survey Fiche Links & Limits
- **Removed Links:** Modified `pages/1_Survey.py` to remove the HTML link for "Fiche pratique HR Valais" beneath each question.
- **Enforced Submission Limits:** Added a query at the top of `pages/1_Survey.py` that verifies the number of surveys the current user has submitted in the current year against their `max_surveys_per_year` limit.

### 2. Database Schema
- **New Column:** Introduced `max_surveys_per_year` to the `User` table in `db/models.py` with a default of 1.
- **Reseeding:** Ran a database migration script that injected the new column, cleared the data, and ran a fresh seed. `db/seed.py` was updated so that your test accounts (`Employee1_firmA`, `Manager1_firmA`, etc.) remain fully functional!

### 3. Authentication
- **Transition to DB:** Removed the hardcoded `USERS` dictionary in `app.py` and replaced it with a call to `utils.auth.login()`. The app now verifies passwords correctly against the SQLite DB hashes.
- **Session State:** Updated `utils/auth.py` to ensure the `max_surveys_per_year` limit is preserved in the session.

### 4. Admin Panel & HR Management (`pages/7_Admin.py`)
- **Combined View:** Changed the permissions from `admin` to `admin` OR `hr_manager`.
- **Admin Capabilities:**
  - Added a form to create **Firms**, which automatically spawns a `manager_...` and `employee_...` account upon creation.
  - Added a form to create arbitrary users (Admins, Managers, Employees) assigned to any firm.
  - Added an interface to configure the `max_surveys_per_year` limit for any user.
- **Manager Capabilities:**
  - Managers can now access the admin panel, but they only see their **own firm's stats and employees**.
  - They can create **new employees only** (hard-assigned to their firm).
  - They can configure the `max_surveys_per_year` limit for their employees.

## Verification
- Test accounts are `admin1` (Admin) and `Manager1_firmA` (HR Manager).
- You can now test generating a new firm and verifying the auto-generated accounts!
