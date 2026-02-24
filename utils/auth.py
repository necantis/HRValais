"""
utils/auth.py
Session-state-based authentication and RBAC for HR Valais.
"""

from __future__ import annotations
import bcrypt
import streamlit as st
from typing import Optional


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def check_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Session state keys
# ---------------------------------------------------------------------------
_KEY_USER = "hrv_user"


def get_current_user() -> Optional[dict]:
    """Return the logged-in user dict or None."""
    return st.session_state.get(_KEY_USER)


def is_authenticated() -> bool:
    return get_current_user() is not None


def get_role() -> Optional[str]:
    user = get_current_user()
    return user["role"] if user else None


def login(username: str, password: str) -> tuple[bool, str]:
    """
    Attempt login. Returns (success, error_message).
    On success sets st.session_state[_KEY_USER].
    """
    from db.database import get_session
    from db.models import User, Firm

    with get_session() as session:
        user: Optional[User] = (
            session.query(User).filter_by(username=username).first()
        )
        if user is None:
            return False, "Unknown username."
        if not check_password(password, user.hashed_password):
            return False, "Incorrect password."

        firm_name = None
        if user.firm_id:
            firm = session.query(Firm).filter_by(firm_id=user.firm_id).first()
            firm_name = firm.name if firm else None

        st.session_state[_KEY_USER] = {
            "user_id": user.user_id,
            "username": user.username,
            "display_name": user.display_name or user.username,
            "role": user.role,
            "firm_id": user.firm_id,
            "firm_name": firm_name,
        }
    return True, ""


def logout() -> None:
    st.session_state.pop(_KEY_USER, None)


def require_role(*allowed_roles: str) -> bool:
    """
    Guard a page. Returns True if the current user has one of the allowed roles.
    Otherwise shows an error and returns False.
    Call as: if not require_role("hr_manager", "admin"): st.stop()
    """
    user = get_current_user()
    if user is None:
        st.error("🔒 Please log in to access this page.")
        st.stop()
    if user["role"] not in allowed_roles:
        st.error(f"🚫 Access denied. This page requires role: {', '.join(allowed_roles)}.")
        st.stop()
    return True
