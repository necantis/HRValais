"""
utils/test_data.py
Helpers to identify, format, and filter demo/test firms (Alpina Services SA & Rhône Industrie Sàrl).
"""

from __future__ import annotations
from typing import Optional

TEST_FIRM_KEYWORDS = ("alpina", "rhone", "rhône")

def is_test_firm(name: Optional[str]) -> bool:
    """Return True if the firm name matches known test/demo firms."""
    if not name:
        return False
    norm = name.strip().lower()
    return any(k in norm for k in TEST_FIRM_KEYWORDS)

def format_firm_name(name: Optional[str]) -> str:
    """Append '(Test Data)' to test firm names if not already present."""
    if not name:
        return "—"
    if is_test_firm(name) and "(test data)" not in name.lower():
        return f"{name} (Test Data)"
    return name
