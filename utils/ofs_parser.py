"""
utils/ofs_parser.py
Pure-Python PC-Axis (.px) parser for Statistics Switzerland files.
Returns a pd.DataFrame with normalised columns.

Supports CHARSET ANSI / iso-8859-15 files. Multilingual: uses [fr] labels
where available, otherwise falls back to [de].
"""

from __future__ import annotations

import re
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level .px parser
# ---------------------------------------------------------------------------

def _parse_px_header(text: str) -> dict:
    """Extract key→value pairs from the header section of a .px file."""
    meta: dict = {}
    # Strip BOM
    text = text.lstrip("\ufeff")

    # Iterative regex over keyword blocks; values may span multiple lines
    # Pattern: KEYWORD["lang"][(dim)]="value(s)";
    pattern = re.compile(
        r'([A-Z\-]+)'                   # keyword
        r'(?:\[([a-z]+)\])?'            # optional [language]
        r'(?:\("([^"]+)"\))?'           # optional (dimension)
        r'="((?:[^"\\]|\\.|"[^;])*)"'  # value (may contain embedded quotes)
        r'\s*;',
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        keyword = m.group(1)
        lang = m.group(2)       # e.g. 'fr', 'de', None
        dim = m.group(3)        # dimension name or None
        value = m.group(4).replace('\r', '').replace('\n', ' ').strip()

        # Compose key
        key = keyword
        if lang:
            key = f"{keyword}[{lang}]"
        if dim:
            key = f"{key}({dim})"

        meta[key] = value
    return meta


def _split_values(raw: str) -> list[str]:
    """Split a comma-separated list of quoted or unquoted values."""
    # Remove surrounding whitespace/newlines inside the string
    raw = raw.strip()
    items = []
    for part in re.split(r'",\s*"', raw):
        part = part.strip().strip('"')
        if part:
            items.append(part)
    return items


def _get_values_for_dim(meta: dict, dim_name: str, lang: str = "fr") -> list[str]:
    """Return the list of category labels for a dimension, preferring [fr]."""
    key_fr = f"VALUES[{lang}]({dim_name})"
    key_de = f"VALUES[de]({dim_name})"
    key_plain = f"VALUES({dim_name})"
    raw = meta.get(key_fr) or meta.get(key_de) or meta.get(key_plain) or ""
    return _split_values(raw)


def _parse_data_section(text: str) -> list[float]:
    """Extract the DATA= section and return flat list of floats (NaN for '.')."""
    idx = text.find("DATA=")
    if idx == -1:
        return []
    data_str = text[idx + 5:]
    # Strip trailing semicolon
    data_str = data_str.rstrip().rstrip(";")
    values = []
    for token in re.split(r'[\s,]+', data_str.strip()):
        if token in (".", "", '"-"', '"-1"', ".."):
            values.append(np.nan)
        else:
            try:
                values.append(float(token))
            except ValueError:
                values.append(np.nan)
    return values


# ---------------------------------------------------------------------------
# High-level extractors for the two OFS files
# ---------------------------------------------------------------------------

def parse_wages_px(path: Path) -> pd.DataFrame:
    """
    Parse px-x-0304010000_201.px — gross monthly wages.
    Returns DataFrame with columns:
        industry_domain, professional_position, age_bracket, gender,
        year, gross_monthly_median_wage
    """
    try:
        text = path.read_text(encoding="iso-8859-15")
    except Exception as e:
        logger.warning(f"Cannot read {path}: {e}")
        return _synthetic_wages_fallback()

    data_idx = text.find("DATA=")
    if data_idx == -1:
        return _synthetic_wages_fallback()

    header_text = text[:data_idx]
    meta = _parse_px_header(header_text)

    # Discover STUB (row dimensions) and HEADING (column dimensions)
    stub = [s.strip() for s in meta.get("STUB", "").split(",") if s.strip()]
    heading = [s.strip() for s in meta.get("HEADING", "").split(",") if s.strip()]

    all_dims = stub + heading
    # Build list of category labels per dimension
    dim_values: dict[str, list[str]] = {}
    for dim in all_dims:
        labels = _get_values_for_dim(meta, dim)
        if not labels:
            logger.warning(f"No VALUES found for dimension '{dim}' — skipping parse.")
            return _synthetic_wages_fallback()
        dim_values[dim] = labels

    flat_data = _parse_data_section(text)

    # Build multi-index and flatten to rows
    from itertools import product
    keys = [dim_values[d] for d in all_dims]
    combos = list(product(*keys))

    if len(combos) != len(flat_data):
        logger.warning(
            f"Data length mismatch: {len(combos)} combos vs {len(flat_data)} values. "
            "Falling back to synthetic wages."
        )
        return _synthetic_wages_fallback()

    records = []
    for combo, val in zip(combos, flat_data):
        row: dict = dict(zip(all_dims, combo))
        row["gross_monthly_median_wage"] = val
        records.append(row)

    df = pd.DataFrame(records)
    df = _normalise_wages_df(df)
    logger.info(f"Parsed wages: {len(df)} rows from {path.name}")
    return df


def parse_turnover_px(path: Path) -> pd.DataFrame:
    """
    Parse px-x-0304010000_206.px — employee flows / turnover rates.
    Returns DataFrame with columns:
        industry_domain, professional_position, age_bracket, gender,
        year, turnover_rate
    """
    try:
        text = path.read_text(encoding="iso-8859-15")
    except Exception as e:
        logger.warning(f"Cannot read {path}: {e}")
        return _synthetic_turnover_fallback()

    data_idx = text.find("DATA=")
    if data_idx == -1:
        return _synthetic_turnover_fallback()

    header_text = text[:data_idx]
    meta = _parse_px_header(header_text)

    stub = [s.strip() for s in meta.get("STUB", "").split(",") if s.strip()]
    heading = [s.strip() for s in meta.get("HEADING", "").split(",") if s.strip()]
    all_dims = stub + heading

    dim_values: dict[str, list[str]] = {}
    for dim in all_dims:
        labels = _get_values_for_dim(meta, dim)
        if not labels:
            return _synthetic_turnover_fallback()
        dim_values[dim] = labels

    flat_data = _parse_data_section(text)

    from itertools import product
    keys = [dim_values[d] for d in all_dims]
    combos = list(product(*keys))

    if len(combos) != len(flat_data):
        logger.warning("Data length mismatch for turnover .px — using synthetic.")
        return _synthetic_turnover_fallback()

    records = []
    for combo, val in zip(combos, flat_data):
        row: dict = dict(zip(all_dims, combo))
        row["turnover_rate"] = val
        records.append(row)

    df = pd.DataFrame(records)
    df = _normalise_turnover_df(df)
    logger.info(f"Parsed turnover: {len(df)} rows from {path.name}")
    return df


# ---------------------------------------------------------------------------
# Column normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_wages_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map whatever columns were parsed to our canonical schema."""
    rename_candidates = {
        # French labels from the OFS file
        "Division économique": "industry_domain",
        "Classe de salaires": "industry_domain",
        "Branche économique": "industry_domain",
        "Position professionnelle": "professional_position",
        "Classe d'âge": "age_bracket",
        "Sexe": "gender",
        "Année": "year",
        "Total": "professional_position",  # fallback
    }
    df = df.rename(columns={k: v for k, v in rename_candidates.items() if k in df.columns})

    for col in ["industry_domain", "professional_position", "age_bracket", "gender", "year"]:
        if col not in df.columns:
            df[col] = "Unknown"

    df["source_file"] = "_201"
    df["turnover_rate"] = np.nan
    return df


def _normalise_turnover_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_candidates = {
        "Division économique": "industry_domain",
        "Branche économique": "industry_domain",
        "Position professionnelle": "professional_position",
        "Classe d'âge": "age_bracket",
        "Sexe": "gender",
        "Année": "year",
    }
    df = df.rename(columns={k: v for k, v in rename_candidates.items() if k in df.columns})

    for col in ["industry_domain", "professional_position", "age_bracket", "gender", "year"]:
        if col not in df.columns:
            df[col] = "Unknown"

    df["source_file"] = "_206"
    df["gross_monthly_median_wage"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Synthetic fallbacks (OFS-calibrated estimates)
# ---------------------------------------------------------------------------

def _synthetic_wages_fallback() -> pd.DataFrame:
    """Return a small OFS-calibrated synthetic wage table as fallback."""
    logger.warning("Using OFS-calibrated SYNTHETIC wage data (px parse failed).")
    positions = ["Exécution", "Cadres supérieurs et dirigeants"]
    age_brackets = ["< 30 ans", "30-39 ans", "40-49 ans", "50-59 ans", "≥ 60 ans"]
    genders = ["Hommes", "Femmes"]
    base_wages = {
        ("Exécution", "Hommes"): 4800,
        ("Exécution", "Femmes"): 4200,
        ("Cadres supérieurs et dirigeants", "Hommes"): 9500,
        ("Cadres supérieurs et dirigeants", "Femmes"): 8100,
    }
    age_modifier = {"< 30 ans": -600, "30-39 ans": 0, "40-49 ans": 400,
                    "50-59 ans": 600, "≥ 60 ans": 300}
    records = []
    np.random.seed(0)
    for pos in positions:
        for age in age_brackets:
            for gen in genders:
                wage = base_wages[(pos, gen)] + age_modifier[age] + np.random.randint(-200, 200)
                records.append({
                    "industry_domain": "Tous secteurs (synthétique)",
                    "professional_position": pos,
                    "age_bracket": age,
                    "gender": gen,
                    "year": 2022,
                    "gross_monthly_median_wage": float(wage),
                    "turnover_rate": np.nan,
                    "source_file": "_201_synthetic",
                })
    return pd.DataFrame(records)


def _synthetic_turnover_fallback() -> pd.DataFrame:
    """Return a small OFS-calibrated synthetic turnover table as fallback."""
    logger.warning("Using OFS-calibrated SYNTHETIC turnover data (px parse failed).")
    positions = ["Exécution", "Cadres supérieurs et dirigeants"]
    genders = ["Hommes", "Femmes"]
    records = []
    for pos in positions:
        for gen in genders:
            base_rate = 0.12 if pos == "Exécution" else 0.08
            records.append({
                "industry_domain": "Tous secteurs (synthétique)",
                "professional_position": pos,
                "age_bracket": "Tous âges",
                "gender": gen,
                "year": 2022,
                "gross_monthly_median_wage": np.nan,
                "turnover_rate": base_rate + np.random.uniform(-0.02, 0.02),
                "source_file": "_206_synthetic",
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_ofs_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Load both OFS .px files.
    Returns (wages_df, turnover_df, is_real_data: bool)
    is_real_data = True only if both .px files parsed successfully.
    """
    wages_path = data_dir / "px-x-0304010000_201.px"
    turnover_path = data_dir / "px-x-0304010000_206.px"

    wages_df = parse_wages_px(wages_path) if wages_path.exists() else _synthetic_wages_fallback()
    turnover_df = parse_turnover_px(turnover_path) if turnover_path.exists() else _synthetic_turnover_fallback()

    is_real = (
        "_synthetic" not in wages_df["source_file"].iloc[0]
        and "_synthetic" not in turnover_df["source_file"].iloc[0]
    )
    return wages_df, turnover_df, is_real
