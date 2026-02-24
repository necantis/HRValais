"""
utils/kanon.py
k-anonymization helper.
Suppresses any group with fewer than k individuals.
"""

from __future__ import annotations
import pandas as pd


def kanonymize(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    k: int = 5,
) -> pd.DataFrame:
    """
    Aggregate df by group_cols, compute mean of value_cols.
    Groups with fewer than k rows are dropped (suppressed).

    Returns a DataFrame with group_cols + value_cols (means) + '_count' column.
    """
    agg_dict = {col: "mean" for col in value_cols}
    agg_dict["_count"] = (value_cols[0], "count")

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(**{col: (col, "mean") for col in value_cols},
             **{"_count": (value_cols[0], "count")})
        .reset_index()
    )
    suppressed = grouped["_count"] >= k
    result = grouped[suppressed].copy()
    return result
