"""Cleaning rules for the merged Walmart table."""

import pandas as pd

from src.config import KEY_COLS, MARKDOWN_COLS


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the merged data.

    - Parses dates and makes IsHoliday a bool.
    - Fills missing markdowns with 0 (no promotion that week).
    - Forward/back-fills CPI and Unemployment within each store.
    - Drops exact duplicate Store/Dept/Date rows (keeps the first).
    - Keeps negative sales (returns) but flags them in `IsReturnWeek`.
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["IsHoliday"] = out["IsHoliday"].astype(bool)

    for col in MARKDOWN_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0).clip(lower=0.0)

    out = out.sort_values(KEY_COLS)
    for col in ("CPI", "Unemployment"):
        if col in out.columns:
            out[col] = out.groupby("Store")[col].transform(lambda s: s.ffill().bfill())

    out = out.drop_duplicates(subset=KEY_COLS, keep="first")
    out["IsReturnWeek"] = out["Weekly_Sales"] < 0
    return out.reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Raise if basic data-quality rules are broken."""
    if df.duplicated(subset=KEY_COLS).any():
        raise ValueError("Duplicate Store/Dept/Date rows found")
    if df["Weekly_Sales"].isna().any():
        raise ValueError("Weekly_Sales has missing values")
    if df[MARKDOWN_COLS].isna().any().any():
        raise ValueError("Markdown columns still have missing values")
