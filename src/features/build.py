"""Feature engineering for weekly sales forecasting.

Forecasts are made up to TEST_WEEKS (13) weeks ahead, so sales lags shorter than
that would leak future information. We only use lags around one year back.
"""

import pandas as pd

from src.config import MARKDOWN_COLS

STORE_TYPE_CODES = {"A": 0, "B": 1, "C": 2}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    iso = out["Date"].dt.isocalendar()
    out["Year"] = iso["year"].astype(int)
    out["Week"] = iso["week"].astype(int)
    out["Month"] = out["Date"].dt.month
    out["TotalMarkDown"] = out[MARKDOWN_COLS].sum(axis=1)
    if "Type" in out.columns:
        out["TypeCode"] = out["Type"].map(STORE_TYPE_CODES)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add last-year sales for the same Store/Dept and week.

    `Sales_Lag52` is the same week last year; `Sales_Lag52_Smooth` averages weeks
    51-53 back, which is less noisy around moving holidays.
    """
    out = df.sort_values(["Store", "Dept", "Date"]).copy()
    grouped = out.groupby(["Store", "Dept"])["Weekly_Sales"]
    lags = {lag: grouped.shift(lag) for lag in (51, 52, 53)}
    out["Sales_Lag52"] = lags[52]
    out["Sales_Lag52_Smooth"] = pd.concat(lags.values(), axis=1).mean(axis=1, skipna=True)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_lag_features(add_calendar_features(df))
