"""Evaluation helpers: time-based split and the WMAE metric."""

import numpy as np
import pandas as pd

from src.config import HOLIDAY_WEIGHT, TEST_WEEKS


def time_split(df: pd.DataFrame, test_weeks: int = TEST_WEEKS):
    """Split by date: the last `test_weeks` distinct weeks become the test set."""
    dates = np.sort(df["Date"].unique())
    if len(dates) <= test_weeks:
        raise ValueError("Not enough weeks to hold out a test period")
    cutoff = dates[-test_weeks]
    return df[df["Date"] < cutoff].copy(), df[df["Date"] >= cutoff].copy()


def wmae(y_true, y_pred, is_holiday) -> float:
    """Weighted mean absolute error; holiday weeks weigh HOLIDAY_WEIGHT."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.where(np.asarray(is_holiday, dtype=bool), HOLIDAY_WEIGHT, 1)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def score(y_true, y_pred, is_holiday) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return {
        "WMAE": round(wmae(y_true, y_pred, is_holiday), 2),
        "MAE": round(float(np.mean(np.abs(err))), 2),
        "RMSE": round(float(np.sqrt(np.mean(err**2))), 2),
    }
