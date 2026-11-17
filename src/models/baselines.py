"""Baseline forecasters. Any real model should beat these."""

import pandas as pd

GROUP = ["Store", "Dept"]


def moving_average(train: pd.DataFrame, test: pd.DataFrame, window: int = 4) -> pd.Series:
    """Predict the mean of the last `window` training weeks for each Store/Dept."""
    last = (
        train.sort_values("Date")
        .groupby(GROUP)["Weekly_Sales"]
        .apply(lambda s: s.tail(window).mean())
        .rename("pred")
    )
    pred = test[GROUP].merge(last, left_on=GROUP, right_index=True, how="left")["pred"]
    return pred.fillna(train["Weekly_Sales"].mean()).set_axis(test.index)


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame, season: int = 52) -> pd.Series:
    """Predict the sales of the same week one season (52 weeks) earlier.

    Falls back to the moving average when that week is missing.
    """
    hist = train[GROUP + ["Date", "Weekly_Sales"]].copy()
    hist["Date"] = hist["Date"] + pd.Timedelta(weeks=season)
    hist = hist.rename(columns={"Weekly_Sales": "pred"})
    pred = test[GROUP + ["Date"]].merge(hist, on=GROUP + ["Date"], how="left")["pred"]
    pred = pred.set_axis(test.index)
    return pred.fillna(moving_average(train, test))


BASELINES = {
    "seasonal_naive": seasonal_naive,
    "moving_average_4w": moving_average,
}
