import numpy as np
import pytest

from src.models.baselines import moving_average, seasonal_naive
from src.models.evaluate import score, time_split, wmae
from src.models.gbm import SalesForecaster


def test_wmae_weights_holidays():
    # Error of 10 on a holiday week and 0 otherwise: (5*10 + 0) / (5 + 1)
    assert wmae([10, 0], [0, 0], [True, False]) == pytest.approx(50 / 6)


def test_score_perfect_forecast():
    s = score([1, 2, 3], [1, 2, 3], [False, True, False])
    assert s == {"WMAE": 0.0, "MAE": 0.0, "RMSE": 0.0}


def test_time_split_has_no_overlap(sales):
    train, test = time_split(sales, 13)
    assert train["Date"].max() < test["Date"].min()
    assert test["Date"].nunique() == 13


def test_baselines_return_one_prediction_per_row(sales):
    train, test = time_split(sales, 13)
    for fn in (seasonal_naive, moving_average):
        pred = fn(train, test)
        assert len(pred) == len(test) and pred.notna().all()
        assert (pred.index == test.index).all()


def test_gbm_beats_moving_average_and_orders_quantiles(sales):
    train, test = time_split(sales, 13)
    model = SalesForecaster(params={"max_iter": 100, "random_state": 0}).fit(train)
    preds = model.predict(test)
    assert (preds["lower"] <= preds["pred"]).all() and (preds["pred"] <= preds["upper"]).all()
    gbm = wmae(test["Weekly_Sales"], preds["pred"], test["IsHoliday"])
    ma = wmae(test["Weekly_Sales"], moving_average(train, test), test["IsHoliday"])
    assert gbm < ma
    imp = model.feature_importance(test, n_rows=500)
    assert len(imp) > 0 and np.isfinite(imp["importance"]).all()
