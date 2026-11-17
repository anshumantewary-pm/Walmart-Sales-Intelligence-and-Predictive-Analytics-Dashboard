"""Gradient-boosting forecaster with prediction intervals.

Three HistGradientBoosting models are trained with quantile loss (10th, 50th and
90th percentile), so every forecast comes with a likely range. The 50th
percentile is the point forecast.

Quantile models on their own tend to give ranges that are too narrow. The range
is therefore calibrated (split conformal): we hold out the last weeks of the
training data, measure how far actuals fall outside the raw range, and widen the
final range by that amount so it covers about 80% of outcomes.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

FEATURES = [
    "Store",
    "Dept",
    "TypeCode",
    "Size",
    "Week",
    "Month",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "TotalMarkDown",
    "Sales_Lag52",
    "Sales_Lag52_Smooth",
    "DeptStore_Mean",
]

QUANTILES = {"lower": 0.1, "pred": 0.5, "upper": 0.9}

DEFAULT_PARAMS = {"max_iter": 300, "learning_rate": 0.08, "max_leaf_nodes": 63,
                  "min_samples_leaf": 20, "random_state": 0}


@dataclass
class SalesForecaster:
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    models: dict = field(default_factory=dict)
    dept_store_mean: pd.Series | None = None
    global_mean: float = 0.0
    calibrate_weeks: int = 13
    interval_margin: float = 0.0

    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        means = self.dept_store_mean.rename("DeptStore_Mean")
        X = X.join(means, on=["Store", "Dept"])
        X["DeptStore_Mean"] = X["DeptStore_Mean"].fillna(self.global_mean)
        X["IsHoliday"] = X["IsHoliday"].astype(int)
        return X[FEATURES].astype(float)

    def _fit_models(self, train: pd.DataFrame) -> None:
        # Average sales per Store/Dept, computed on training data only.
        self.dept_store_mean = train.groupby(["Store", "Dept"])["Weekly_Sales"].mean()
        self.global_mean = float(train["Weekly_Sales"].mean())
        X = self._matrix(train)
        y = train["Weekly_Sales"].to_numpy()
        # Holiday weeks count 5x in WMAE, so weight them 5x while training too.
        w = np.where(train["IsHoliday"].astype(bool), 5.0, 1.0)
        self.models = {}
        for name, q in QUANTILES.items():
            model = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.params)
            self.models[name] = model.fit(X, y, sample_weight=w)

    def fit(self, train: pd.DataFrame) -> "SalesForecaster":
        self.interval_margin = 0.0
        dates = np.sort(train["Date"].unique())
        if self.calibrate_weeks and len(dates) > 2 * self.calibrate_weeks:
            cutoff = dates[-self.calibrate_weeks]
            fit_part, calib = train[train["Date"] < cutoff], train[train["Date"] >= cutoff]
            self._fit_models(fit_part)
            raw = self.predict(calib)
            y = calib["Weekly_Sales"].to_numpy()
            # How far each actual falls outside the raw range (negative = inside).
            scores = np.maximum(raw["lower"].to_numpy() - y, y - raw["upper"].to_numpy())
            n = len(scores)
            level = min(1.0, np.ceil((n + 1) * 0.8) / n)
            self.interval_margin = max(0.0, float(np.quantile(scores, level)))
        self._fit_models(train)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._matrix(df)
        out = pd.DataFrame({name: m.predict(X) for name, m in self.models.items()},
                           index=df.index)
        # Quantile models are trained separately and can cross; keep them ordered.
        ordered = np.sort(out[["lower", "pred", "upper"]].to_numpy(), axis=1)
        out[["lower", "pred", "upper"]] = ordered
        out["lower"] -= self.interval_margin
        out["upper"] += self.interval_margin
        return out

    def feature_importance(self, df: pd.DataFrame, n_rows: int = 5000,
                           seed: int = 0) -> pd.DataFrame:
        """Permutation importance of the median model: how much MAE worsens
        when each feature is shuffled."""
        sample = df.sample(min(n_rows, len(df)), random_state=seed)
        X = self._matrix(sample)
        result = permutation_importance(
            self.models["pred"], X, sample["Weekly_Sales"], n_repeats=3,
            random_state=seed, scoring="neg_mean_absolute_error",
        )
        return (
            pd.DataFrame({"feature": FEATURES, "importance": result.importances_mean})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
