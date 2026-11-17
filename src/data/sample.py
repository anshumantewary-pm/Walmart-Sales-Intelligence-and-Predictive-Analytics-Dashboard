"""Generate SYNTHETIC data in the same shape as the Walmart dataset.

Used for tests, CI and trying the dashboard before the real data is downloaded.
The numbers are made up and must not be presented as real Walmart results.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MARKDOWN_COLS, RAW_FILES

# Holiday weeks (week-ending Fridays) as defined in the original dataset.
HOLIDAY_DATES = pd.to_datetime([
    "2010-02-12", "2010-09-10", "2010-11-26", "2010-12-31",
    "2011-02-11", "2011-09-09", "2011-11-25", "2011-12-30",
    "2012-02-10", "2012-09-07", "2012-11-23", "2012-12-28",
])


def make_sample(n_stores: int = 5, n_depts: int = 6, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-02-05", "2012-10-26", freq="W-FRI")
    week = dates.isocalendar().week.to_numpy().astype(float)
    is_holiday = dates.isin(HOLIDAY_DATES)

    stores = pd.DataFrame({
        "Store": np.arange(1, n_stores + 1),
        "Type": rng.choice(["A", "B", "C"], size=n_stores, p=[0.5, 0.35, 0.15]),
        "Size": rng.integers(40_000, 220_000, size=n_stores),
    })

    feature_rows = []
    for store in stores["Store"]:
        n = len(dates)
        # Like the real data, markdowns are only recorded from Nov 2011 onwards.
        recorded = dates >= pd.Timestamp("2011-11-11")
        promo_week = recorded & (rng.random(n) < 0.6)
        markdowns = {
            c: np.where(promo_week & (rng.random(n) < 0.5), rng.gamma(2.0, 2_000, n), np.nan)
            for c in MARKDOWN_COLS
        }
        feature_rows.append(pd.DataFrame({
            "Store": store,
            "Date": dates,
            "Temperature": 60 + 25 * np.sin(2 * np.pi * (week - 15) / 52) + rng.normal(0, 4, n),
            "Fuel_Price": 2.7 + np.linspace(0, 1.0, n) + rng.normal(0, 0.05, n),
            **markdowns,
            "CPI": 210 + np.linspace(0, 12, n) + rng.normal(0, 0.3, n),
            "Unemployment": 8.0 - np.linspace(0, 1.2, n) + rng.normal(0, 0.1, n),
            "IsHoliday": is_holiday,
        }))
    features = pd.concat(feature_rows, ignore_index=True)

    size_by_store = stores.set_index("Store")["Size"]
    train_rows = []
    for store in stores["Store"]:
        for dept in range(1, n_depts + 1):
            base = rng.uniform(5_000, 40_000) * size_by_store[store] / 150_000
            season = 1 + 0.15 * np.sin(2 * np.pi * (week - 10) / 52)
            christmas = np.where(week >= 47, 1 + 0.4 * (week - 46) / 6, 1.0)
            holiday = np.where(is_holiday, 1.12, 1.0)
            noise = rng.normal(1, 0.06, len(dates))
            sales = base * season * christmas * holiday * noise
            train_rows.append(pd.DataFrame({
                "Store": store,
                "Dept": dept,
                "Date": dates,
                "Weekly_Sales": sales.round(2),
                "IsHoliday": is_holiday,
            }))
    train = pd.concat(train_rows, ignore_index=True)
    return {"train": train, "features": features, "stores": stores}


def write_sample(out_dir: Path, **kwargs) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in make_sample(**kwargs).items():
        df.to_csv(out_dir / RAW_FILES[name], index=False)
