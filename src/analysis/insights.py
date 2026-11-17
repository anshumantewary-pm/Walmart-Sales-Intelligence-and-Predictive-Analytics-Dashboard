"""Business analyses on the cleaned sales table.

Each function returns a small DataFrame that the pipeline saves and the
dashboard displays.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MACRO_COLS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]


def store_week_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Total sales per store and week, with that week's store-level features."""
    feats = [c for c in ["IsHoliday", "Type", "TotalMarkDown", *MACRO_COLS] if c in df.columns]
    agg = {"Weekly_Sales": "sum", **{c: "first" for c in feats}}
    return df.groupby(["Store", "Date"], as_index=False).agg(agg)


def holiday_lift(df: pd.DataFrame) -> pd.DataFrame:
    """Average store-week sales in holiday vs normal weeks, by store type.

    Note: in the Walmart data the "Christmas" holiday flag falls on the week
    *after* Christmas, which is usually a slow week, so lift can be negative.
    """
    sw = store_week_totals(df)
    table = sw.groupby(["Type", "IsHoliday"])["Weekly_Sales"].mean().unstack("IsHoliday")
    table = table.rename(columns={False: "normal_week_avg", True: "holiday_week_avg"})
    table["lift_pct"] = (table["holiday_week_avg"] / table["normal_week_avg"] - 1) * 100
    return table.round(2).reset_index()


def markdown_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Compare store sales in markdown weeks, the week after, and other weeks.

    Each store's sales are divided by that store's average, so big and small
    stores are comparable. If the week after a markdown is below normal, the
    discount mostly pulled sales forward rather than creating new ones.
    """
    sw = store_week_totals(df).sort_values(["Store", "Date"])
    # Markdowns are only recorded from Nov 2011 in the Walmart data. Before that,
    # "no markdown" really means "unknown", so only compare within the recorded period.
    recorded = sw.loc[sw["TotalMarkDown"] > 0, "Date"]
    if recorded.empty:
        return pd.DataFrame(columns=["period", "avg_relative_sales", "weeks",
                                     "vs_no_markdown_pct"])
    sw = sw[sw["Date"] >= recorded.min()].copy()
    sw["rel_sales"] = sw["Weekly_Sales"] / sw.groupby("Store")["Weekly_Sales"].transform("mean")
    has_md = sw["TotalMarkDown"] > 0
    after_md = has_md.groupby(sw["Store"]).shift(1, fill_value=False) & ~has_md
    sw["period"] = np.select([has_md, after_md], ["markdown_week", "week_after_markdown"],
                             default="no_markdown")
    out = sw.groupby("period")["rel_sales"].agg(["mean", "count"]).reset_index()
    base = out.loc[out["period"] == "no_markdown", "mean"]
    out["vs_no_markdown_pct"] = (out["mean"] / base.iloc[0] - 1) * 100 if len(base) else np.nan
    return out.rename(columns={"mean": "avg_relative_sales", "count": "weeks"}).round(3)


def macro_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between store-week sales and macro factors.

    Values are de-meaned within each store first, so the result reflects how a
    store's sales move with the factor over time, not that big stores differ.
    """
    sw = store_week_totals(df)
    cols = ["Weekly_Sales", *[c for c in MACRO_COLS if c in sw.columns]]
    demeaned = sw[cols] - sw.groupby("Store")[cols].transform("mean")
    corr = demeaned.corr(method="spearman")["Weekly_Sales"].drop("Weekly_Sales")
    out = corr.rename("spearman_corr").reset_index().rename(columns={"index": "factor"})
    out["strength"] = pd.cut(out["spearman_corr"].abs(), [-0.01, 0.1, 0.3, 1.0],
                             labels=["negligible", "weak", "moderate+"])
    return out.round(3)


def store_segments(df: pd.DataFrame, k: int = 3, seed: int = 0) -> pd.DataFrame:
    """Cluster stores by size of sales, volatility, holiday lift and Q4 share."""
    sw = store_week_totals(df)
    sw["Q4"] = sw["Date"].dt.quarter == 4
    g = sw.groupby("Store")
    profile = pd.DataFrame({
        "avg_weekly_sales": g["Weekly_Sales"].mean(),
        "volatility_cv": g["Weekly_Sales"].std() / g["Weekly_Sales"].mean(),
        "holiday_lift": sw[sw["IsHoliday"]].groupby("Store")["Weekly_Sales"].mean()
        / sw[~sw["IsHoliday"]].groupby("Store")["Weekly_Sales"].mean() - 1,
        "q4_share": sw[sw["Q4"]].groupby("Store")["Weekly_Sales"].sum()
        / g["Weekly_Sales"].sum(),
    }).fillna(0)
    k = min(k, len(profile))
    X = StandardScaler().fit_transform(profile)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    profile["cluster"] = labels
    # Name clusters by average sales so labels are stable and readable.
    rank = profile.groupby("cluster")["avg_weekly_sales"].mean().rank(ascending=False)
    names = {1: "High volume", 2: "Mid volume", 3: "Low volume"}
    profile["segment"] = profile["cluster"].map(
        lambda c: names.get(int(rank[c]), f"Segment {int(rank[c])}"))
    return profile.drop(columns="cluster").round(3).reset_index()


SEGMENT_PLAYBOOK = {
    "High volume": "Protect availability: tight replenishment, extra holiday staffing.",
    "Mid volume": "Grow basket size: targeted markdowns on high-lift departments.",
    "Low volume": "Control cost: leaner inventory, fewer broad markdowns.",
}


def inventory_cost(actual, pred, overstock_cost: float = 0.25,
                   understock_cost: float = 0.40) -> dict:
    """Rough cost of forecast error.

    Each $1 over-forecast is assumed to cost `overstock_cost` (holding, waste,
    clearance); each $1 under-forecast costs `understock_cost` (lost margin).
    Both are assumptions to tune, not Walmart figures.
    """
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    over = np.clip(pred - actual, 0, None).sum()
    under = np.clip(actual - pred, 0, None).sum()
    total = overstock_cost * over + understock_cost * under
    return {
        "overstock_$": round(float(over), 2),
        "understock_$": round(float(under), 2),
        "est_cost_$": round(float(total), 2),
        "cost_pct_of_sales": round(float(total / max(actual.sum(), 1e-9) * 100), 3),
    }
