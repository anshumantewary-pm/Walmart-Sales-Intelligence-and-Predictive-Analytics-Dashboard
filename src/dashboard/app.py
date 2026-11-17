"""Streamlit dashboard.

    streamlit run src/dashboard/app.py

Reads the outputs of `python -m src.pipeline` from data/processed/ and models/.
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so `src` imports work under `streamlit run`

from src.analysis.insights import SEGMENT_PLAYBOOK  # noqa: E402
from src.config import MODEL_DIR, PROCESSED_DIR  # noqa: E402

MODEL_PATH = MODEL_DIR / "forecaster.pkl"
MODEL_LABELS = {"gbm": "Gradient boosting", "seasonal_naive": "Seasonal naive",
                "moving_average_4w": "4-week moving average"}

st.set_page_config(page_title="Walmart Sales Intelligence", page_icon="📊", layout="wide")


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def money(x: float) -> str:
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"${x:,.0f}"


if not (PROCESSED_DIR / "metadata.json").exists():
    st.title("Walmart Sales Intelligence")
    st.warning("No processed data yet. Run `python -m src.pipeline` (real data in data/raw/) "
               "or `python -m src.pipeline --sample` (synthetic demo), then reload.")
    st.stop()

meta = json.loads((PROCESSED_DIR / "metadata.json").read_text())
sales = load_table("sales_clean")
forecasts = load_table("forecasts")

st.title("Walmart Sales Intelligence & Forecasting")
if meta["data_source"] == "synthetic":
    st.info("Showing **synthetic demo data**. Numbers are made up; run the pipeline on the "
            "real Walmart files for real insights.")
st.caption(f"{meta['rows']:,} rows · trained through {meta['train_end']} · "
           f"{meta['test_weeks']}-week holdout · last run {meta['run_at']}")

# ---- Filters ---------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    types = st.multiselect("Store type", sorted(sales["Type"].unique()),
                           default=sorted(sales["Type"].unique()))
    stores = st.multiselect("Stores", sorted(sales["Store"].unique()),
                            help="Leave empty for all stores")
filtered = sales[sales["Type"].isin(types)]
if stores:
    filtered = filtered[filtered["Store"].isin(stores)]
if filtered.empty:
    st.warning("No data for these filters.")
    st.stop()

tab_overview, tab_store, tab_forecast, tab_insights, tab_whatif = st.tabs(
    ["Overview", "Store drill-down", "Forecast", "Insights", "What-if"])

# ---- Overview --------------------------------------------------------------
with tab_overview:
    weekly = filtered.groupby("Date", as_index=False)["Weekly_Sales"].sum()
    yearly = filtered.groupby("Year")["Weekly_Sales"].sum()
    hol = filtered.groupby("IsHoliday")["Weekly_Sales"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sales", money(filtered["Weekly_Sales"].sum()))
    c2.metric("Avg weekly sales", money(weekly["Weekly_Sales"].mean()))
    # Compare same weeks of the latest year with the year before, so partial years are fair.
    last_year = filtered["Year"].max()
    weeks_seen = filtered.loc[filtered["Year"] == last_year, "Week"].unique()
    ytd = filtered[filtered["Week"].isin(weeks_seen)].groupby("Year")["Weekly_Sales"].sum()
    if len(ytd) >= 2 and ytd.iloc[-2] > 0:
        c3.metric(f"YoY growth ({last_year}, same weeks)",
                  f"{(ytd.iloc[-1] / ytd.iloc[-2] - 1) * 100:+.1f}%")
    else:
        c3.metric("YoY growth", "n/a")
    if True in hol.index and False in hol.index:
        c4.metric("Holiday lift (dept-week)", f"{(hol[True] / hol[False] - 1) * 100:+.1f}%")

    fig = px.line(weekly, x="Date", y="Weekly_Sales", title="Total weekly sales")
    hol_dates = filtered.loc[filtered["IsHoliday"], "Date"].unique()
    for d in hol_dates:
        fig.add_vline(x=d, line_dash="dot", line_color="gray", opacity=0.4)
    fig.update_layout(yaxis_title="Sales ($)", xaxis_title=None)
    st.plotly_chart(fig, width="stretch")
    st.caption("Dotted lines mark holiday weeks.")

    left, right = st.columns(2)
    by_type = filtered.groupby("Type", as_index=False)["Weekly_Sales"].sum()
    left.plotly_chart(px.pie(by_type, names="Type", values="Weekly_Sales",
                             title="Sales share by store type", hole=0.5),
                      width="stretch")
    top_depts = (filtered.groupby("Dept", as_index=False)["Weekly_Sales"].sum()
                 .nlargest(10, "Weekly_Sales"))
    top_depts["Dept"] = top_depts["Dept"].astype(str)
    right.plotly_chart(px.bar(top_depts, x="Weekly_Sales", y="Dept", orientation="h",
                              title="Top 10 departments").update_yaxes(autorange="reversed"),
                       width="stretch")

# ---- Store drill-down ------------------------------------------------------
with tab_store:
    store = st.selectbox("Store", sorted(filtered["Store"].unique()))
    s = filtered[filtered["Store"] == store]
    info = s.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Type", info["Type"])
    c2.metric("Size (sq ft)", f"{int(info['Size']):,}")
    c3.metric("Total sales", money(s["Weekly_Sales"].sum()))
    sw = s.groupby("Date", as_index=False)["Weekly_Sales"].sum()
    st.plotly_chart(px.line(sw, x="Date", y="Weekly_Sales", title=f"Store {store} weekly sales"),
                    width="stretch")
    heat = s.pivot_table(index="Dept", columns="Month", values="Weekly_Sales", aggfunc="mean")
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                              title="Average weekly sales by department and month",
                              labels={"color": "Sales ($)"}), width="stretch")

# ---- Forecast --------------------------------------------------------------
with tab_forecast:
    st.subheader("Holdout accuracy")
    m = pd.DataFrame(meta["metrics"]).T
    m.index = [MODEL_LABELS.get(i, i) for i in m.index]
    cols = [c for c in ["WMAE", "MAE", "RMSE", "interval_80_coverage", "est_cost_$",
                        "cost_pct_of_sales"] if c in m.columns]
    st.dataframe(m[cols], width="stretch")
    best_base = min(meta["metrics"]["seasonal_naive"]["WMAE"],
                    meta["metrics"]["moving_average_4w"]["WMAE"])
    gain = (1 - meta["metrics"]["gbm"]["WMAE"] / best_base) * 100
    saving = (meta["metrics"]["seasonal_naive"]["est_cost_$"]
              - meta["metrics"]["gbm"]["est_cost_$"])
    st.markdown(f"Gradient boosting is **{gain:.1f}% better (WMAE)** than the best baseline. "
                f"With the assumed inventory costs, that is about **{money(saving)}** less "
                f"forecast-error cost over the {meta['test_weeks']}-week holdout than "
                f"seasonal naive.")
    st.caption("WMAE weights holiday weeks 5x. Inventory cost assumes $0.25 per $1 "
               "over-forecast and $0.40 per $1 under-forecast; tune these in "
               "src/analysis/insights.py.")

    fc = forecasts[forecasts["Store"].isin(filtered["Store"].unique())]
    level = st.radio("Level", ["All selected stores", "Single store / department"],
                     horizontal=True)
    if level == "Single store / department":
        c1, c2 = st.columns(2)
        fs = c1.selectbox("Store ", sorted(fc["Store"].unique()))
        fd = c2.selectbox("Department", sorted(fc.loc[fc["Store"] == fs, "Dept"].unique()))
        fc = fc[(fc["Store"] == fs) & (fc["Dept"] == fd)]
    agg = fc.groupby("Date", as_index=False)[
        ["Weekly_Sales", "gbm", "gbm_lower", "gbm_upper", "seasonal_naive"]].sum()
    hist = filtered[filtered["Date"] < agg["Date"].min()]
    if level == "Single store / department":
        hist = hist[(hist["Store"] == fs) & (hist["Dept"] == fd)]
    hist = hist.groupby("Date", as_index=False)["Weekly_Sales"].sum().tail(26)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["Date"], y=agg["gbm_upper"], mode="lines",
                             line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=agg["Date"], y=agg["gbm_lower"], fill="tonexty", mode="lines",
                             line=dict(width=0), name="80% range",
                             fillcolor="rgba(31,119,180,0.2)"))
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Weekly_Sales"], name="History",
                             line=dict(color="gray")))
    fig.add_trace(go.Scatter(x=agg["Date"], y=agg["Weekly_Sales"], name="Actual",
                             line=dict(color="black")))
    fig.add_trace(go.Scatter(x=agg["Date"], y=agg["gbm"], name="Gradient boosting",
                             line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=agg["Date"], y=agg["seasonal_naive"], name="Seasonal naive",
                             line=dict(color="#ff7f0e", dash="dash")))
    fig.update_layout(title="Forecast vs actual on the holdout weeks", yaxis_title="Sales ($)")
    st.plotly_chart(fig, width="stretch")
    if level == "All selected stores":
        st.caption("The 80% range is summed across series, so it is only a rough guide "
                   "at this level.")

    imp = load_table("feature_importance").head(12)
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                           title="What drives the forecast (permutation importance, $ MAE)")
                    .update_yaxes(autorange="reversed"), width="stretch")

# ---- Insights --------------------------------------------------------------
with tab_insights:
    st.subheader("Holiday lift by store type")
    st.dataframe(load_table("holiday_lift"), width="stretch", hide_index=True)
    st.caption("In the Walmart data the Christmas flag marks the week *after* Christmas, "
               "which drags holiday averages down.")

    st.subheader("Do markdowns create sales or just move them?")
    md = load_table("markdown_effect")
    st.plotly_chart(px.bar(md, x="period", y="vs_no_markdown_pct",
                           title="Sales vs weeks without markdowns (%)"),
                    width="stretch")
    st.caption("Sales are scaled to each store's average. A dip in the week after a markdown "
               "suggests the discount pulled demand forward.")

    st.subheader("Does the economy move sales?")
    st.dataframe(load_table("macro_correlation"), width="stretch", hide_index=True)
    st.caption("Spearman correlation within each store over time.")

    st.subheader("Store segments")
    seg = load_table("store_segments")
    st.plotly_chart(px.scatter(seg, x="avg_weekly_sales", y="volatility_cv", color="segment",
                               size="q4_share", hover_data=["Store", "holiday_lift"],
                               title="Stores by sales volume and volatility"),
                    width="stretch")
    for name, action in SEGMENT_PLAYBOOK.items():
        n = int((seg["segment"] == name).sum())
        if n:
            st.markdown(f"- **{name}** ({n} stores): {action}")

# ---- What-if ---------------------------------------------------------------
with tab_whatif:
    st.subheader("What-if simulator")
    st.write("Change the conditions for the holdout weeks and see how the model's forecast "
             "responds. This shows what the model learned, not a proven causal effect.")
    if not MODEL_PATH.exists():
        st.warning("No trained model found. Run the pipeline first.")
        st.stop()
    model = load_model()
    c1, c2 = st.columns(2)
    ws = c1.selectbox("Store  ", sorted(filtered["Store"].unique()))
    wd = c2.selectbox("Department ", sorted(filtered.loc[filtered["Store"] == ws, "Dept"]
                                            .unique()))
    rows = filtered[(filtered["Store"] == ws) & (filtered["Dept"] == wd)
                    & (filtered["Date"] >= forecasts["Date"].min())].copy()
    if rows.empty:
        st.warning("No holdout weeks for this store/department.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    md_mult = c1.slider("Markdown spend (x current)", 0.0, 3.0, 1.0, 0.1)
    md_add = c1.number_input("Extra markdown $ per week", 0, 100_000, 0, 1_000)
    temp = c2.slider("Temperature change (°F)", -20, 20, 0)
    fuel = c2.slider("Fuel price change ($/gal)", -1.0, 1.0, 0.0, 0.05)
    cpi = c3.slider("CPI change (%)", -5.0, 5.0, 0.0, 0.5)
    unemp = c3.slider("Unemployment change (pts)", -3.0, 3.0, 0.0, 0.1)

    scenario = rows.copy()
    scenario["TotalMarkDown"] = scenario["TotalMarkDown"] * md_mult + md_add
    scenario["Temperature"] += temp
    scenario["Fuel_Price"] += fuel
    scenario["CPI"] *= 1 + cpi / 100
    scenario["Unemployment"] += unemp
    base = model.predict(rows)["pred"]
    new = model.predict(scenario)["pred"]
    delta = new.sum() - base.sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline forecast", money(base.sum()))
    c2.metric("Scenario forecast", money(new.sum()), f"{delta / base.sum() * 100:+.1f}%")
    c3.metric("Change", money(delta))
    plot = pd.DataFrame({"Date": rows["Date"], "Baseline": base.values,
                         "Scenario": new.values})
    st.plotly_chart(px.line(plot, x="Date", y=["Baseline", "Scenario"],
                            title="Forecast under the scenario"), width="stretch")
