# Walmart Sales Intelligence & Predictive Analytics Dashboard

An end-to-end analytics project built on Walmart's historical weekly sales data. It cleans and explores the data, finds the drivers of store and department performance (holidays, temperature, fuel price, CPI, unemployment, markdowns), forecasts future weekly sales, and shows everything in an interactive dashboard for business users.

> **Status:** early stage. This repository holds the license and project plan so far. The sections below describe the intended scope and structure; tick items off the roadmap as they land.

---

## Goals

- **Descriptive:** Which stores, departments and store types drive revenue? How do sales move across seasons and holidays?
- **Diagnostic:** How much do holidays, markdowns and macro factors (CPI, fuel price, unemployment, temperature) affect sales?
- **Predictive:** Forecast weekly sales per store/department several weeks ahead, with prediction intervals.
- **Prescriptive:** Turn forecasts into actions such as inventory and staffing signals and markdown timing.

## Dataset

The project targets the public Walmart weekly sales dataset (45 stores, 2010–2012), which typically comes as:

| File | Contents |
| --- | --- |
| `train.csv` | Store, Dept, Date, Weekly_Sales, IsHoliday |
| `features.csv` | Store, Date, Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment, IsHoliday |
| `stores.csv` | Store, Type (A/B/C), Size |

Raw data is **not** committed. Download it into `data/raw/` (see *Getting started*).

## Planned features

- Data pipeline: merging, type fixes, missing-markdown handling, outlier and negative-sales treatment
- Exploratory analysis notebook with key business findings
- Feature engineering: lags, rolling means, holiday windows (Super Bowl, Labor Day, Thanksgiving, Christmas), calendar features
- Forecasting models compared on a time-based holdout:
  - Baselines: seasonal naive, moving average
  - Statistical: SARIMA / Prophet
  - Machine learning: LightGBM / XGBoost on engineered features
- Evaluation with **WMAE** (holiday weeks weighted ×5), plus MAE/RMSE
- Interactive dashboard (Streamlit or Power BI) with:
  - KPI tiles (total sales, growth year over year, holiday lift)
  - Store, department and store-type drill-downs
  - Forecast vs. actual charts with confidence bands
  - What-if sliders for markdowns and macro factors

## Proposed project structure

```
.
├── data/
│   ├── raw/            # original CSVs (git-ignored)
│   └── processed/      # cleaned/feature tables (git-ignored)
├── notebooks/          # EDA and modelling experiments
├── src/
│   ├── data/           # loading & cleaning
│   ├── features/       # feature engineering
│   ├── models/         # training, evaluation, forecasting
│   └── dashboard/      # Streamlit app
├── models/             # saved model artifacts (git-ignored)
├── reports/figures/    # exported charts
├── tests/
├── requirements.txt
└── README.md
```

## Tech stack (planned)

Python · pandas · NumPy · scikit-learn · LightGBM/XGBoost · statsmodels/Prophet · Plotly · Streamlit (or Power BI) · pytest

## Getting started (once code lands)

```bash
git clone https://github.com/anshumantewary-pm/Walmart-Sales-Intelligence-and-Predictive-Analytics-Dashboard.git
cd Walmart-Sales-Intelligence-and-Predictive-Analytics-Dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place train.csv, features.csv, stores.csv in data/raw/
streamlit run src/dashboard/app.py
```

## Roadmap

- [ ] Add `requirements.txt` and the folder skeleton
- [ ] Data cleaning pipeline + EDA notebook
- [ ] Baseline forecasts and evaluation harness (time-based split, WMAE)
- [ ] Gradient-boosting model with lag and holiday features
- [ ] Streamlit dashboard with KPIs, drill-downs and forecasts
- [ ] Unit tests + GitHub Actions CI (lint + tests)
- [ ] Deploy the dashboard (Streamlit Community Cloud) and link a live demo here
- [ ] Add dashboard screenshots and a short "Key insights" section

## License

Released under the [MIT License](LICENSE).
