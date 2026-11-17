# Walmart Sales Intelligence & Predictive Analytics Dashboard

![CI](https://github.com/anshumantewary-pm/Walmart-Sales-Intelligence-and-Predictive-Analytics-Dashboard/actions/workflows/ci.yml/badge.svg)

An end-to-end analytics project on Walmart's historical weekly sales data. It cleans and merges the data, works out what drives store and department performance (holidays, markdowns, temperature, fuel price, CPI, unemployment), forecasts weekly sales with likely ranges, and shows everything in an interactive Streamlit dashboard with a what-if simulator.

> **Data note:** the pipeline runs on the real Walmart files when they are in `data/raw/`. Without them, `--sample` generates **synthetic** data with the same columns so you can try everything. The dashboard shows a banner whenever it is showing synthetic numbers. Don't quote those numbers as real results.

## Quick start

```bash
git clone https://github.com/anshumantewary-pm/Walmart-Sales-Intelligence-and-Predictive-Analytics-Dashboard.git
cd Walmart-Sales-Intelligence-and-Predictive-Analytics-Dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Option A: synthetic demo data
python -m src.pipeline --sample

# Option B: real data. Put train.csv, features.csv, stores.csv in data/raw/, then:
python -m src.pipeline

streamlit run src/dashboard/app.py
```

Or with Docker (it uses `data/raw/` if present, otherwise synthetic data):

```bash
make docker   # then open http://localhost:8501
```

## Dataset

The public Walmart weekly sales dataset (45 stores, Feb 2010 – Oct 2012), available on Kaggle as *Walmart Recruiting – Store Sales Forecasting*:

| File | Contents |
| --- | --- |
| `train.csv` | Store, Dept, Date, Weekly_Sales, IsHoliday |
| `features.csv` | Store, Date, Temperature, Fuel_Price, MarkDown1–5, CPI, Unemployment, IsHoliday |
| `stores.csv` | Store, Type (A/B/C), Size |

Raw data is git-ignored and never committed.

## What it does

**Data pipeline** (`src/data`)
- Merges the three files and checks that required columns exist
- Fills missing markdowns with 0 and fills CPI/Unemployment within each store
- Removes duplicate rows and flags negative-sales (return) weeks
- Checks data quality: no duplicate Store/Dept/Date rows and no missing sales

**Forecasting** (`src/features`, `src/models`)
- Time-based holdout: the last 13 weeks are never seen in training
- Features: calendar, store type and size, markdowns, macro factors, and last-year sales (no lags shorter than the 13-week horizon, so the future can't leak into training)
- Baselines: seasonal naive (same week last year) and a 4-week moving average
- Gradient boosting (scikit-learn `HistGradientBoostingRegressor`) with quantile loss, giving a forecast plus an 80% range. The range is calibrated on recent training weeks so it really covers about 80%.
- Metrics: **WMAE** (holiday weeks count 5x), MAE, RMSE, and how often actuals fall inside the range
- Permutation feature importance, and an experiment log in `reports/experiments.csv`

**Business insights** (`src/analysis/insights.py`)
- **Holiday lift** by store type
- **Markdown effectiveness:** sales in markdown weeks and in the week after, compared with weeks without markdowns, which shows whether discounts pull demand forward
- **Economic factors:** within-store correlation of sales with temperature, fuel, CPI and unemployment
- **Store segmentation:** k-means on volume, volatility, holiday lift and Q4 share, plus a suggested action for each segment
- **Inventory cost of forecast error:** turns over- and under-forecasting into an estimated $ cost (cost rates are adjustable assumptions)

**Dashboard** (`src/dashboard/app.py`): five tabs
1. **Overview:** KPIs, weekly trend with holiday markers, store-type share, top departments
2. **Store drill-down:** store trend and a department × month heatmap
3. **Forecast:** model comparison, forecast vs actual with the 80% range, $ impact, feature importance
4. **Insights:** holiday lift, markdown effect, economic factors, store segments
5. **What-if:** change markdown spend, temperature, fuel, CPI and unemployment and see the forecast respond

## Project structure

```
├── data/raw/                 # Walmart CSVs (git-ignored)
├── data/processed/           # pipeline outputs (git-ignored)
├── src/
│   ├── config.py             # paths and constants
│   ├── pipeline.py           # end-to-end run: python -m src.pipeline
│   ├── data/                 # load, clean, validate, synthetic sample
│   ├── features/             # feature engineering
│   ├── models/               # baselines, gradient boosting, evaluation
│   ├── analysis/             # business insights
│   └── dashboard/            # Streamlit app
├── tests/                    # pytest suite (data, features, models, insights, pipeline, dashboard)
├── .github/workflows/ci.yml  # lint + tests + pipeline smoke run
├── Dockerfile, Makefile, .pre-commit-config.yaml
└── requirements.txt, requirements-dev.txt, pyproject.toml
```

## Development

```bash
make test        # pytest
make lint        # ruff
pre-commit install   # auto-lint on every commit
```

CI runs lint, the test suite and a synthetic pipeline run on every push to `main` and on every pull request.

## Key findings

_Run the pipeline on the real data and record the top 5 findings here, with numbers. For example: holiday lift by store type, whether markdowns pull demand forward, which macro factors matter, and the forecast's improvement over the seasonal-naive baseline._

## Roadmap

- [x] Data pipeline, validation and synthetic sample data
- [x] Baselines, gradient-boosting model, calibrated prediction ranges, WMAE evaluation
- [x] Business insights: holiday lift, markdowns, macro factors, segments, inventory cost
- [x] Streamlit dashboard with what-if simulator
- [x] Tests, CI, pre-commit, Docker
- [ ] Run on the real Walmart data and fill in *Key findings*
- [ ] Deploy to Streamlit Community Cloud and add the live link + screenshots/GIF here
- [ ] Try LightGBM/Prophet and tune hyperparameters, logged in `reports/experiments.csv`
- [ ] Optional: natural-language Q&A over the data with an LLM

## License

Released under the [MIT License](LICENSE).
