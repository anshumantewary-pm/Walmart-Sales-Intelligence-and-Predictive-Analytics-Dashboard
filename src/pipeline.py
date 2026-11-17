"""Run the whole project end to end.

    python -m src.pipeline            # real data in data/raw/
    python -m src.pipeline --sample   # synthetic demo data

Outputs land in data/processed/ (tables, metrics, forecasts), models/ (the
trained forecaster) and reports/experiments.csv (one row per run).
"""

import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.analysis import insights
from src.config import MODEL_DIR, PROCESSED_DIR, RAW_DIR, ROOT, TEST_WEEKS
from src.data.clean import clean, validate
from src.data.load import load_raw
from src.data.sample import write_sample
from src.features.build import build_features
from src.models.baselines import BASELINES
from src.models.evaluate import score, time_split
from src.models.gbm import SalesForecaster


def run(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR, model_dir: Path | None = None,
        sample: bool = False, test_weeks: int = TEST_WEEKS, log_path: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir or MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path or ROOT / "reports" / "experiments.csv")

    if sample:
        raw_dir = ROOT / "data" / "sample"
        write_sample(raw_dir)
        print(f"Wrote SYNTHETIC sample data to {raw_dir}")

    t0 = time.time()
    df = clean(load_raw(raw_dir))
    validate(df)
    df = build_features(df)
    df.to_csv(out_dir / "sales_clean.csv", index=False)
    print(f"Cleaned {len(df):,} rows, {df['Store'].nunique()} stores, "
          f"{df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")

    # --- Forecasting ---------------------------------------------------------
    train, test = time_split(df, test_weeks)
    forecasts = test[["Store", "Dept", "Date", "IsHoliday", "Weekly_Sales"]].copy()
    metrics = {}
    for name, fn in BASELINES.items():
        forecasts[name] = fn(train, test)
        metrics[name] = score(test["Weekly_Sales"], forecasts[name], test["IsHoliday"])

    model = SalesForecaster().fit(train)
    preds = model.predict(test)
    forecasts["gbm"] = preds["pred"]
    forecasts["gbm_lower"] = preds["lower"]
    forecasts["gbm_upper"] = preds["upper"]
    metrics["gbm"] = score(test["Weekly_Sales"], forecasts["gbm"], test["IsHoliday"])
    inside = (test["Weekly_Sales"] >= preds["lower"]) & (test["Weekly_Sales"] <= preds["upper"])
    metrics["gbm"]["interval_80_coverage"] = round(float(inside.mean()), 3)

    for name in [*BASELINES, "gbm"]:
        metrics[name].update(insights.inventory_cost(test["Weekly_Sales"], forecasts[name]))

    forecasts.to_csv(out_dir / "forecasts.csv", index=False)
    model.feature_importance(test).to_csv(out_dir / "feature_importance.csv", index=False)
    with open(model_dir / "forecaster.pkl", "wb") as f:
        pickle.dump(model, f)

    # --- Business insights ---------------------------------------------------
    insights.holiday_lift(df).to_csv(out_dir / "holiday_lift.csv", index=False)
    insights.markdown_effect(df).to_csv(out_dir / "markdown_effect.csv", index=False)
    insights.macro_correlation(df).to_csv(out_dir / "macro_correlation.csv", index=False)
    insights.store_segments(df).to_csv(out_dir / "store_segments.csv", index=False)

    meta = {
        "data_source": "synthetic" if sample else "walmart",
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(df),
        "train_end": f"{train['Date'].max():%Y-%m-%d}",
        "test_weeks": test_weeks,
        "metrics": metrics,
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    _log_experiment(log_path, meta, model.params)

    print("\nHoldout results (lower is better):")
    print(pd.DataFrame(metrics).T[["WMAE", "MAE", "RMSE", "cost_pct_of_sales"]].to_string())
    return meta


def _log_experiment(path: Path, meta: dict, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{
        "run_at": meta["run_at"], "data_source": meta["data_source"], "model": name,
        "test_weeks": meta["test_weeks"], "params": json.dumps(params) if name == "gbm" else "",
        **{k: v for k, v in m.items() if k in ("WMAE", "MAE", "RMSE", "interval_80_coverage")},
    } for name, m in meta["metrics"].items()]
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", action="store_true", help="use synthetic demo data")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--test-weeks", type=int, default=TEST_WEEKS)
    args = parser.parse_args()
    run(raw_dir=args.raw_dir, sample=args.sample, test_weeks=args.test_weeks)


if __name__ == "__main__":
    main()
