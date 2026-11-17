import json

from src.pipeline import run


def test_pipeline_end_to_end(raw_dir, tmp_path):
    out = tmp_path / "processed"
    meta = run(raw_dir=raw_dir, out_dir=out, model_dir=tmp_path / "models",
               log_path=tmp_path / "experiments.csv")
    for name in ["sales_clean", "forecasts", "feature_importance", "holiday_lift",
                 "markdown_effect", "macro_correlation", "store_segments"]:
        assert (out / f"{name}.csv").exists(), name
    assert (tmp_path / "models" / "forecaster.pkl").exists()
    assert json.loads((out / "metadata.json").read_text())["metrics"] == meta["metrics"]
    assert set(meta["metrics"]) == {"seasonal_naive", "moving_average_4w", "gbm"}
    assert (tmp_path / "experiments.csv").read_text().count("\n") == 4  # header + 3 models
