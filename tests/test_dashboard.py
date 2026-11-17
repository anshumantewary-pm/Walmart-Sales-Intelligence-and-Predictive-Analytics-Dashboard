"""Smoke test: the dashboard renders every tab without raising."""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.pipeline import run  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "src", "dashboard", "app.py")


@pytest.fixture
def built(raw_dir, tmp_path, monkeypatch):
    out, models = tmp_path / "processed", tmp_path / "models"
    run(raw_dir=raw_dir, out_dir=out, model_dir=models, log_path=tmp_path / "log.csv")
    monkeypatch.setenv("WSI_PROCESSED_DIR", str(out))
    monkeypatch.setenv("WSI_MODEL_DIR", str(models))
    # src.config reads the env vars at import time, so reload it for the app.
    import importlib

    import src.config
    importlib.reload(src.config)
    yield
    monkeypatch.delenv("WSI_PROCESSED_DIR")
    monkeypatch.delenv("WSI_MODEL_DIR")
    importlib.reload(src.config)


def test_dashboard_renders(built):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 5
    assert any("Walmart" in t.value for t in at.title)


def test_dashboard_without_data_shows_hint(tmp_path, monkeypatch):
    import importlib

    import src.config
    monkeypatch.setenv("WSI_PROCESSED_DIR", str(tmp_path / "empty"))
    importlib.reload(src.config)
    try:
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception
        assert any("src.pipeline" in w.value for w in at.warning)
    finally:
        monkeypatch.delenv("WSI_PROCESSED_DIR")
        importlib.reload(src.config)
