"""Project paths and shared constants."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
# Output folders can be overridden (used by tests and deployments).
PROCESSED_DIR = Path(os.environ.get("WSI_PROCESSED_DIR", ROOT / "data" / "processed"))
MODEL_DIR = Path(os.environ.get("WSI_MODEL_DIR", ROOT / "models"))

RAW_FILES = {
    "train": "train.csv",
    "features": "features.csv",
    "stores": "stores.csv",
}

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]
KEY_COLS = ["Store", "Dept", "Date"]

# Holiday weeks count 5x in the competition metric (WMAE).
HOLIDAY_WEIGHT = 5
# Number of most recent weeks held out for evaluation.
TEST_WEEKS = 13
