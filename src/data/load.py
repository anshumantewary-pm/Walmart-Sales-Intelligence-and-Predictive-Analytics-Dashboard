"""Load the raw Walmart CSVs and merge them into one table."""

from pathlib import Path

import pandas as pd

from src.config import RAW_DIR, RAW_FILES

REQUIRED_COLUMNS = {
    "train": {"Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"},
    "features": {"Store", "Date", "Temperature", "Fuel_Price", "CPI", "Unemployment"},
    "stores": {"Store", "Type", "Size"},
}


def _read(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the Walmart dataset into {path.parent} "
            "or run the pipeline with --sample to use synthetic data."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS[name] - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return df


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read train, features and stores and merge them on Store and Date."""
    raw_dir = Path(raw_dir)
    train = _read(raw_dir / RAW_FILES["train"], "train")
    features = _read(raw_dir / RAW_FILES["features"], "features")
    stores = _read(raw_dir / RAW_FILES["stores"], "stores")

    for df in (train, features):
        df["Date"] = pd.to_datetime(df["Date"])

    # IsHoliday exists in both train and features; keep the train version.
    features = features.drop(columns=["IsHoliday"], errors="ignore")
    merged = train.merge(features, on=["Store", "Date"], how="left")
    merged = merged.merge(stores, on="Store", how="left")
    return merged
