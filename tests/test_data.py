import pandas as pd
import pytest

from src.config import MARKDOWN_COLS
from src.data.clean import clean, validate
from src.data.load import load_raw


def test_load_merges_all_files(raw_dir):
    df = load_raw(raw_dir)
    assert {"Weekly_Sales", "Temperature", "Type", "Size"} <= set(df.columns)
    assert df["Type"].notna().all()


def test_load_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--sample"):
        load_raw(tmp_path)


def test_clean_fills_markdowns_and_dedupes(raw_dir):
    df = load_raw(raw_dir)
    df = pd.concat([df, df.head(3)])  # add duplicates
    out = clean(df)
    validate(out)
    assert out[MARKDOWN_COLS].isna().sum().sum() == 0
    assert not out.duplicated(["Store", "Dept", "Date"]).any()


def test_clean_flags_returns():
    df = pd.DataFrame({
        "Store": [1, 1], "Dept": [1, 1], "Date": ["2011-01-07", "2011-01-14"],
        "Weekly_Sales": [100.0, -5.0], "IsHoliday": [False, False],
        "CPI": [200.0, None], "Unemployment": [None, 7.0],
    })
    out = clean(df)
    assert out["IsReturnWeek"].tolist() == [False, True]
    assert out["CPI"].notna().all() and out["Unemployment"].notna().all()


def test_validate_rejects_duplicates(sales):
    with pytest.raises(ValueError, match="Duplicate"):
        validate(pd.concat([sales, sales.head(1)]))


def test_dates_are_weekly_and_sorted(sales):
    for _, g in sales.groupby(["Store", "Dept"]):
        gaps = g["Date"].diff().dropna().dt.days
        assert (gaps == 7).all()
