import pandas as pd
import pytest

from src.data.clean import clean
from src.data.load import load_raw
from src.data.sample import write_sample
from src.features.build import build_features


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw")
    write_sample(path, n_stores=4, n_depts=3)
    return path


@pytest.fixture(scope="session")
def sales(raw_dir) -> pd.DataFrame:
    return build_features(clean(load_raw(raw_dir)))
