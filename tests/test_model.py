import pandas as pd
import pytest

from bmf import model


@pytest.mark.slow
def test_read():
    data = model.read_data_ml1m()
    for ds in data.values():
        assert sorted(ds.features.keys()) == ["movie_id", "rating", "user_id"]


@pytest.fixture
def train_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "movie_id": [0, 2, 2],
            "rating": [0, 1, 2],
        }
    )


def test_analyse(train_data: pd.DataFrame):
    assert model.df_analyze(train_data) == model.MLStats(users=3, movies=2, ratings=3)


def test_to_dense(train_data):
    assert model.to_dense(train_data).shape == (3, 2)
