"""Test baselines.py."""

import numpy as np
import pandas as pd
import pytest
from bmf import baselines


def test_split():
    """Check the title splitting method."""
    assert baselines.split_title("BaselineMethod") == "Baseline Method"  # noqa: S101 test


@pytest.fixture()
def train_data() -> pd.DataFrame:
    """Build an example training set."""
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "movie_id": [2, 3],
            "rating": [0, 1],
        }
    )


def test_global_mean(train_data: pd.DataFrame):
    """Check the GlobalMeanBaseline."""
    pred = baselines.GlobalMeanBaseline(train_data)
    assert np.isclose(pred.predict(1, 2), 0.5)  # noqa: S101 test


def test_rmse(train_data: pd.DataFrame):
    """Check the RMSE computation."""
    pred = baselines.GlobalMeanBaseline(train_data)
    assert np.isclose(pred.rmse(train_data), 0.5)  # noqa: S101 test


@pytest.fixture()
def mean_of_mean_pred(train_data: pd.DataFrame) -> baselines.MeanOfMeansBaseline:
    """Build a MeanOfMeansBaseline."""
    return baselines.MeanOfMeansBaseline(train_data)


@pytest.mark.parametrize(
    ("uid", "mid", "expected"),
    [
        (1, 3, (0.5 + 0 + 1) / 3),
        (1, 1, (0.5 + 0) / 2),
        (3, 3, (0.5 + 1) / 2),
        (0, 0, (0.5) / 1),
    ],
    ids=["all_means", "user_global", "movie_global", "global_only"],
)
def test_mean_of_mean(
    uid: int,
    mid: int,
    expected: float,
    mean_of_mean_pred: baselines.MeanOfMeansBaseline,
):
    """Check the behaviour of mean of mean."""
    assert np.isclose(mean_of_mean_pred.predict(uid, mid), expected)  # noqa: S101 test
