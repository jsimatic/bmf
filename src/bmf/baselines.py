"""Balines for our predictions.

Inspired from https://www.pymc.io/projects/examples/en/latest/case_studies/probabilistic_matrix_factorization.html
"""

import re

import pandas as pd
import numpy as np

from . import model


def split_title(title):
    """Change "BaselineMethod" to "Baseline Method"."""
    return re.sub(r"([A-Z])", r" \1", title).strip()


class Baseline:
    """Calculate baseline predictions."""

    def __init__(self, train_data: pd.DataFrame):
        """Simple heuristic-based transductive learning to fill in missing
        values in data matrix."""
        raise NotImplementedError("baseline prediction not implemented for base class")

    def predict(self, user_id: int, movie_id: int) -> float:
        raise NotImplementedError("baseline prediction not implemented for base class")

    def rmse(self, test_data: pd.DataFrame):
        """Calculate root mean squared error for predictions on test data."""
        errs = [
            self.predict(e.user_id, e.movie_id) - e.rating
            for e in test_data.itertuples()
        ]
        return np.sqrt(np.mean(np.square(errs)))

    def __str__(self):
        return split_title(self.__class__.__name__)


# Implement the 3 baselines.


class UniformRandomBaseline(Baseline):
    """Fill missing values with uniform random values."""

    def __init__(self, train_data: pd.DataFrame):
        self.rng = np.random.default_rng(12)
        self.rmin = train_data.rating.min()
        self.rmax = train_data.rating.max()

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.rng.uniform(self.rmin, self.rmax)


class GlobalMeanBaseline(Baseline):
    """Fill in missing values using the global mean."""

    def __init__(self, train_data: pd.DataFrame):
        self.gmean = train_data.rating.mean()

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.gmean


class MeanOfMeansBaseline(Baseline):
    """Fill in missing values using mean of user/item/global means."""

    def __init__(self, train_data: pd.DataFrame):
        dense = model.to_dense(train_data)
        self.global_means = dense.mean(axis=None)
        self.user_means = dense.mean(axis=1)
        self.movie_means = dense.mean(axis=0)

    def predict(self, user_id: int, movie_id: int) -> float:
        return np.nanmean(
            (
                self.global_means,
                self.user_means.get(user_id, np.NaN),
                self.movie_means.get(movie_id, np.NaN),
            )
        )


baseline_methods = {}
baseline_methods["ur"] = UniformRandomBaseline
baseline_methods["gm"] = GlobalMeanBaseline
baseline_methods["mom"] = MeanOfMeansBaseline
