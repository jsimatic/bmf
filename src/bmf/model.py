"""Read and transform datasets"""

from dataclasses import dataclass

import pandas as pd
import datasets as ds

_ML1M_DATASET = "ashraq/movielens_ratings"


def read_data_ml1m() -> ds.DatasetDict:
    """Reads the reference dataset"""
    data = ds.load_dataset(_ML1M_DATASET)
    return data.remove_columns(["imdbId", "tmdbId", "genres", "title", "posters"])


def ds_to_df(dataset: ds.Dataset) -> pd.DataFrame:
    """Export a dataset as a dataframe"""
    return dataset.data.to_pandas()


@dataclass
class MLStats:
    users: int
    movies: int
    ratings: int
    sparsity: float

    def __str__(self) -> str:
        return " - ".join(
            [
                f"Users: {self.users}",
                f"Movies: {self.movies}",
                f"Ratings: {self.ratings}",
                f"Sparsity: {self.sparsity}",
            ]
        )


def df_analyze(df: pd.DataFrame) -> MLStats:
    """Number of user, movies and sparsity"""
    users = df.user_id.unique().shape[0]
    movies = df.movie_id.unique().shape[0]
    ratings = len(df)
    sparsity = 1 - len(df) / (users * movies)
    return MLStats(users, movies, ratings, sparsity)


def to_dense(data: pd.DataFrame) -> pd.DataFrame:
    """Convert tabular data to dense matrix."""
    return data.pivot(index="user_id", columns="movie_id", values="rating")
