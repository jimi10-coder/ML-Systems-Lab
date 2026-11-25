
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Clip numerical values using the 1st and 99th percentiles
    calculated from the training data.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        self.lower_bounds_ = np.nanpercentile(X, 1, axis=0)
        self.upper_bounds_ = np.nanpercentile(X, 99, axis=0)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        return np.clip(
            X,
            self.lower_bounds_,
            self.upper_bounds_
        )


def build_preprocessor(numeric_columns, categorical_columns):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper()),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    return preprocessor
