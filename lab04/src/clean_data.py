
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
from prepare_data import load_adult_dataset
from preprocessing import build_preprocessor


def split_dataset(data):
    X = data.drop(columns=["class"])
    y = data["class"]

    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )


def clean_dataset(X_train, X_test, y_train):
    # Work on copies so the original dataset stays unchanged.
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()

    # Convert '?' into missing values.
    X_train = X_train.replace("?", np.nan)
    X_test = X_test.replace("?", np.nan)

    # Remove exact duplicate training rows only.
    training_data = X_train.copy()
    training_data["class"] = y_train

    before = len(training_data)
    training_data = training_data.drop_duplicates()
    after = len(training_data)

    duplicates_removed = before - after

    X_train = training_data.drop(columns=["class"])
    y_train = training_data["class"]

    print("Duplicate training rows removed:", duplicates_removed)

    # Identify numerical and categorical columns.
    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["category", "object"]
    ).columns.tolist()

    print("Numerical columns:", len(numeric_columns))
    print("Categorical columns:", len(categorical_columns))

    # Build and fit the preprocessor using training data only.
    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the already-fitted preprocessor to test data.
    X_test_processed = preprocessor.transform(X_test)

    # Save the fitted preprocessor.
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        preprocessor,
        results_dir / "preprocessor.joblib"
    )

    print("\nPreprocessing completed!")
    print("Training shape:", X_train_processed.shape)
    print("Testing shape:", X_test_processed.shape)
    print("Preprocessor saved successfully.")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        preprocessor
    )


if __name__ == "__main__":
    df = load_adult_dataset()

    X_train, X_test, y_train, y_test = split_dataset(df)

    X_train_processed, X_test_processed, y_train, preprocessor = clean_dataset(
        X_train, X_test, y_train
    )

    # Save processed datasets
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_npz(results_dir / "X_train_processed.npz", X_train_processed)
    save_npz(results_dir / "X_test_processed.npz", X_test_processed)

    # Save target labels
    y_train.to_csv(results_dir / "y_train.csv", index=False)
    y_test.to_csv(results_dir / "y_test.csv", index=False)

    print("Processed datasets and labels saved successfully.")