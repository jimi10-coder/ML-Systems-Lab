
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from prepare_data import load_adult_dataset
from preprocessing import build_preprocessor
from feature_engineering import add_engineered_features


def prepare_data(X_train, X_test, y_train, use_engineered_features=False):

    # Work on copies.
    X_train = X_train.copy().replace("?", np.nan)
    X_test = X_test.copy().replace("?", np.nan)

    # Add the new features when requested.
    if use_engineered_features:
        X_train = add_engineered_features(X_train)
        X_test = add_engineered_features(X_test)

    # Remove exact duplicate training rows only.
    train_data = X_train.copy()
    train_data["class"] = y_train

    train_data = train_data.drop_duplicates()

    y_train = train_data["class"]
    X_train = train_data.drop(columns=["class"])

    # Identify column types.
    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["category", "object"]
    ).columns.tolist()

    # Fit preprocessing only on training data.
    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train


def evaluate_model(X_train, X_test, y_train, y_test, model_name):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, pos_label=">50K")

    print(f"\n{model_name}")
    print("Accuracy:", round(accuracy, 4))
    print("F1-score:", round(f1, 4))

    return accuracy, f1


if __name__ == "__main__":

    # Load the raw dataset.
    df = load_adult_dataset()

    X = df.drop(columns=["class"])
    y = df["class"]

    # Split before cleaning or feature engineering.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Experiment 1: Original features.
    X_train_base, X_test_base, y_train_base = prepare_data(
        X_train,
        X_test,
        y_train,
        use_engineered_features=False
    )

    baseline_results = evaluate_model(
        X_train_base,
        X_test_base,
        y_train_base,
        y_test,
        "BASELINE MODEL - Original Features"
    )

    # Experiment 2: Original + engineered features.
    X_train_new, X_test_new, y_train_new = prepare_data(
        X_train,
        X_test,
        y_train,
        use_engineered_features=True
    )

    engineered_results = evaluate_model(
        X_train_new,
        X_test_new,
        y_train_new,
        y_test,
        "ENGINEERED MODEL - Additional Features"
    )

    # Compare the results.
    results = pd.DataFrame({
        "Model": ["Baseline", "Engineered"],
        "Accuracy": [
            baseline_results[0],
            engineered_results[0]
        ],
        "F1-score": [
            baseline_results[1],
            engineered_results[1]
        ]
    })

    print("\n========== MODEL COMPARISON ==========")
    print(results.to_string(index=False))

    results.to_csv(
        "lab04/results/feature_comparison.csv",
        index=False
    )

    print("\nResults saved to lab04/results/feature_comparison.csv")
