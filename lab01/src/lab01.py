
import csv
import time
import statistics
import os
from pathlib import Path

import numpy as np
import sklearn
import joblib
import psutil

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Set random seed
SEED = 42
np.random.seed(SEED)

print("Python version:", __import__("sys").version)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)

# Load the Breast Cancer Wisconsin dataset
X, y = load_breast_cancer(return_X_y=True)

print("\nDataset loaded successfully")
print("Dataset shape:", X.shape)
print("Number of samples:", len(y))

# Split the dataset: 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Define the models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=SEED
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=SEED
    ),
}

# Create the results folder
output_path = Path(__file__).resolve().parents[1] / "results"
output_path.mkdir(parents=True, exist_ok=True)

results = []

# Get the current Python process
process = psutil.Process(os.getpid())

# Train and evaluate each model
for name, model in models.items():

    print("\n" + "=" * 45)
    print("Model:", name)

    # Warm-up training run
    model.fit(X_train, y_train)

    # Measure training time over 5 runs
    training_times = []

    for _ in range(5):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        training_times.append(elapsed)

    median_training_time = statistics.median(training_times)

    # Measure saved model size
    model_filename = name.lower().replace(" ", "_") + ".joblib"
    model_path = output_path / model_filename

    joblib.dump(model, model_path)

    model_size_kb = os.path.getsize(model_path) / 1024

    # Measure current process memory usage
    memory_mb = process.memory_info().rss / (1024 * 1024)

    # Measure single-sample inference time over 100 runs
    sample = X_test[0:1]
    inference_times = []

    for _ in range(100):
        start = time.perf_counter()
        model.predict(sample)
        elapsed = time.perf_counter() - start
        inference_times.append(elapsed)

    median_inference_ms = (
        statistics.median(inference_times) * 1000
    )

    # Calculate test accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save measurements
    results.append({
        "model": name,
        "accuracy": f"{accuracy:.4f}",
        "median_training_seconds": f"{median_training_time:.6f}",
        "median_inference_ms": f"{median_inference_ms:.4f}",
        "model_size_kb": f"{model_size_kb:.2f}",
        "process_memory_mb": f"{memory_mb:.2f}",
    })

    # Print results
    print(f"Test accuracy: {accuracy:.4f}")
    print(
        f"Median training time: "
        f"{median_training_time:.6f} seconds"
    )
    print(
        f"Median inference time: "
        f"{median_inference_ms:.4f} ms"
    )
    print(f"Model size: {model_size_kb:.2f} KB")
    print(f"Process memory usage: {memory_mb:.2f} MB")
    print(f"Saved model: {model_path}")

# Save results to CSV
csv_path = output_path / "baseline_accuracy.csv"

fieldnames = [
    "model",
    "accuracy",
    "median_training_seconds",
    "median_inference_ms",
    "model_size_kb",
    "process_memory_mb",
]

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 45)
print("All measurements completed.")
print("Results saved to:", csv_path)
