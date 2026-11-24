
import csv
import os

results = [
    {
        "Model": "MLP",
        "Parameters": 235146,
        "Test Accuracy (%)": 87.57,
        "Training Time (seconds)": 75.01
    },
    {
        "Model": "CNN",
        "Parameters": 20490,
        "Test Accuracy (%)": 89.90,
        "Training Time (seconds)": 183.40
    },
    {
        "Model": "Attention",
        "Parameters": 70794,
        "Test Accuracy (%)": 84.13,
        "Training Time (seconds)": 218.83
    }
]

results_folder = os.path.join(
    os.path.dirname(__file__), "..", "results"
)
os.makedirs(results_folder, exist_ok=True)

results_file = os.path.join(
    results_folder, "training_results.csv"
)

with open(results_file, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Training results saved successfully!")
print(results_file)
