
import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the results folder
results_folder = os.path.join(
    os.path.dirname(__file__), "..", "results"
)

# Read training results
df = pd.read_csv(
    os.path.join(results_folder, "training_results.csv")
)

# Measured inference latency in milliseconds
latency_batch1 = {
    "MLP": 0.2266,
    "CNN": 0.4311,
    "Attention": 0.5915
}

latency_batch128 = {
    "MLP": 0.7608,
    "CNN": 6.6082,
    "Attention": 3.3671
}

df["Latency Batch 1 (ms)"] = df["Model"].map(latency_batch1)
df["Latency Batch 128 (ms)"] = df["Model"].map(latency_batch128)

# Graph 1: Accuracy vs Parameters
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Parameters"],
    df["Test Accuracy (%)"],
    s=100
)

for _, row in df.iterrows():
    plt.annotate(
        row["Model"],
        (row["Parameters"], row["Test Accuracy (%)"]),
        xytext=(5, 7),
        textcoords="offset points"
    )

plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy vs Number of Parameters")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(results_folder, "accuracy_vs_parameters.png")
)
plt.close()

# Graph 2: Accuracy vs Inference Latency
plt.figure(figsize=(8, 5))

plt.scatter(
    df["Latency Batch 1 (ms)"],
    df["Test Accuracy (%)"],
    s=100,
    label="Batch size 1"
)

plt.scatter(
    df["Latency Batch 128 (ms)"],
    df["Test Accuracy (%)"],
    s=100,
    marker="x",
    label="Batch size 128"
)

for _, row in df.iterrows():
    plt.annotate(
        row["Model"],
        (row["Latency Batch 1 (ms)"], row["Test Accuracy (%)"]),
        xytext=(5, 7),
        textcoords="offset points"
    )

plt.xlabel("Inference Latency (ms)")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy vs Inference Latency")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(results_folder, "accuracy_vs_latency.png")
)
plt.close()

# Save combined results, including latency
df.to_csv(
    os.path.join(results_folder, "final_comparison.csv"),
    index=False
)

print("Both graphs created successfully!")
print("Saved in:", results_folder)
print("Files:")
print("- accuracy_vs_parameters.png")
print("- accuracy_vs_latency.png")
print("- final_comparison.csv")
