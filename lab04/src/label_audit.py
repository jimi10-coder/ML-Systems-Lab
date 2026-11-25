
from pathlib import Path
import pandas as pd
from prepare_data import load_adult_dataset

# Load the original dataset
data = load_adult_dataset()

# Identify input features and target label
feature_columns = [column for column in data.columns if column != "class"]

# Find identical feature rows that have different income labels
conflicts = (
    data.groupby(
        feature_columns,
        dropna=False,
        observed=True
    )["class"]
    .nunique()
)

conflicting_groups = conflicts[conflicts > 1]

print("\n========== LABEL QUALITY AUDIT ==========")
print("Conflicting feature groups:", len(conflicting_groups))

# Count how many original rows belong to conflicting groups
conflicting_rows = data.set_index(feature_columns).index.isin(
    conflicting_groups.index
)

print("Rows involved in conflicts:", conflicting_rows.sum())

# Save the audit report
results_dir = Path(__file__).resolve().parent.parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / "label_audit_report.txt", "w") as file:
    file.write("LABEL QUALITY AUDIT\n")
    file.write(f"Conflicting feature groups: {len(conflicting_groups)}\n")
    file.write(f"Rows involved in conflicts: {conflicting_rows.sum()}\n")

print("\nAudit report saved successfully.")
# Select 50 random rows for manual inspection
sample_50 = data.sample(n=50, random_state=42)

# Save the sample to CSV
sample_path = results_dir / "sample_50_rows_for_audit.csv"
sample_50.to_csv(sample_path, index=False)

print("\n50 random rows saved to:")
print(sample_path)