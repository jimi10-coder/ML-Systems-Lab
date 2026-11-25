
from pathlib import Path
import hashlib

# Locate the results folder
results_dir = Path(__file__).resolve().parent.parent / "results"

# Files to hash
files_to_hash = [
    "X_train_processed.npz",
    "X_test_processed.npz",
    "y_train.csv",
    "y_test.csv"
]

# Calculate SHA256
report = []

for filename in files_to_hash:
    file_path = results_dir / filename

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    file_hash = sha256.hexdigest()

    report.append(f"{filename}: {file_hash}")
    print(f"{filename}: {file_hash}")

# Save hashes
with open(results_dir / "sha256_hashes.txt", "w") as file:
    file.write("\n".join(report))

print("\nSHA256 hashes saved successfully.")
