
import subprocess
import sys
from pathlib import Path

# Locate the Lab 04 directory
src_dir = Path(__file__).resolve().parent
project_dir = src_dir.parent

# Scripts to execute in order
scripts = [
    "prepare_data.py",
    "clean_data.py",
    "experiments.py",
    "label_audit.py",
    "hash_data.py"
]

print("\n========== LAB 04 PIPELINE ==========\n")

for script in scripts:
    script_path = src_dir / script

    print(f"\nRunning {script}...")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=project_dir.parent
    )

    if result.returncode != 0:
        print(f"\nERROR: {script} failed.")
        sys.exit(result.returncode)

print("\n========== PIPELINE COMPLETED ==========")
print("All scripts executed successfully.")
