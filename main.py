import os
import sys
from datetime import datetime

from config import CSV_FILE, ATLASES, FC_VERSIONS, EXPERIMENT_MODES, BASE_FC_DIR, BASE_DATA_DIR
from experiments.experiment_runner import (
    run_pooled_cv,
    run_leave_one_site_out,
    run_within_site_cv
)

# -----------------------------
# Create timestamped output folder
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("results", f"run_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Results will be saved in: {OUTPUT_DIR}")

# -----------------------------
# Labels CSV
# -----------------------------
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found. Please provide a labels CSV.")

# -----------------------------
# Loop over Atlases and FC Versions
# -----------------------------
for atlas in ATLASES:
    atlas_dir = os.path.join(BASE_FC_DIR, atlas)
    if not os.path.exists(atlas_dir):
        print(f"Atlas directory not found: {atlas_dir}")
        continue

    for fc_version in FC_VERSIONS:
        print(f"\n--- Processing Atlas: {atlas}, FC: {fc_version} ---")

        for exp_mode in EXPERIMENT_MODES:
            print(f"\n>>> Running Experiment Mode: {exp_mode}")

            if exp_mode == "pooled_cv":
                run_pooled_cv(atlas_dir, CSV_FILE, fc_version, OUTPUT_DIR)
            elif exp_mode == "leave_one_site_out":
                run_leave_one_site_out(atlas_dir, CSV_FILE, fc_version, OUTPUT_DIR)
            elif exp_mode == "within_site_cv":
                run_within_site_cv(atlas_dir, CSV_FILE, fc_version, OUTPUT_DIR)
            else:
                print(f"Unknown experiment mode: {exp_mode}")

print("\nAll experiments completed!")
