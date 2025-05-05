# run_all_py.py
import subprocess
import os
import sys
import datetime

# --- Configuration ---
# Match the settings you want for the runs
ARCH = "resnet18"
EPOCHS = 50
BATCH_SIZE = 64
WORKERS = 2
FP16 = "--fp16-precision" # Include if using, otherwise set to ""
RUN_EVAL = "--run_linear_eval" # Include if running eval, otherwise set to "" or use "--no_linear_eval"

# Define the two sets of experiments
experiments_set1 = {
    "subset_fraction": 0.25,
    "augmentations": ["baseline", "jitter", "blur", "gray", "rotate", "solarize", "erase", "all"]
}

experiments_set2 = {
    "subset_fraction": [0.05, 0.10, 0.15, 0.20],
    "augmentations": ["all"]
}
# --- /Configuration ---

# --- Helper to build and run command ---
def run_experiment(subset_fraction, augmentation_name):
    """Builds and executes the run.py command for one experiment."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("-" * 60)
    print(f"[{timestamp}] Starting Experiment:")
    print(f"  Subset Fraction: {subset_fraction}")
    print(f"  Augmentation:    {augmentation_name}")
    print("-" * 60)

    # Construct the command arguments as a list for robustness
    command = [
        sys.executable, # Use the current python interpreter
        "run.py",
        "--arch", ARCH,
        "--subset_fraction", str(subset_fraction),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--workers", str(WORKERS),
        "--augmentations", augmentation_name
    ]
    # Add optional flags if they are set
    if FP16:
        command.append(FP16)
    if RUN_EVAL:
        command.append(RUN_EVAL)

    print(f"Executing command: {' '.join(command)}") # Print the command being run

    # Run the command
    # Using shell=False and passing a list is generally safer
    # Use check=False so this script continues even if run.py fails for one experiment
    process_result = subprocess.run(command, shell=False, check=False)

    # Check the result
    if process_result.returncode == 0:
        print("-" * 60)
        print(f"Experiment (Subset {subset_fraction}, Aug {augmentation_name}) finished successfully.")
        print("-" * 60)
    else:
        print("=" * 60)
        print(f"ERROR: Experiment (Subset {subset_fraction}, Aug {augmentation_name}) failed with return code {process_result.returncode}.")
        print("Continuing with the next experiment...")
        print("=" * 60)
    print("\n")


# --- Main Execution ---
start_time = datetime.datetime.now()
print(f"Starting full experiment suite at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Run Set 1
print("\n### Running Set 1: Augmentation Comparison at 25% Subset ###\n")
subset1 = experiments_set1["subset_fraction"]
for aug_name in experiments_set1["augmentations"]:
    run_experiment(subset1, aug_name)

# Run Set 2
print("\n### Running Set 2: Data Size Comparison with 'all' Augmentations ###\n")
aug_name2 = experiments_set2["augmentations"][0]
for subset2 in experiments_set2["subset_fraction"]:
    run_experiment(subset2, aug_name2)

end_time = datetime.datetime.now()
print("#############################################")
print("## All experiments finished!")
print(f"## Started at:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"## Ended at:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"## Total Duration: {end_time - start_time}")
print("#############################################")