# utils.py

import os
import shutil
import torch
import yaml
import csv # Added import

# --- CSV Helper Functions ---

def init_csv(file_path, headers):
    """Creates a CSV file and writes the header row if the file doesn't exist."""
    # Check if file exists to prevent overwriting headers
    file_exists = os.path.isfile(file_path)
    # Use 'a' mode (append) to create file if not exists, and allow appending
    # newline='' prevents extra blank rows in CSV on Windows
    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        # Write header only if the file did not exist before opening
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()

def append_csv(file_path, row_dict):
    """Appends a dictionary as a row to the specified CSV file."""
    # Assumes file and headers already exist (call init_csv first)
    headers = list(row_dict.keys())
    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writerow(row_dict)

# --- /CSV Helper Functions ---


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves model checkpoint."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    """Saves command line arguments to a YAML file."""
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    # Convert args namespace to dict for saving non-primitive types if any
    args_dict = vars(args)
    # Clean up non-serializable items if necessary (e.g., device objects)
    args_to_save = {}
    for key, value in args_dict.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
             # Add other serializable types if needed
             args_to_save[key] = value
        else:
             args_to_save[key] = str(value) # Convert others to string representation


    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        try:
            yaml.dump(args_to_save, outfile, default_flow_style=False)
        except yaml.representer.RepresenterError:
             print("Warning: Could not save all arguments to config.yml due to non-serializable types.")
             yaml.dump({k: str(v) for k, v in args_to_save.items()}, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) # Original had reshape
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True) # Added contiguous() for safety
            res.append(correct_k.mul_(100.0 / batch_size))
        return res