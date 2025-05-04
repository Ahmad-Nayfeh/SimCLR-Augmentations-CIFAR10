# evaluation.py

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time
import datetime
import os

# Assuming utils.py is in the same directory or accessible
from utils import accuracy, append_csv, init_csv

def get_cifar10_eval_loaders(batch_size=512, num_workers=4, data_path='./data'):
    """Loads CIFAR-10 train/test datasets for linear evaluation."""
    eval_transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(data_path, train=True, download=True,
                                    transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, drop_last=False, shuffle=True)

    test_dataset = datasets.CIFAR10(data_path, train=False, download=True,
                                   transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_workers, drop_last=False, shuffle=False)

    print("CIFAR-10 evaluation datasets loaded.")
    return train_loader, test_loader

def load_backbone(checkpoint_path, device):
    """Loads the backbone from a SimCLR checkpoint."""
    print(f"Loading checkpoint for evaluation: {checkpoint_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'arch' not in checkpoint:
        raise KeyError("Architecture ('arch') not found in checkpoint.")
    arch = checkpoint['arch']
    print(f"Architecture from checkpoint: {arch}")

    num_classes = 10 # CIFAR-10
    if arch == 'resnet18':
        model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    elif arch == 'resnet50':
        model = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    state_dict = checkpoint['state_dict']
    cleaned_state_dict = {}
    expected_prefix = 'backbone.'
    # Adjust prefix if necessary based on how model was saved
    # Sometimes it might be nested further, e.g., 'model.backbone.'
    prefix_found = any(k.startswith(expected_prefix) for k in state_dict.keys())
    if not prefix_found:
         # Fallback check for slightly different saving patterns
         if any(k.startswith('model.' + expected_prefix) for k in state_dict.keys()):
             expected_prefix = 'model.' + expected_prefix
         else:
             # If no expected prefix, maybe it was saved without one? Log a warning.
             print(f"Warning: Expected prefix '{expected_prefix}' not found in state_dict keys. Assuming keys match standard ResNet.")
             expected_prefix = '' # Try loading directly

    for k, v in state_dict.items():
        if k.startswith(expected_prefix):
            new_key = k[len(expected_prefix):]
            cleaned_state_dict[new_key] = v
        elif not expected_prefix: # If trying direct loading
            cleaned_state_dict[k] = v

    load_log = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"Loading backbone weights summary:")
    print(f"  Missing keys: {load_log.missing_keys}")
    print(f"  Unexpected keys: {load_log.unexpected_keys}")

    # Verify only fc layer is missing
    expected_missing = {'fc.weight', 'fc.bias'}
    if set(load_log.missing_keys) != expected_missing:
        print(f"Warning: Unexpected missing keys found! {load_log.missing_keys}")

    model = model.to(device)
    return model, arch


def train_linear_classifier(model, train_loader, test_loader, device, epochs=100, lr=3e-4, weight_decay=0.0):
    """Trains the linear classifier on top of the frozen backbone."""
    eval_start_time = time.time()

    # Freeze all layers except the classifier
    for name, param in model.named_parameters():
        if not name.startswith('fc.'):
            param.requires_grad = False
        else:
            param.requires_grad = True # Ensure fc is trainable

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2 # Should be fc.weight, fc.bias
    print(f"Training linear classifier (parameters: fc.weight, fc.bias) for {epochs} epochs...")

    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    best_top1_accuracy = 0.0
    final_top1_accuracy = 0.0
    final_top5_accuracy = 0.0

    for epoch in range(epochs):
        model.train() # Set model to training mode
        top1_train_accuracy_epoch = 0
        train_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1_train = accuracy(logits, y_batch, topk=(1,))[0]
            top1_train_accuracy_epoch += top1_train.item()
            train_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_top1_train_accuracy_epoch = top1_train_accuracy_epoch / train_batches

        # Validation Phase
        model.eval() # Set model to evaluation mode
        top1_accuracy_val = 0
        top5_accuracy_val = 0
        val_batches = 0
        with torch.no_grad():
            for x_batch_val, y_batch_val in test_loader:
                x_batch_val, y_batch_val = x_batch_val.to(device), y_batch_val.to(device)
                logits_val = model(x_batch_val)
                top1, top5 = accuracy(logits_val, y_batch_val, topk=(1, 5))
                top1_accuracy_val += top1[0].item()
                top5_accuracy_val += top5[0].item()
                val_batches += 1

        final_top1_accuracy = top1_accuracy_val / val_batches
        final_top5_accuracy = top5_accuracy_val / val_batches

        if final_top1_accuracy > best_top1_accuracy:
             best_top1_accuracy = final_top1_accuracy

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1: # Print every 10 epochs and the last one
            print(f"  Eval Epoch {epoch+1}/{epochs}\tTrain Acc@1: {avg_top1_train_accuracy_epoch:.2f}%"
                  f"\tTest Acc@1: {final_top1_accuracy:.2f}%"
                  f"\tTest Acc@5: {final_top5_accuracy:.2f}%")

    eval_end_time = time.time()
    eval_time_seconds = eval_end_time - eval_start_time
    print(f"\nLinear evaluation training finished.")
    print(f"  Final Test Top-1 Accuracy: {final_top1_accuracy:.2f}%")
    print(f"  Final Test Top-5 Accuracy: {final_top5_accuracy:.2f}%")
    print(f"  Best Test Top-1 Accuracy: {best_top1_accuracy:.2f}%")
    print(f"  Evaluation Time: {datetime.timedelta(seconds=eval_time_seconds)}")

    return final_top1_accuracy, final_top5_accuracy, eval_time_seconds


def run_evaluation(args, checkpoint_path):
    """Runs the full linear evaluation pipeline and logs results."""
    print("\n--- Starting Linear Evaluation ---")

    # --- Setup ---
    device = args.device # Use device from main args
    accuracy_csv_path = 'linear_probing_accuracy_results.csv' # Central CSV file
    accuracy_csv_headers = [
        'run_id', 'timestamp', 'augmentation_config', 'backbone',
        'top1_accuracy', 'top5_accuracy',
        'total_pretrain_time_seconds', 'eval_time_seconds'
    ]
    init_csv(accuracy_csv_path, accuracy_csv_headers)
    print(f"Accuracy results will be logged to: {accuracy_csv_path}")
    # --- /Setup ---

    # --- Load Data ---
    eval_train_loader, eval_test_loader = get_cifar10_eval_loaders(
        batch_size=args.batch_size, # Use same batch size for consistency? Or maybe larger like 512? Let's try 512.
        num_workers=args.workers,
        data_path=args.data
    )
    # --- /Load Data ---

    # --- Load Model ---
    try:
        model, backbone_arch = load_backbone(checkpoint_path, device)
    except Exception as e:
        print(f"Error loading backbone: {e}. Skipping evaluation.")
        return
    # --- /Load Model ---

    # --- Train Classifier ---
    try:
        top1_acc, top5_acc, eval_time_sec = train_linear_classifier(
            model, eval_train_loader, eval_test_loader, device,
            epochs=100 # Standard 100 epochs for linear eval
        )
    except Exception as e:
        print(f"Error during linear classifier training: {e}. Skipping logging.")
        # Optionally log the error state if needed
        top1_acc, top5_acc, eval_time_sec = -1.0, -1.0, -1.0 # Indicate failure
    # --- /Train Classifier ---

    # --- Log Results ---
    log_data = {
        'run_id': args.run_id_,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'augmentation_config': args.augmentation_config_string_,
        'backbone': backbone_arch,
        'top1_accuracy': f"{top1_acc:.2f}",
        'top5_accuracy': f"{top5_acc:.2f}",
        'total_pretrain_time_seconds': f"{args.total_train_time_seconds_:.2f}",
        'eval_time_seconds': f"{eval_time_sec:.2f}"
    }

    try:
        append_csv(accuracy_csv_path, log_data)
        print(f"Successfully logged evaluation results for run {args.run_id_}")
    except Exception as e:
        print(f"Error logging evaluation results to CSV: {e}")
    # --- /Log Results ---

    print("--- Linear Evaluation Finished ---")