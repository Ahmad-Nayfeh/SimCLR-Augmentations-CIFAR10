# run.py

import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from utils import init_csv # Import the CSV helper
from evaluation import run_evaluation # Import the evaluation function

import datetime # Added for timestamp and timing
import os       # Added for path joining
import time     # Added for timing
import logging  # Import logging
import sys      # Import sys
import gc       # Import garbage collector

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='cifar10', # Changed default to cifar10
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=['resnet18', 'resnet50'], # Explicitly list supported architectures
                    help='model architecture: resnet18 | resnet50 (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', # Adjusted default for Colab
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', # Adjusted default
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, # Adjusted default
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

# --- Arguments from Phase 1 ---
parser.add_argument('--subset_fraction', default=1.0, type=float,
                    help='Fraction of the dataset to use (0.0 to 1.0]. Default 1.0')
parser.add_argument('--augmentations', default='baseline', type=str,
                    help="Comma-separated list of augmentations to enable beyond baseline "
                         "(e.g., 'baseline,jitter,blur' or 'all'). "
                         "Options: baseline,jitter,blur,gray,rotate,solarize,erase,all")
# --- /Arguments ---

# --- New Argument for Evaluation ---
parser.add_argument('--run_linear_eval', action='store_true', default=True, # Enable by default
                    help='Run linear evaluation after pre-training.')
parser.add_argument('--no_linear_eval', action='store_false', dest='run_linear_eval',
                    help='Disable linear evaluation after pre-training.')
# --- /New Argument ---


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # --- Set Seed if provided ---
    if args.seed is not None:
        torch.manual_seed(args.seed)
        # Potentially add numpy/random seeds if needed elsewhere
        print(f"Using manual seed: {args.seed}")
    # --- /Set Seed ---

    # --- Parse Augmentations ---
    valid_augs = {'jitter', 'blur', 'gray', 'rotate', 'solarize', 'erase'}
    if args.augmentations.lower() == 'all':
        active_augmentations = sorted(list(valid_augs))
        args.augmentation_config_string_ = 'all'
    elif args.augmentations.lower() == 'baseline':
         active_augmentations = []
         args.augmentation_config_string_ = 'baseline'
    else:
        req_augs = {aug.strip().lower() for aug in args.augmentations.split(',')}
        active_augmentations = sorted(list(req_augs.intersection(valid_augs)))
        args.augmentation_config_string_ = '+'.join(active_augmentations) if active_augmentations else 'baseline'

    args.active_augmentations_list_ = active_augmentations
    # --- /Parse Augmentations ---

    # --- Generate Run ID ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.run_id_ = f"run_{timestamp}_{args.arch}_{args.augmentation_config_string_}_ep{args.epochs}_bs{args.batch_size}"
    # --- /Generate Run ID ---

    # --- Setup Logging ---
    log_dir = os.path.join('runs', args.run_id_)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir_ = log_dir
    # Configure root logger
    log_file_path = os.path.join(args.log_dir_, 'main_run.log')
    for handler in logging.root.handlers[:]: # Clear existing handlers
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])

    logging.info(f"Run ID: {args.run_id_}")
    logging.info(f"Requested augmentations raw: '{args.augmentations}'")
    logging.info(f"Active augmentations (beyond baseline): {active_augmentations}")
    logging.info(f"Augmentation configuration string: {args.augmentation_config_string_}")
    logging.info(f"Log directory: {args.log_dir_}")

    # Define CSV file path (place it outside the run-specific folder to aggregate results)
    args.loss_csv_path_ = 'simclr_loss_results.csv'
    loss_csv_headers = ['run_id', 'timestamp', 'augmentation_config', 'backbone', 'subset_fraction', 'epoch', 'loss', 'epoch_time_seconds']
    init_csv(args.loss_csv_path_, loss_csv_headers)
    logging.info(f"Logging SimCLR loss results to: {args.loss_csv_path_}")
    # --- /Setup Logging ---

    # --- Device Setup ---
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True if args.seed is not None else False
        cudnn.benchmark = False if args.seed is not None else True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    logging.info(f"Using device: {args.device}")
    # --- /Device Setup ---

    # --- Data Loading (Pre-training) ---
    dataset_pt = ContrastiveLearningDataset(args.data)
    train_dataset_pt = dataset_pt.get_dataset(args.dataset_name, args.n_views,
                                        subset_fraction=args.subset_fraction,
                                        active_augmentations=active_augmentations)
    train_loader_pt = torch.utils.data.DataLoader(
        train_dataset_pt, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    # --- /Data Loading (Pre-training) ---


    # --- Model, Optimizer, Scheduler (Pre-training) ---
    model_pt = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    optimizer_pt = torch.optim.Adam(model_pt.parameters(), args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader_pt) * args.epochs
    logging.info(f"Total pre-training steps: {total_steps}")
    scheduler_pt = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pt, T_max=total_steps, eta_min=0, last_epoch=-1)
    # --- /Model, Optimizer, Scheduler (Pre-training) ---

    # --- Pre-training ---
    logging.info(f"Starting SimCLR pre-training for {args.epochs} epochs...")
    start_time = time.time()

    simclr_trainer = SimCLR(model=model_pt, optimizer=optimizer_pt, scheduler=scheduler_pt, args=args)
    # Train and get the path to the saved checkpoint
    final_checkpoint_path = simclr_trainer.train(train_loader_pt)

    end_time = time.time()
    total_training_time_seconds = end_time - start_time
    args.total_train_time_seconds_ = total_training_time_seconds # Store for logging
    logging.info(f"Pre-training finished.")
    logging.info(f"Total Pre-training Time: {datetime.timedelta(seconds=total_training_time_seconds)}")
    logging.info(f"Final checkpoint saved at: {final_checkpoint_path}")
    # --- /Pre-training ---


    # --- Linear Evaluation (Optional) ---
    if args.run_linear_eval:
        # --- Optional: Cleanup GPU memory before evaluation ---
        logging.info("Cleaning up pre-training objects from memory...")
        del simclr_trainer, model_pt, optimizer_pt, scheduler_pt, train_loader_pt, train_dataset_pt, dataset_pt
        gc.collect() # Run garbage collection
        if args.device == torch.device('cuda'):
            torch.cuda.empty_cache()
            logging.info("CUDA cache cleared.")
        # --- /Optional: Cleanup ---

        # Run evaluation using the function from evaluation.py
        run_evaluation(args=args, checkpoint_path=final_checkpoint_path)
    else:
        logging.info("Skipping linear evaluation as per arguments.")
    # --- /Linear Evaluation ---

    logging.info(f"Run {args.run_id_} finished.")


if __name__ == "__main__":
    main()