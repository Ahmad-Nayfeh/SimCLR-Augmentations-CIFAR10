# simclr.py

import logging
import os
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# Import new CSV helper and save_config_file, accuracy, save_checkpoint
from utils import save_config_file, accuracy, save_checkpoint, append_csv

import time       # Added for epoch timing
import datetime   # Added for timestamp in logging

# torch.manual_seed(0) # Seed is now set in run.py if provided

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        # Use the log_dir passed via args for TensorBoard and log file
        self.writer = SummaryWriter(log_dir=self.args.log_dir_)
        log_file_path = os.path.join(self.args.log_dir_, 'training.log')
        print(f"Console logging to: {log_file_path}")
        # Clear previous handlers if any, to avoid duplicate logs on re-runs in same session
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO, # Changed default level to INFO for cleaner console output
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler(sys.stdout) # Also print logs to console
            ]
        )

        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

        # --- Get logging info from args ---
        self.run_id = self.args.run_id_
        self.augmentation_config = self.args.augmentation_config_string_
        self.loss_csv_path = self.args.loss_csv_path_
        # --- /Get logging info ---


    def info_nce_loss(self, features):
        # Correct calculation for NT-Xent loss

        # features shape: [n_views * batch_size, feature_dim]
        # Example: [2*128, 128]
        n_views = self.args.n_views
        batch_size_eff = features.shape[0] // n_views # Effective batch size

        labels = torch.cat([torch.arange(batch_size_eff) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        # labels shape: [n_views * batch_size, n_views * batch_size]

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # similarity_matrix shape: [n_views * batch_size, n_views * batch_size]

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        # labels shape: [N, N], similarity_matrix shape: [N, N] where N = n_views * batch_size
        labels = labels[~mask].view(labels.shape[0], -1)
        # labels shape: [N, N-1]
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # similarity_matrix shape: [N, N-1]

        # select and combine multiple positives
        # For n_views=2, there is only one positive counterpart, resulting in shape [N, 1]
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        # Logits 'labels' are all zeros, indicating the first column (positives) is the target
        nt_xent_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, nt_xent_labels # Renamed labels for clarity

    def train(self, train_loader):
        """Trains the SimCLR model and returns the path to the saved checkpoint.""" # Modified docstring

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file (arguments) in the run's log directory
        save_config_file(self.args.log_dir_, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with device: {self.args.device}.")
        logging.info(f"Run ID: {self.run_id}")
        logging.info(f"Logging TensorBoard to: {self.args.log_dir_}")

        logging.info(f"Optimizer: {self.optimizer}")
        logging.info(f"Scheduler: {self.scheduler.__class__.__name__}")


        for epoch_counter in range(self.args.epochs):
            epoch_start_time = time.time() # Start epoch timer
            epoch_loss = 0.0
            epoch_top1_acc = 0.0 # Accumulate within-batch accuracy

            # Use tqdm for progress bar
            batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch_counter}/{self.args.epochs}", unit="batch", leave=True) # Keep bar after loop

            for images, _ in batch_iterator:
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features) # labels here are targets for CE loss (all zeros)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Accumulate loss and accuracy for epoch average
                epoch_loss += loss.item()
                # Calculate accuracy based on the NT-Xent logits/labels
                top1, _ = accuracy(logits, labels, topk=(1, 5))
                epoch_top1_acc += top1[0].item() # Get Python number

                # Log metrics periodically to TensorBoard (and potentially console via tqdm)
                if n_iter % self.args.log_every_n_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0] # Use get_last_lr()
                    self.writer.add_scalar('Loss/train_step', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('Accuracy/top1_contr_step', top1[0].item(), global_step=n_iter) # Clarify accuracy type
                    self.writer.add_scalar('Misc/learning_rate', current_lr, global_step=n_iter)
                    # Update tqdm description
                    batch_iterator.set_postfix(loss=loss.item(), acc=top1[0].item(), lr=f"{current_lr:.1e}")


                n_iter += 1
                # Step the scheduler *after* the optimizer step, at each step
                self.scheduler.step()


            # --- End of Epoch Logging ---
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            if len(train_loader) > 0:
                 avg_epoch_loss = epoch_loss / len(train_loader)
                 avg_epoch_top1_acc = epoch_top1_acc / len(train_loader)
            else:
                 avg_epoch_loss = 0.0
                 avg_epoch_top1_acc = 0.0


            # Log average epoch metrics to TensorBoard
            self.writer.add_scalar('Loss/train_epoch', avg_epoch_loss, global_step=epoch_counter)
            self.writer.add_scalar('Accuracy/top1_contr_epoch', avg_epoch_top1_acc, global_step=epoch_counter) # Clarify accuracy type

            # Log epoch metrics to CSV
            log_data = {
                'run_id': self.run_id,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'augmentation_config': self.augmentation_config,
                'backbone': self.args.arch,
                'subset_fraction': self.args.subset_fraction,
                'epoch': epoch_counter,
                'loss': f"{avg_epoch_loss:.6f}", # Format loss for consistency
                'epoch_time_seconds': f"{epoch_duration:.2f}" # Format time
            }
            append_csv(self.loss_csv_path, log_data)

            logging.info(f"Epoch: {epoch_counter}\tAvg Loss: {avg_epoch_loss:.4f}\tAvg Top1 Acc (contrastive): {avg_epoch_top1_acc:.2f}%\tDuration: {datetime.timedelta(seconds=epoch_duration)}")
            # --- /End of Epoch Logging ---


        logging.info("Training has finished.")
        # Save model checkpoint in the run's log directory
        checkpoint_name = f'checkpoint_{self.args.epochs:04d}.pth.tar'
        checkpoint_path = os.path.join(self.args.log_dir_, checkpoint_name)

        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(), # Save scheduler state too
            'args': vars(self.args) # Save args Namespace as dict if needed
        }, is_best=False, filename=checkpoint_path) # Pass the full path

        logging.info(f"Model checkpoint and metadata has been saved at: {checkpoint_path}") # Log final path

        # Return the path to the saved checkpoint
        return checkpoint_path