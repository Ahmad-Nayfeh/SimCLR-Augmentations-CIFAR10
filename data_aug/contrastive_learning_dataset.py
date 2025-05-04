# contrastive_learning_dataset.py

import torch
from torch.utils.data import Subset
from torchvision.transforms import transforms
from torchvision import datasets
# Note: Assuming gaussian_blur.py defining GaussianBlur class exists in data_aug folder
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import numpy as np # Needed for subsetting if using numpy for indices

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, active_augmentations=[]):
        """
        Return a set of data augmentation transformations based on the active_augmentations list.
        Args:
            size (int): Target image size.
            active_augmentations (list): List of strings specifying which augmentations to apply
                                         beyond the baseline. Options: 'jitter', 'blur', 'gray',
                                         'rotate', 'solarize', 'erase'. 'baseline' is always applied.
        Returns:
            torchvision.transforms.Compose: Composition of transformations.
        """
        # --- Baseline Augmentations ---
        # Replace RandomResizedCrop with RandomCrop for CIFAR-10
        augmentation_list = [
            transforms.RandomCrop(size, padding=4), # CIFAR-10 is 32x32
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        # --- Variant Augmentations ---
        s = 1.0 # Strength parameter, kept from original code for jitter reference

        # 1. Color Jitter
        if 'jitter' in active_augmentations:
            color_jitter = transforms.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s)
            augmentation_list.append(transforms.RandomApply([color_jitter], p=0.8))

        # 2. Grayscale
        if 'gray' in active_augmentations:
            # RandomGrayscale already has probability p
            augmentation_list.append(transforms.RandomGrayscale(p=0.2))

        # 3. Gaussian Blur
        if 'blur' in active_augmentations:
            # Original code applied blur always. We need RandomApply for p=0.5
            # kernel_size is calculated based on image size (e.g., 3 for 32x32)
            kernel_size = int(0.1 * size)
            if kernel_size % 2 == 0: # Ensure kernel size is odd
                 kernel_size += 1
            gaussian_blur = GaussianBlur(kernel_size=kernel_size) # sigma is random [0.1, 2.0] within this class
            augmentation_list.append(transforms.RandomApply([gaussian_blur], p=0.5))

        # 4. Random Rotation
        if 'rotate' in active_augmentations:
            # Assuming 30 degrees max rotation, apply with p=0.5
            augmentation_list.append(transforms.RandomApply([transforms.RandomRotation(30)], p=0.5))

        # 5. Random Solarize
        if 'solarize' in active_augmentations:
            # Assuming threshold 128, apply with p=0.2
            augmentation_list.append(transforms.RandomSolarize(128, p=0.2))

        # 6. Random Erasing
        if 'erase' in active_augmentations:
             # Using torchvision defaults for scale/ratio, apply with p=0.5
             augmentation_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False))


        # --- Finalization ---
        augmentation_list.append(transforms.ToTensor())

        data_transforms = transforms.Compose(augmentation_list)
        print(f"Applied Augmentations: {augmentation_list}") # For verification
        return data_transforms

    def get_dataset(self, name, n_views, subset_fraction=1.0, active_augmentations=[]):
        """
        Gets the specified dataset, applies contrastive view generation, and optionally subsets it.

        Args:
            name (str): Name of the dataset ('cifar10' or 'stl10').
            n_views (int): Number of contrastive views to generate.
            subset_fraction (float): Fraction of the dataset to use (0.0 to 1.0]. Default is 1.0 (full dataset).
            active_augmentations (list): List of augmentation names to enable.
        Returns:
            torch.utils.data.Dataset: The final dataset object.
        """
        valid_datasets = {
            'cifar10': lambda: self._get_cifar10_dataset(n_views, subset_fraction, active_augmentations),
            'stl10': lambda: self._get_stl10_dataset(n_views, subset_fraction, active_augmentations) # Added subsetting capability here too
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()

    def _get_cifar10_dataset(self, n_views, subset_fraction, active_augmentations):
        """ Helper function specifically for CIFAR-10 """
        # Create the augmentation pipeline
        pipeline = self.get_simclr_pipeline_transform(size=32, active_augmentations=active_augmentations)
        # Create the view generator wrapper
        transform_wrapper = ContrastiveLearningViewGenerator(pipeline, n_views)

        # Load the full dataset with the transform
        full_transformed_dataset = datasets.CIFAR10(self.root_folder, train=True,
                                                    transform=transform_wrapper,
                                                    download=True)

        if subset_fraction < 1.0:
            num_samples = int(len(full_transformed_dataset) * subset_fraction)
            # Use torch.randperm for random indices. For perfect reproducibility across runs
            # of the *subset selection*, you might want a fixed np.random.choice or similar.
            indices = torch.randperm(len(full_transformed_dataset))[:num_samples]
            print(f"Using {num_samples} samples from CIFAR-10 train set ({subset_fraction*100:.1f}%).")
            dataset = Subset(full_transformed_dataset, indices)
        else:
            print(f"Using full CIFAR-10 train set ({len(full_transformed_dataset)} samples).")
            dataset = full_transformed_dataset

        return dataset

    def _get_stl10_dataset(self, n_views, subset_fraction, active_augmentations):
        """ Helper function specifically for STL10 """
        # Create the augmentation pipeline
        pipeline = self.get_simclr_pipeline_transform(size=96, active_augmentations=active_augmentations)
        # Create the view generator wrapper
        transform_wrapper = ContrastiveLearningViewGenerator(pipeline, n_views)

        # Load the full dataset with the transform ('unlabeled' split for SimCLR pre-training)
        full_transformed_dataset = datasets.STL10(self.root_folder, split='unlabeled',
                                                 transform=transform_wrapper,
                                                 download=True)

        if subset_fraction < 1.0:
            num_samples = int(len(full_transformed_dataset) * subset_fraction)
            indices = torch.randperm(len(full_transformed_dataset))[:num_samples]
            print(f"Using {num_samples} samples from STL-10 unlabeled set ({subset_fraction*100:.1f}%).")
            dataset = Subset(full_transformed_dataset, indices)
        else:
            print(f"Using full STL-10 unlabeled set ({len(full_transformed_dataset)} samples).")
            dataset = full_transformed_dataset

        return dataset