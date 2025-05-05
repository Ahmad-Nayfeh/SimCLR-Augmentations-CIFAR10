import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import STL10 # Changed dataset
import matplotlib.pyplot as plt
import numpy as np
import random
import math # Needed for GaussianBlur

# --- Define GaussianBlur Class (copied from previous script) ---
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.blur = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        is_pil = not isinstance(img, torch.Tensor)
        if is_pil:
            img_tensor = self.pil_to_tensor(img).unsqueeze(0)
        else:
             img_tensor = img.unsqueeze(0) if len(img.shape) == 3 else img

        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))
        with torch.no_grad():
            if isinstance(img_tensor, torch.Tensor):
                 img_tensor = self.blur(img_tensor)
                 img_tensor = img_tensor.squeeze()
            else:
                 print("Warning: GaussianBlur skipping non-tensor input.")
        if is_pil:
            img_out = self.tensor_to_pil(img_tensor)
        else:
            img_out = img_tensor
        return img_out
# --- End GaussianBlur Class ---

def imshow_single(img_pil, title=None):
    """Helper function to display a single PIL image"""
    plt.imshow(img_pil)
    if title:
        plt.title(title)
    plt.axis('off')

def display_tensor(ax, tensor_img, title):
    """Helper function to display a tensor image safely on a subplot axis"""
    if tensor_img is not None and isinstance(tensor_img, torch.Tensor):
        img_display = tensor_img.cpu().detach()
        img_display = torch.clamp(img_display, 0, 1)
        ax.imshow(np.transpose(img_display.numpy(), (1, 2, 0)))
        ax.set_title(title)
    else:
        img_size = 96 # Default STL-10 size
        ax.imshow(np.zeros((img_size, img_size, 3)))
        ax.set_title(f"{title}\n(Error)")
    ax.axis('off')

# --- 1. Load STL-10 Dataset ---
try:
    stl10_unlabeled_raw = STL10(root='./data', split='unlabeled', download=True)
    print("STL-10 unlabeled dataset loaded/downloaded successfully.")
except Exception as e:
    print(f"Error loading/downloading STL-10: {e}")
    exit()

# --- 2. Select 10 Random Images ---
num_images_total = len(stl10_unlabeled_raw)
random_indices = random.sample(range(num_images_total), 10)
sample_images_pil = [stl10_unlabeled_raw[i][0] for i in random_indices]

# --- 3. Display Random Images for Selection ---
print("Please choose the index (0-9) of the image you want to visualize:")
plt.figure(figsize=(20, 4))
for i, img_pil in enumerate(sample_images_pil):
    plt.subplot(1, 10, i + 1)
    plt.imshow(img_pil)
    plt.title(f"Index: {i}")
    plt.axis('off')
plt.suptitle("Randomly Selected STL-10 Images (Unlabeled Split)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- 4. Get User Input for ONE image ---
chosen_index = -1
while chosen_index == -1:
    try:
        idx_str = input(f"Enter index for the image (0-9): ")
        idx = int(idx_str)
        if 0 <= idx <= 9:
            chosen_index = idx
        else:
            print("Invalid index. Please enter a number between 0 and 9.")
    except ValueError:
        print("Invalid input. Please enter a number.")

original_image_pil = sample_images_pil[chosen_index]
print(f"Selected image with index: {chosen_index}")

# --- 5. Define Augmentation Pipelines (Based on your STL-10 Report) ---
img_size = 96 # STL-10 image size

# Baseline components (PIL ops) - From your report Section III.C
baseline_pil = [
    transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5), # Use p=0.5 from report for consistency
]
# Individual augmentations (PIL ops) - Parameters from report Section III.C
jitter_tfm = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
blur_tfm = GaussianBlur(kernel_size=9, sigma_min=0.1, sigma_max=2.0)
gray_tfm = transforms.RandomGrayscale(p=0.2) # p=0.2 from report
# Conversion
to_tensor = transforms.ToTensor()

# Store pipelines including 'Original' for the grid plot
# Note: Using p=1.0 in RandomApply for visualization guarantees effect is shown
augmentation_pipelines_grid = {
    "Original": transforms.Compose([to_tensor]), # Just convert original to tensor
    "Baseline": transforms.Compose(baseline_pil + [to_tensor]),
    "Color": transforms.Compose(
        baseline_pil +
        [transforms.RandomApply([jitter_tfm], p=1.0), # Use p=1.0 for vis
         to_tensor]
    ),
    "Blur": transforms.Compose(
        baseline_pil +
        [transforms.RandomApply([blur_tfm], p=1.0), # Use p=1.0 for vis
         to_tensor]
    ),
    "Gray": transforms.Compose(
        baseline_pil +
        [transforms.RandomGrayscale(p=1.0), # Use p=1.0 for vis
         to_tensor]
    ),
    "All": transforms.Compose( # Combined pipeline from report
        baseline_pil +
        [
            transforms.RandomApply([jitter_tfm], p=0.8), # Keep original p=0.8
            gray_tfm, # Keep original p=0.2
            transforms.RandomApply([blur_tfm], p=0.5), # Keep original p=0.5
            to_tensor
        ]
    )
}
augmentation_names_grid = list(augmentation_pipelines_grid.keys()) # Should be 6 names

# --- 6. Apply Augmentations ---
augmented_image_tensors = {}
for name, pipeline in augmentation_pipelines_grid.items():
    try:
        # Apply the defined pipeline
        augmented_tensor = pipeline(original_image_pil)
        # Double-check if it's a tensor
        if not isinstance(augmented_tensor, torch.Tensor):
            print(f"Warning: Pipeline '{name}' did not return a tensor. Converting.")
            augmented_image_tensors[name] = to_tensor(augmented_tensor)
        else:
            augmented_image_tensors[name] = augmented_tensor
    except Exception as e:
        print(f"Error applying augmentation '{name}': {e}")
        augmented_image_tensors[name] = torch.zeros(3, img_size, img_size) # Placeholder

# --- 7. Display Plot 1: Original Image ---
plt.figure(figsize=(5, 5)) # Adjust size as needed for 96x96
imshow_single(original_image_pil, title=f"Original Image (Index: {chosen_index})")
plt.tight_layout()
plt.savefig("STL-10 AOriginal Image.png", dpi=300)
plt.show()

# --- 8. Display Plot 2: Augmented Images Grid (2x3) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10)) # Rows=2, Cols=3
# Flatten axes array for easy iteration
axes_flat = axes.flatten()

# Display Original + 5 augmentations in the 2x3 grid
for i, name in enumerate(augmentation_names_grid):
    if i < len(axes_flat): # Ensure we don't exceed grid size
        display_tensor(axes_flat[i], augmented_image_tensors[name], name)

# Hide any unused subplots if the number of items < 6
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis('off')

fig.suptitle(f"STL-10 Augmentations Applied (Original Index: {chosen_index})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("STL-10 Augmentations Applied.png", dpi=300)
plt.show()

print("\nVisualization complete. You should have two separate plots.")