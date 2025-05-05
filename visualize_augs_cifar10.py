import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
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
        img_size = 32 # Default CIFAR size
        ax.imshow(np.zeros((img_size, img_size, 3)))
        ax.set_title(f"{title}\n(Error)")
    ax.axis('off')

# --- 1. Load CIFAR-10 Dataset ---
try:
    cifar_train_raw = CIFAR10(root='./data', train=True, download=True)
    print("CIFAR-10 dataset loaded/downloaded successfully.")
except Exception as e:
    print(f"Error loading/downloading CIFAR-10: {e}")
    exit()

# --- 2. Select 10 Random Images ---
num_images_total = len(cifar_train_raw)
random_indices = random.sample(range(num_images_total), 10)
sample_images_pil = [cifar_train_raw[i][0] for i in random_indices]

# --- 3. Display Random Images for Selection ---
print("Please choose the index (0-9) of the image you want to visualize:")
plt.figure(figsize=(15, 3))
for i, img_pil in enumerate(sample_images_pil):
    plt.subplot(1, 10, i + 1)
    plt.imshow(img_pil)
    plt.title(f"Index: {i}")
    plt.axis('off')
plt.suptitle("Randomly Selected CIFAR-10 Images")
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

# --- 5. Define Augmentation Pipelines (Same as before) ---
img_size = 32
kernel_size = int(0.1 * img_size)
if kernel_size % 2 == 0: kernel_size += 1

baseline_pil = [
    transforms.RandomCrop(img_size, padding=4, pad_if_needed=True),
    transforms.RandomHorizontalFlip(p=1.0), # Use p=1.0 for visualization consistency
]
jitter_tfm = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
blur_tfm = GaussianBlur(kernel_size=kernel_size)
gray_tfm = transforms.RandomGrayscale(p=1.0)
rotate_tfm = transforms.RandomRotation(30)
solarize_tfm = transforms.RandomSolarize(128, p=1.0)
erase_tfm = transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
to_tensor = transforms.ToTensor()

# Store pipelines without 'Original' for the grid plot
augmentation_pipelines_grid = {
    "Baseline": transforms.Compose(baseline_pil + [to_tensor]),
    "Jitter": transforms.Compose(baseline_pil + [transforms.RandomApply([jitter_tfm], p=1.0), to_tensor]),
    "Blur": transforms.Compose(baseline_pil + [transforms.RandomApply([blur_tfm], p=1.0), to_tensor]),
    "Grayscale": transforms.Compose(baseline_pil + [gray_tfm, to_tensor]),
    "Rotate": transforms.Compose(baseline_pil + [transforms.RandomApply([rotate_tfm], p=1.0), to_tensor]),
    "Solarize": transforms.Compose(baseline_pil + [solarize_tfm, to_tensor]),
    "Erase": transforms.Compose(baseline_pil + [to_tensor, erase_tfm]),
    "All": transforms.Compose(
        baseline_pil +
        [
            transforms.RandomApply([jitter_tfm], p=0.8),
            gray_tfm,
            transforms.RandomApply([blur_tfm], p=1.0),
            transforms.RandomApply([rotate_tfm], p=0.5),
            solarize_tfm,
            to_tensor,
            erase_tfm
        ]
    )
}
augmentation_names_grid = list(augmentation_pipelines_grid.keys()) # Should be 8 names

# --- 6. Apply Augmentations ---
augmented_image_tensors = {}
for name, pipeline in augmentation_pipelines_grid.items():
    try:
        augmented_image_tensors[name] = pipeline(original_image_pil)
    except Exception as e:
        print(f"Error applying augmentation '{name}': {e}")
        augmented_image_tensors[name] = torch.zeros(3, img_size, img_size) # Placeholder

# --- 7. Display Plot 1: Original Image ---
plt.figure(figsize=(4, 4)) # Adjust size as needed
imshow_single(original_image_pil, title=f"Original Image (Index: {chosen_index})")
plt.tight_layout()
plt.savefig("Original Image.png", dpi=300)
plt.show()

# --- 8. Display Plot 2: Augmented Images Grid (2x4) ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8)) # Rows=2, Cols=4
# Flatten axes array for easy iteration if needed, or use 2D indexing
axes_flat = axes.flatten()

for i, name in enumerate(augmentation_names_grid):
    display_tensor(axes_flat[i], augmented_image_tensors[name], name)

# Hide any unused subplots if the number of augmentations is not exactly 8
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].axis('off')

fig.suptitle(f"CIFAR-10 Augmentations Applied (Original Index: {chosen_index})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("CIFAR-10 Augmentations Applied.png", dpi=300)
plt.show()

print("\nVisualization complete. You should have two separate plots.")