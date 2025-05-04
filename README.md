# 🔬 SimCLR (Modified for Augmentation Studies on CIFAR-10)

This repository is built based on the original SimCLR framework by [sthalles](https://github.com/sthalles/SimCLR), and has been **enhanced** to support:

- Training on **subsets of CIFAR-10** (e.g., 25%, 50%)
- Flexible control over **data augmentations** via command-line
- Built-in **Linear Evaluation** (Top-1 / Top-5 Accuracy)
- Centralized **CSV Logging** of loss and accuracy for easy analysis
- Automatic **run ID tracking** and **organized logs** per experiment

---
## 📁 Repository Overview

```

SimCLR/  
├── run.py # Main script: pre-training + linear evaluation  
├── simclr.py # SimCLR training pipeline (with logging)  
├── evaluation.py # Linear probing evaluation script  
├── utils.py # Logging helpers (CSV, config)  
├── data_aug/  
│ └── contrastive_learning_dataset.py # Augmentation & dataset logic  
├── models/  
│ └── resnet_simclr.py  
├── runs/ # Auto-created. Logs, checkpoints per run  
├── simclr_loss_results.csv # Global CSV logging SimCLR loss (auto-appended)  
├── linear_probing_accuracy_results.csv # Global CSV logging final eval accuracy  
├── requirements.txt  
├── README.md # ← You are here

````

---
## ⚙️ Key Modifications from Original SimCLR

> Based on changes documented in `Summary of the modifications.md`

### ✅ Dataset Subsetting
- `--subset_fraction` lets you train on only a portion of CIFAR-10.
- Dynamically uses `torch.utils.data.Subset` after applying SimCLR views.

### ✅ Augmentation Flexibility
- Augmentations are configurable via the `--augmentations` flag:
  - `jitter`, `blur`, `gray`, `rotate`, `solarize`, `erase`
- All augmentations are built on top of a **baseline**:
  - `RandomCrop(32, padding=4)` + `RandomHorizontalFlip(p=0.5)`

### ✅ Augmentation Pipeline is Modular
- You can choose:
  - `--augmentations baseline`
  - `--augmentations blur,jitter`
  - `--augmentations all` (to apply all 6 variants)

### ✅ Logging Enhancements
- Every run gets a unique `run_id` and folder inside `runs/`
- Logs include:
  - `config.yml`, `training.log`, TensorBoard summaries
  - Final `checkpoint_*.pth.tar`

### ✅ CSV Logs for Comparison
- `simclr_loss_results.csv`: Logs epoch-wise loss, time, augmentations, backbone
- `linear_probing_accuracy_results.csv`: Logs Top-1 / Top-5 accuracy, timings, config

### ✅ Integrated Linear Evaluation
- No need to run a separate notebook
- By default, `run.py` will:
  - Train SimCLR
  - Load frozen encoder
  - Train + test linear classifier
  - Log results

---
## 🚀 How to Use

### ✅ 1. Setup
```bash
pip install -r requirements.txt
````

_Make sure PyTorch + torchvision are installed correctly._

---
### ✅ 2. Run Your First Experiment

Example:

```bash
python run.py --arch resnet18 --augmentations baseline --subset_fraction 0.25 --epochs 50 --batch-size 128
```

---
## 🔧 CLI Arguments Overview

|Category|Flag|Description|
|---|---|---|
|**Dataset**|`--dataset-name`|`cifar10` or `stl10` (default: `cifar10`)|
||`--subset_fraction`|Fraction of dataset to use (e.g., `0.25`)|
||`-data`|Path to dataset root (default: `./datasets`)|
|**Augmentations**|`--augmentations`|Comma-separated list: `jitter,blur,gray,rotate,solarize,erase`|
|||Special: `baseline`, `all`|
|**Model**|`--arch`|`resnet18` or `resnet50`|
||`--out_dim`|Output dim of projection head (default: 128)|
|**Training**|`--epochs`|Number of training epochs (default: 50)|
||`--batch-size`|Batch size (default: 128)|
||`--learning-rate` or `--lr`|Initial LR for Adam (default: 0.0003)|
||`--weight-decay` or `--wd`|Weight decay (default: 1e-4)|
||`--temperature`|NT-Xent temperature (default: 0.07)|
||`--fp16-precision`|Use mixed precision training|
|**Evaluation**|`--run_linear_eval`|Enable evaluation (default: True)|
||`--no_linear_eval`|Disable evaluation|
|**Misc**|`--workers` or `-j`|DataLoader workers (default: 4)|
||`--seed`|Seed for reproducibility|
||`--disable-cuda`|Force CPU mode|
||`--gpu-index`|Which GPU to use (default: 0)|
||`--log-every-n-steps`|Logging frequency (default: 100)|

---
## 🧪 Examples

### 🔹 Run ResNet-18 with Baseline + Blur + Jitter on 25% of CIFAR-10:

```bash
python run.py --arch resnet18 --augmentations blur,jitter --subset_fraction 0.25 --epochs 50
```

### 🔹 Run ResNet-50 with All Augmentations on Full CIFAR-10:

```bash
python run.py --arch resnet50 --augmentations all --subset_fraction 1.0 --epochs 50
```

### 🔹 Skip evaluation (only pre-train):

```bash
python run.py --arch resnet18 --augmentations rotate,solarize --no_linear_eval
```

---
## 📊 Outputs

- **`runs/<run_id>/`** — Logs, config, TensorBoard, model checkpoint
- **`simclr_loss_results.csv`** — Aggregated SimCLR loss across runs
- **`linear_probing_accuracy_results.csv`** — Final accuracy for each run
- **Console Logs** — Shows all training + eval progress with `tqdm` and `logging`

---
## 🤝 Credits

Built on top of:

- [sthalles/SimCLR](https://github.com/sthalles/SimCLR)

Extended and maintained by **Ahmad Nayfeh** for research on augmentation effects in SSL for **Digital Image Processing @ KFUPM**.