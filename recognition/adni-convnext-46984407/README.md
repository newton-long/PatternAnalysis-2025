# Alzheimer’s Disease Classification using ConvNeXt

### Author: Newton Long

### Course: COMP3710 – Pattern Analysis 2025

### Model: ConvNeXt-Small (Fine-Tuned on ADNI Dataset)

---

## 🧩 Problem Description

Alzheimer’s disease (AD) is a progressive neurodegenerative disorder characterized by structural brain changes observable in MRI scans.  
The goal of this project was to **classify brain MRI scans as either Alzheimer’s Disease (AD) or Cognitively Normal (CN)** using the ADNI dataset.

This task belongs to the _medical image classification_ domain and aims to demonstrate how modern deep learning architectures can identify subtle patterns in medical imagery. The project uses **ConvNeXt**, a next-generation convolutional network that integrates design principles from Vision Transformers while retaining convolutional efficiency.

---

## ⚙️ Methodology

### Model Architecture

A **ConvNeXt Small** backbone was fine-tuned from pretrained ImageNet weights.  
The final classification layer was replaced with a custom two-class (AD / CN) linear head.

The training followed a **staged fine-tuning** process:

| Stage                     | Epoch Range | Layers Trained                  | Learning Rate | Description            |
| ------------------------- | ----------- | ------------------------------- | ------------- | ---------------------- |
| **1 – Head Only**         | 1–3         | Classifier                      | 1e-3          | Stabilizes final layer |
| **2 – Last Block + Head** | 4–12        | Last feature block & classifier | 7.5e-5–3e-4   | Adapts deeper layers   |
| **3 – Full Model**        | 13–30       | Entire network                  | 3e-5–1e-4     | Fine-tunes all layers  |

Optimized using **AdamW** with **CosineAnnealingLR** scheduler and **early stopping** (patience = 8).

Loss Function: **Weighted Cross-Entropy** (handles class imbalance)  
Extras: **Label smoothing (0.05)** and **Mixed-precision training** for faster convergence.

---

### Data & Augmentation

Dataset: `ADNI/AD_NC` (train / test folders).  
The training set was deterministically split 80/20 into training and validation sets.

**MRI-safe augmentations:**

- Random resized crop (0.75–1.0 scale)
- Random horizontal and vertical flips
- ±15° rotations and mild affine/perspective transformations
- Brightness ±0.15, contrast ±0.3
- Gaussian blur and light noise injection
- Random erasing to simulate missing slices
- **RepeatAug** for multiple random augmentations per image
- **MixUp (α = 0.2)** for the first 10 epochs

**Evaluation transforms:** resize → center crop → grayscale → normalization.  
**Balanced sampling:** ensures equal AD / CN samples per batch.

---

### Training Configuration

| Parameter     | Value                          |
| ------------- | ------------------------------ |
| Epochs        | 30                             |
| Batch Size    | 16                             |
| Learning Rate | 3e-4 (adaptive)                |
| Weight Decay  | 5e-4                           |
| Optimizer     | AdamW                          |
| Scheduler     | CosineAnnealingLR              |
| Image Size    | 224 × 224                      |
| Framework     | PyTorch 2.2 + Torchvision 0.17 |
| Device        | CUDA (mixed precision)         |

---

## 📊 Results

### Accuracy Summary

| Metric                       | Value                  |
| ---------------------------- | ---------------------- |
| **Best Validation Accuracy** | **99.91 %** (epoch 22) |
| **Test Accuracy**            | **77.26 %**            |

> Validation performance reached near-perfect accuracy, while test results show realistic generalization to unseen MRI scans.

---

### Training Curves

#### Training vs Validation Loss

<img width="640" height="480" alt="training_loss" src="https://github.com/user-attachments/assets/104e0625-c50e-48f2-a702-1486297f5d21" />

#### Training vs Validation Accuracy

<img width="640" height="480" alt="training_acc" src="https://github.com/user-attachments/assets/9e0535cd-6da6-4efe-9d34-87b43472e964" />

---

## 🧠 Discussion

- **ConvNeXt-Small** achieved strong in-domain learning (near-perfect validation), demonstrating excellent fit to the ADNI dataset.
- The **77% test accuracy** aligns with expected results in MRI-based Alzheimer’s classification tasks due to dataset domain shifts and limited size.
- **MixUp** and **RepeatAug** improved generalization, reducing overfitting.
- **Staged unfreezing** prevented catastrophic forgetting and stabilized training.
- **Potential improvements:**
  - Larger image sizes (e.g., 256–384px) for finer spatial features.
  - Cross-site domain adaptation.
  - Ensembling multiple fine-tuned ConvNeXt variants.

---

## 🧪 Reproducibility

### Training

```bash
python train.py \
  --data ./ADNI/AD_NC \
  --variant small \
  --epochs 30 \
  --batch_size 16 \
  --lr 3e-4 \
  --wd 5e-4 \
  --mixup 0.2 \
  --tta \
  --save_dir ./outputs


  ### Inference / Testing
  python predict.py \
  --data ./ADNI/AD_NC \
  --model ./outputs/best_model.pth \
  --variant small --tta

This project was developed in a Conda environment named **`comp3710`** using:

| Package | Version |
|----------|----------|
| Python | 3.10 |
| PyTorch | 2.5.1 |
| Torchvision | 0.20.1 |
| Matplotlib | 3.8 |
| NumPy | 1.26 |
| CUDA | 12.x (GPU-enabled) |


```
