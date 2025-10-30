# ADNI Alzheimer’s Classification with ConvNeXt-Small

This project fine-tunes a pretrained **ConvNeXt-Small** to classify **ADNI** brain MRI slices into:
- **AD** — Alzheimer’s Disease
- **NC** — Normal Control

The pipeline uses **staged fine-tuning**, **MixUp**, **label smoothing**, **AdamW + cosine LR**, **mixed precision**, **early stopping**, and **test-time augmentation (TTA)**. It produces clear visualisations (loss/accuracy curves + confusion matrix) and a classification report.

---

## 1) Problem & Approach (short theory + how it works)

We treat 2D MRI slices as inputs to a modern convolutional backbone (ConvNeXt-Small). Starting from ImageNet-pretrained weights:

1. **Staged fine-tuning** prevents catastrophic forgetting and stabilises transfer from natural images to medical MRIs:
   - **Stage 1 (epochs 1–3):** train only the **classifier head**.
   - **Stage 2 (epochs 4–12):** unfreeze the **last ConvNeXt block** + head.
   - **Stage 3 (epochs 13+):** unfreeze **all layers** and fine-tune end-to-end.

2. **Regularisation & optimisation**:
   - **MixUp (α=0.2)** early in training → smoother decision boundaries on small datasets; fades out later so the model “locks in”.
   - **Label smoothing (0.05)** in cross-entropy → reduces overconfident errors.
   - **Class weights** computed from the training set → handle class imbalance.
   - **AdamW** optimiser with **cosine learning-rate annealing**.
   - **AMP (autocast + GradScaler)** → faster training and lower VRAM.
   - **Early stopping** on validation accuracy (patience=8).

3. **Evaluation**:
   - **TTA** averages logits of original + horizontally flipped images.
   - We generate a **confusion matrix** and **classification report** on the test set.

**Pre-processing/transforms** (typical choices for medical classification):
- Resize to `image_size=224`.
- Center/resize crop, normalization to ImageNet stats.
- Light intensity/flip augments in training; no heavy spatial warps to preserve anatomy.

**Split justification**:
- We use `val_split=0.2` to monitor generalisation and enable early stopping without sacrificing too much training signal—appropriate for a limited dataset like ADNI.

**Algorithm Visualisation**:
<img width="1710" height="1144" alt="adni_convnext_flow" src="https://github.com/user-attachments/assets/86fbdbe4-4614-45d8-abdc-20403c172767" />
---

## 2) Reproducible Commands (exact)

### Train
```bash
python train.py \
  --data ./ADNI/AD_NC \
  --variant small \
  --epochs 50 \
  --batch_size 16 \
  --lr 3e-4 \
  --wd 5e-4 \
  --mixup 0.2 \
  --tta \
  --save_dir ./outputs_final_run
```
## This outputs training visuals:
```
### Training Accuracy
<img width="640" height="480" alt="training_acc" src="https://github.com/user-attachments/assets/3c03bdaa-bb43-4037-a0b4-2e0b1ade0e98" />

The training accuracy (blue) increases gradually over time as the network learns to fit the data.
Validation accuracy (orange) rises rapidly and then plateaus around 98–99%, showing that the model has converged.
The slight gap between training and validation curves indicates mild overfitting, which is expected on small datasets like ADNI but controlled effectively by MixUp and weight decay.
The late-epoch spike in training accuracy corresponds to the full-unfreezing stage, where all layers were fine-tuned, tightening the model’s fit.

### Training Validation
<img width="640" height="480" alt="training_loss" src="https://github.com/user-attachments/assets/21c65773-6775-4fc0-85fb-2de899c43f5d" />

Both training and validation loss decrease smoothly across epochs, confirming stable optimisation.
The validation loss stabilises earlier and remains low, indicating the model generalises well.
The consistent downward trend without divergence shows that the learning rate schedule and regularisation (MixUp, label smoothing, weight decay) effectively prevented overfitting and training instability.



## Test + Visualise
```
python predict.py \
  --data ./ADNI/AD_NC \
  --model ./outputs_final_run/best_model.pth \
  --variant small \
  --tta
```
## Which saves:
./outputs_final_run/confusion_matrix.png

<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/554e4e43-08db-46a3-9d06-60e8ef0e665c" />

**Core dependencies (exact stack used):**
| Package | Version | Purpose |
|----------|----------|----------|
| `python` | 3.10 | Core interpreter used in COMP3710 environment |
| `pytorch` | 2.5.1 | Deep learning framework for model training |
| `torchvision` | 0.20.1 | Provides ConvNeXt pretrained models and image transforms |
| `torchaudio` | 2.5.1 | Installed automatically with PyTorch (not directly used) |
| `numpy` | 1.26.x | Numerical computation and tensor manipulation |
| `matplotlib` | 3.8.x | Generates training loss/accuracy plots |
| `scikit-learn` | 1.5.x | Provides confusion matrix and classification report utilities |
| `tqdm` | 4.66.x | Progress bar during training and evaluation loops |
| `pandas` | 2.2.x | Lightweight dataset indexing and class weight computation |
| `pillow` (PIL) | 10.x | Image loading and resizing for ADNI dataset |
| `argparse` | builtin | Command-line interface for training/testing scripts |
| `pathlib` | builtin | Path handling for dataset and model files |



