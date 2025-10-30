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

## This saves:
```
./outputs_final_run/
  best_model.pth
  training_loss.png
  training_acc.png
  summary.txt
  ```
## Test + Visualise
```
python predict.py \
  --data ./ADNI/AD_NC \
  --model ./outputs_final_run/best_model.pth \
  --variant small \
  --tta
```
Which saves:
./outputs_final_run/confusion_matrix.png
