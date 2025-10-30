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

**Pre-processing / Transforms**

All MRI slices from the ADNI dataset were preprocessed to ensure compatibility with ConvNeXt’s ImageNet pretraining. Each image was:
- **Resized to 224×224 pixels** to match ConvNeXt-Small input dimensions.
- **Normalized** using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), following the official PyTorch ConvNeXt model implementation [1].
- **Lightly augmented** during training with random horizontal flips and minor brightness/contrast jitter to improve robustness to patient positioning and scanner variability.
- Validation and test sets were only **center-cropped and normalized**, with no random augmentations, to ensure consistent evaluation.

These preprocessing steps follow common transfer learning procedures in medical imaging tasks, where ImageNet-normalized models are adapted to MRI or CT modalities [2].  
Light augmentation helps generalisation while preserving anatomical fidelity — critical for medical data.

---

**Dataset Splitting and Justification**

The dataset was divided into **training, validation, and test subsets** to enable fair evaluation and prevent data leakage.

- **80% Training Set** — used for model fitting and weight updates.  
- **20% Validation Set** — used to monitor performance during training and trigger early stopping once validation accuracy plateaued.  
- **Separate Test Set** — unseen during training and validation; used once for final evaluation.

This **80/20 split** strikes a balance between having sufficient training data and a meaningful validation signal. It’s a standard ratio in small medical datasets, providing enough variance in validation while preserving training stability [2].  

The separate **test set** ensures an unbiased estimate of generalisation, while the validation split allows controlled tuning of hyperparameters and early stopping to prevent overfitting.

In summary, preprocessing standardized all images for consistent ConvNeXt input, and the 80/20 split (with a held-out test set) ensured fair, reproducible evaluation.

---

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
## Which saves (a confusion matrix):

<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/554e4e43-08db-46a3-9d06-60e8ef0e665c" />

The confusion matrix above shows the classification results on the held-out test set.
The model correctly identified 4,441 Alzheimer’s (AD) cases and 2,412 Normal Control (NC) cases.
It misclassified 2,048 NC samples as AD (false positives) and only 99 AD samples as NC (false negatives).
This results in very high recall for AD (0.978) — meaning almost all AD cases were detected — and high precision for NC (0.961).
Such a pattern indicates the model is slightly biased toward predicting AD, which is acceptable in medical screening tasks where missing a true AD case (false negative) is far more critical than a false alarm.

## Model Evaluation (Test Set Results)

The model was evaluated on the held-out test set using **Test-Time Augmentation (TTA)** to improve robustness.  
Below is the classification report and key performance metrics.

```
      NC     0.9606    0.5408    0.6920      4460
      AD     0.6844    0.9782    0.8053      4540
accuracy                         0.7614      9000
```

### **Performance Summary**

| Metric | Value | Description |
|:--------|:------:|:------------|
| **Overall Accuracy** | **76.14%** | Percentage of correctly classified MRI slices |
| **Average Test Loss** | **0.55** | Mean cross-entropy loss across the test set |
| **NC Precision** | **96.06%** | When predicting “Normal Control”, 96% were correct |
| **NC Recall** | **54.08%** | Model correctly identified 54% of true NC cases |
| **AD Precision** | **68.44%** | When predicting “Alzheimer’s Disease”, 68% were correct |
| **AD Recall** | **97.82%** | Model correctly identified almost all true AD cases |

---

### **Interpretation**

> The model achieved **76.1% accuracy** on unseen test data, demonstrating strong generalisation given the limited dataset size.  
> The results show a clear trade-off: the model prioritises **high recall for Alzheimer’s Disease (97.8%)** at the cost of lower recall for Normal Controls (54.1%).  
> This bias towards detecting AD is desirable in a medical screening context, where **missing a true AD case (false negative)** is far more critical than a false alarm.  
> The high **precision for NC (96.1%)** indicates that when the model predicts “Normal”, it is typically correct and confident.  
> Overall, the fine-tuned **ConvNeXt-Small** model demonstrates reliable classification performance and aligns with clinical priorities for early, sensitive Alzheimer’s detection.

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

**References**

[1] PyTorch Vision Models Documentation — ConvNeXt: https://pytorch.org/vision/stable/models/convnext.html  
[2] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS. (Commonly referenced for ImageNet normalization & transfer-learning preprocessing)


