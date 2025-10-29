#!/usr/bin/env python3
# train.py — staged fine-tuning for ConvNeXt on ADNI (head → last → all)
# ---------------------------------------------------------------
# This script fine-tunes a pretrained ConvNeXt model on the ADNI dataset.
# It uses staged unfreezing (classifier head → last block → full model),
# mixed-precision training, mixup augmentation, and early stopping.

import argparse, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevent GUI issues when saving plots remotely
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler  # For mixed precision
from torch.utils.data import ConcatDataset, Subset

from dataset import get_dataloaders  # Custom dataset loader
from modules import build_model       # Model constructor (ConvNeXt)


# ---------- helper functions ----------
def accuracy(logits, targets):
    """Compute batch accuracy."""
    preds = logits.argmax(dim=1)           # Get predicted class index
    return (preds == targets).float().mean().item()


def current_lr(optimizer):
    """Return the current learning rate (for logging)."""
    return optimizer.param_groups[0]["lr"]


def plot_curves(history, out_dir):
    """Plot training and validation curves for loss and accuracy."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss curves ---
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png")
    plt.close()

    # --- Accuracy curves ---
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "training_acc.png")
    plt.close()


def count_class_weights_from_train_dataset(train_ds, device):
    """
    Compute class weights to handle class imbalance.
    Works for datasets wrapped in ConcatDataset or Subset.
    Returns a GPU tensor of per-class weights.
    """

    def counts_from_base(base, indices):
        """Helper to count samples per class given a dataset + indices."""
        targets = getattr(base, "targets", None)
        if targets is None:
            targets = [y for _, y in base.samples]  # fallback
        n_classes = len(base.classes)
        counts = [0] * n_classes
        for i in indices:
            counts[targets[i]] += 1
        return counts

    # Case 1: dataset is a ConcatDataset (multiple subsets combined)
    if isinstance(train_ds, ConcatDataset):
        base_to_idxset = {}
        for sub in train_ds.datasets:
            base = getattr(sub, "base", None)
            idxs = getattr(sub, "indices", None)
            if base is None or idxs is None:
                continue
            if base not in base_to_idxset:
                base_to_idxset[base] = set()
            base_to_idxset[base].update(idxs)

        total_counts = None
        for base, idxset in base_to_idxset.items():
            c = counts_from_base(base, list(idxset))
            total_counts = c if total_counts is None else [a + b for a, b in zip(total_counts, c)]

        total = sum(total_counts)
        weights = [total / (c if c > 0 else 1) for c in total_counts]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    # Case 2: dataset is a Subset
    if isinstance(train_ds, Subset):
        base, idxs = train_ds.dataset, train_ds.indices
        counts = counts_from_base(base, idxs)
        total = sum(counts)
        weights = [total / (c if c > 0 else 1) for c in counts]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    # Case 3: plain dataset
    targets = getattr(train_ds, "targets", None)
    if targets is None:
        targets = [y for _, y in train_ds.samples]
    n_classes = len(train_ds.classes)
    counts = [0] * n_classes
    for y in targets:
        counts[y] += 1
    total = sum(counts)
    weights = [total / (c if c > 0 else 1) for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------- MixUp augmentation ----------
def mixup_data(x, y, alpha=0.2):
    """
    Perform MixUp augmentation.
    Randomly blends pairs of images and labels by a factor λ ~ Beta(α, α).
    """
    if alpha <= 0:
        return x, y, None, 1.0  # Disabled
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)  # Random shuffle
    mixed_x = lam * x + (1 - lam) * x[idx]           # Blend inputs
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(crit, pred, y_a, y_b, lam):
    """Compute the MixUp loss using blended targets."""
    if y_b is None:
        return crit(pred, y_a)
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)


# ---------- freezing helper ----------
def set_trainable(model, stage):
    """
    Freeze or unfreeze parts of ConvNeXt during staged fine-tuning.
    Stages:
      - 'head': only classifier
      - 'last': last feature block + classifier
      - 'all': full model
    """
    for p in model.parameters():
        p.requires_grad = False  # Freeze everything by default

    if stage == "head":
        for p in model.backbone.classifier.parameters():
            p.requires_grad = True

    elif stage == "last":
        for p in model.backbone.features[-1].parameters():
            p.requires_grad = True
        for p in model.backbone.classifier.parameters():
            p.requires_grad = True

    elif stage == "all":
        for p in model.parameters():
            p.requires_grad = True


# ---------- core training & evaluation ----------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, mixup_alpha=0.0):
    """
    Train the model for one epoch using mixed precision and MixUp.
    """
    model.train()
    loss_sum, acc_sum = 0.0, 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Apply MixUp augmentation
        images, ya, yb, lam = mixup_data(images, labels, alpha=mixup_alpha)

        # Forward + backward pass under mixed precision
        with autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = mixup_criterion(criterion, logits, ya, yb, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        acc_sum += accuracy(logits, labels)

    n = len(loader)
    return loss_sum / n, acc_sum / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta=False):
    """
    Evaluate model performance (optional test-time augmentation).
    Returns average loss and accuracy.
    """
    model.eval()
    loss_sum, acc_sum = 0.0, 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # Average predictions with horizontally flipped images if TTA is enabled
        if tta:
            logits = (model(images) + model(torch.flip(images, dims=[3]))) / 2
        else:
            logits = model(images)

        loss = criterion(logits, labels)
        loss_sum += loss.item()
        acc_sum += accuracy(logits, labels)

    n = len(loader)
    return loss_sum / n, acc_sum / n


# ---------- main training pipeline ----------
def main():
    # --- Argument parsing ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="small", choices=["tiny","small","base","large"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--save_dir", default="./outputs")
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data, batch_size=args.batch_size, image_size=args.image_size, val_split=args.val_split
    )

    # --- Build model ---
    model = build_model(num_classes=2, variant=args.variant, pretrained=True).to(device)

    # Compute class weights to handle imbalance
    class_weights = count_class_weights_from_train_dataset(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # Mixed precision gradient scaler
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    # Track best validation accuracy and early stopping counter
    best_val_acc, best_epoch, bad_epochs = 0.0, 0, 0
    history = {k: [] for k in ["train_loss","train_acc","val_loss","val_acc","lr"]}

    start = time.time()

    # --- Epoch loop ---
    for epoch in range(1, args.epochs+1):
        # Stage 1: fine-tune only the head
        if epoch <= 3:
            phase = "head"
            set_trainable(model, "head")
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-3, weight_decay=args.wd
            )

        # Stage 2: unfreeze last block + classifier
        elif 4 <= epoch <= 12:
            phase = "last"
            set_trainable(model, "last")
            optimizer = optim.AdamW([
                {"params": model.backbone.features[-1].parameters(), "lr": 7.5e-5},
                {"params": model.backbone.classifier.parameters(), "lr": 3e-4}
            ], weight_decay=args.wd)

        # Stage 3: full fine-tuning
        else:
            phase = "all"
            set_trainable(model, "all")
            optimizer = optim.AdamW([
                {"params": model.backbone.features[:-1].parameters(), "lr": 3e-5},
                {"params": model.backbone.features[-1].parameters(), "lr": 6e-5},
                {"params": model.backbone.classifier.parameters(), "lr": 1e-4}
            ], weight_decay=args.wd)

        # --- Learning rate scheduler ---
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs-epoch+1), eta_min=1e-5)

        # MixUp gradually fades after 10 epochs
        mixup_alpha = args.mixup if epoch <= 10 else max(0.0, args.mixup * (1 - (epoch - 10)/max(1, args.epochs - 10)))

        # --- Train + Validate ---
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, mixup_alpha)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, tta=False)
        scheduler.step()

        # --- Record metrics ---
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
        history["lr"].append(current_lr(optimizer))

        print(f"Epoch {epoch:02d}/{args.epochs} [{phase}] | "
              f"lr={current_lr(optimizer):.2e} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}")

        # --- Early stopping logic ---
        if va_acc > best_val_acc:
            best_val_acc, best_epoch, bad_epochs = va_acc, epoch, 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")  # Save best model
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # --- Testing phase ---
    state = torch.load(save_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device, tta=args.tta)

    print(f"Test | loss={te_loss:.4f} acc={te_acc:.4f}")
    print(f"Total time: {(time.time() - start) / 60:.1f} min")

    # --- Save training summary ---
    plot_curves(history, save_dir)
    with open(save_dir / "summary.txt", "w") as f:
        f.write(f"Best Val Acc: {best_val_acc:.4f} (epoch {best_epoch})\n")
        f.write(f"Test Acc: {te_acc:.4f}\n")


if __name__ == "__main__":
    main()
