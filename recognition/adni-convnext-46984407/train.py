#!/usr/bin/env python3
# train.py — ConvNeXt fine-tuning utilities and skeleton

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, Subset


# ---------- helpers ----------
def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def plot_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png"); plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "training_acc.png"); plt.close()


def count_class_weights_from_train_dataset(train_ds, device):
    """Compute per-class weights robustly across dataset/concat/subset wrappers."""
    def counts_from_base(base, indices):
        targets = getattr(base, "targets", None)
        if targets is None:
            targets = [y for _, y in base.samples]
        n_classes = len(base.classes)
        counts = [0] * n_classes
        for i in indices:
            counts[targets[i]] += 1
        return counts

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

    if isinstance(train_ds, Subset):
        base, idxs = train_ds.dataset, train_ds.indices
        counts = counts_from_base(base, idxs)
        total = sum(counts)
        weights = [total / (c if c > 0 else 1) for c in counts]
        return torch.tensor(weights, dtype=torch.float32, device=device)

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


# ---------- MixUp ----------
def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(crit, pred, y_a, y_b, lam):
    if y_b is None:
        return crit(pred, y_a)
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)


# ---------- freeze helpers ----------
def set_trainable(model, stage):
    """Freeze/unfreeze ConvNeXt backbone progressively."""
    for p in model.parameters():
        p.requires_grad = False

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", default="small", choices=["tiny", "small", "base", "large"])
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Dataset path: {args.data}")


if __name__ == "__main__":
    main()
