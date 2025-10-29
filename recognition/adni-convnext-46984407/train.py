#!/usr/bin/env python3
# train.py — ConvNeXt fine-tuning utilities and skeleton

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


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
