#!/usr/bin/env python3
# predict.py — run inference on test set for ConvNeXt ADNI model + confusion matrix

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from modules import build_model


@torch.no_grad()
def plot_confusion_matrix(model, loader, device, out_dir, tta=False):
    """Generate and save confusion matrix for test predictions."""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if tta:
            logits = (model(images) + model(torch.flip(images, dims=[3]))) / 2
        else:
            logits = model(images)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NC", "AD"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    out_path = out_dir / "confusion_matrix.png"
    plt.savefig(out_path)
    plt.close()
    print(f"🧩 Confusion matrix saved to: {out_path}")

    # Also print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["NC", "AD"], digits=4))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset (e.g. ./ADNI/AD_NC)")
    ap.add_argument("--model", required=True, help="Path to saved .pth checkpoint")
    ap.add_argument("--variant", default="small", choices=["tiny","small","base","large"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--tta", action="store_true", help="Enable test-time augmentation (flip)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load dataloaders ---
    _, _, test_loader = get_dataloaders(
        args.data, batch_size=args.batch_size, image_size=args.image_size, val_split=args.val_split
    )

    # --- Build model + load weights ---
    model = build_model(num_classes=2, variant=args.variant, pretrained=True).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, total_acc = 0.0, 0.0

    # --- Evaluate test set ---
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if args.tta:
            logits = (model(imgs) + model(torch.flip(imgs, dims=[3]))) / 2
        else:
            logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc += (logits.argmax(1) == labels).float().mean().item()

    n = len(test_loader)
    print(f"\nTest accuracy: {total_acc/n:.4f}")
    print(f"Average test loss: {total_loss/n:.4f}")

    # --- Save confusion matrix ---
    out_dir = Path(args.model).parent
    plot_confusion_matrix(model, test_loader, device, out_dir, tta=args.tta)


if __name__ == "__main__":
    main()
