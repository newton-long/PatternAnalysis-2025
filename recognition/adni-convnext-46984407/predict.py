#!/usr/bin/env python3
# predict.py — run inference on test set for ConvNeXt ADNI model

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from dataset import get_dataloaders
from modules import build_model

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

    # dataloaders (train/val/test)
    _, _, test_loader = get_dataloaders(
        args.data, batch_size=args.batch_size, image_size=args.image_size, val_split=args.val_split
    )

    # build model + load checkpoint
    model = build_model(num_classes=2, variant=args.variant, pretrained=True).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, total_acc = 0.0, 0.0

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
    print(f"\n✅ Test accuracy: {total_acc/n:.4f}")
    print(f"Average test loss: {total_loss/n:.4f}")

if __name__ == "__main__":
    main()
