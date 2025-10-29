#!/usr/bin/env python3
# train.py — skeleton for ConvNeXt fine-tuning

import argparse
from pathlib import Path
import torch


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

    print(f"Device: {device}")
    print(f"Data directory: {args.data}")
    print("Argument parser initialized successfully.")


if __name__ == "__main__":
    main()
