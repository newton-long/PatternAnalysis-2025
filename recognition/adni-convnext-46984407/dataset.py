# dataset.py — base skeleton for MRI-safe data pipeline

from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms

# --------------------- Constants ---------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def main():
    print("Dataset pipeline skeleton initialized.")


if __name__ == "__main__":
    main()
