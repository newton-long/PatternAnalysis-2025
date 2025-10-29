# dataset.py — adds RepeatAugDataset wrapper

from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------- augmentations ----------
def _build_train_strong_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------- RepeatAug wrapper ----------
class RepeatAugDataset(Dataset):
    """Repeats samples multiple times per epoch to apply different augmentations."""
    def __init__(self, base: datasets.ImageFolder, indices: List[int],
                 transform: transforms.Compose, repeat: int = 1):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.repeat = int(max(1, repeat))

    def __len__(self):
        return len(self.indices) * self.repeat

    def __getitem__(self, i: int):
        base_idx = self.indices[i % len(self.indices)]
        path, target = self.base.samples[base_idx]
        img = self.base.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target


if __name__ == "__main__":
    print("RepeatAugDataset added successfully.")
