# dataset.py — data pipeline with MRI-safe augmentations

from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms

# --------------------- Constants ---------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# --------------------- Augmentations ---------------------
def _build_train_strong_transforms(image_size: int) -> transforms.Compose:
    """Strong but anatomically safe MRI augmentations."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
            )
        ], p=0.5),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.3)], p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.5),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * (1.0 + 0.05 * torch.randn(1).item())),
        transforms.Lambda(lambda t: t + 0.02 * torch.randn_like(t)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(
            p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'
        ),
    ])


def _build_train_cleanish_transforms(image_size: int) -> transforms.Compose:
    """Moderate view for stability and regularisation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_eval_transforms(image_size: int) -> transforms.Compose:
    """Deterministic evaluation transforms."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


if __name__ == "__main__":
    print("Augmentations ready.")
