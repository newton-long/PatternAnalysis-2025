#!/usr/bin/env python3
# dataset.py — ultra-optimised, MRI-safe data pipeline with balanced sampling and RepeatAug
# ------------------------------------------------------------------------------
# Provides train/val/test dataloaders for MRI image classification.
# Includes:
#  - MRI-safe augmentations (no geometric distortion of anatomy)
#  - RepeatAug to simulate multiple random augmentations per image
#  - Balanced sampling (WeightedRandomSampler)
#  - Deterministic splits for reproducibility
# ------------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms


# --------------------- Constants ---------------------
# Standard normalization values from ImageNet (same as pretrained ConvNeXt backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# --------------------- Augmentations ---------------------
def _build_train_strong_transforms(image_size: int) -> transforms.Compose:
    """
    Build strong yet anatomically 'safe' augmentations for MRI images.
    These transformations increase diversity without unrealistic deformation.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        # Small affine and perspective transformations to simulate scanner noise
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
            )
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomPerspective(distortion_scale=0.3)
        ], p=0.5),
        # Mild brightness and contrast variations
        transforms.ColorJitter(brightness=0.15, contrast=0.3),
        # Gaussian blur simulates out-of-focus scans
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
        ], p=0.5),
        transforms.Grayscale(num_output_channels=3),     # ensure model expects 3-channel input
        transforms.ToTensor(),
        # Add light intensity noise to simulate MRI machine variations
        transforms.Lambda(lambda t: t * (1.0 + 0.05 * torch.randn(1).item())),
        transforms.Lambda(lambda t: t + 0.02 * torch.randn_like(t)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Random erasing mimics local occlusion / missing slices
        transforms.RandomErasing(
            p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'
        ),
    ])


def _build_train_cleanish_transforms(image_size: int) -> transforms.Compose:
    """
    Build moderate augmentations for cleaner, stable training views.
    These are less aggressive than the strong transforms.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_eval_transforms(image_size: int) -> transforms.Compose:
    """
    Deterministic evaluation pipeline for validation and test.
    Ensures consistent preprocessing (no randomness).
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# --------------------- RepeatAug Wrapper ---------------------
class RepeatAugDataset(Dataset):
    """
    Wraps an ImageFolder dataset to repeat each image multiple times per epoch,
    allowing multiple random augmentations of the same MRI scan.
    This helps the model generalize better.
    """
    def __init__(self, base: datasets.ImageFolder, indices: List[int],
                 transform: transforms.Compose, repeat: int = 1):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
        self.repeat = int(max(1, repeat))  # minimum repeat = 1

    def __len__(self):
        return len(self.indices) * self.repeat

    def __getitem__(self, i: int):
        base_idx = self.indices[i % len(self.indices)]
        path, target = self.base.samples[base_idx]
        img = self.base.loader(path)  # load the MRI scan
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# --------------------- Utility: deterministic split ---------------------
def _deterministic_split_indices(n: int, val_split: float, seed: int = 1337):
    """
    Split dataset indices into train and validation sets deterministically.
    Uses a fixed random seed for reproducibility across runs.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_size = int(n * val_split)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return train_idx, val_idx


# --------------------- Public API ---------------------
def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    val_split: float = 0.2,
    seed: int = 1337,
    num_workers: int = 2,
):
    """
    Main dataloader function for ADNI MRI classification.
    Returns train, validation, and test dataloaders.
    Features:
        - Deterministic split
        - Balanced WeightedRandomSampler
        - RepeatAug for data diversity
        - MRI-safe augmentations
    """
    root = Path(data_dir)
    # Support both `ADNI` and `ADNI/AD_NC` directory formats
    if (root / "AD_NC").exists():
        root = root / "AD_NC"
    train_dir = root / "train"
    test_dir  = root / "test"

    # Load the training set once (without transform yet)
    base_train = datasets.ImageFolder(train_dir, transform=None)
    n_classes = len(base_train.classes)
    print(f"[INFO] Found {len(base_train)} training images across {n_classes} classes.")

    # Deterministic train/val split
    train_idx, val_idx = _deterministic_split_indices(
        len(base_train), val_split=val_split, seed=seed
    )

    # Build augmentation pipelines
    t_strong = _build_train_strong_transforms(image_size)
    t_clean  = _build_train_cleanish_transforms(image_size)
    t_eval   = _build_eval_transforms(image_size)

    # Combine strong and clean datasets for training diversity
    ds_train_strong = RepeatAugDataset(base_train, train_idx, transform=t_strong, repeat=3)
    ds_train_clean  = RepeatAugDataset(base_train, train_idx, transform=t_clean, repeat=1)
    train_combined: Dataset = ConcatDataset([ds_train_strong, ds_train_clean])

    # Validation and test subsets (evaluation transforms)
    val_subset  = Subset(datasets.ImageFolder(train_dir, transform=t_eval), val_idx)
    test_dataset = datasets.ImageFolder(test_dir, transform=t_eval)

    # --------------------- Balanced Sampling ---------------------
    # Compute per-class sample weights to ensure balanced sampling (e.g. AD vs NC)
    targets = torch.tensor([base_train.samples[i][1] for i in train_idx])
    class_sample_count = torch.tensor([(targets == t).sum() for t in torch.unique(targets)])
    weight_per_class = 1.0 / class_sample_count.float()
    weights = torch.tensor([weight_per_class[t] for t in targets])
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # --------------------- Dataloaders ---------------------
    train_loader = DataLoader(
        train_combined,
        batch_size=batch_size,
        sampler=sampler,                  # balance classes
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    print(f"[INFO] Train: {len(train_combined)} imgs × repeat, Val: {len(val_subset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


# --------------------- CLI Check ---------------------
if __name__ == "__main__":
    # Command-line entry for quick verification
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to ADNI root (e.g. ./ADNI or ./ADNI/AD_NC)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    tl, vl, te = get_dataloaders(
        args.data,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    print(f"[Check] Train batches: {len(tl)} | Val batches: {len(vl)} | Test batches: {len(te)}")
