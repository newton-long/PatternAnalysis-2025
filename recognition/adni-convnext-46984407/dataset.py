# dataset.py
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(data_dir: str, batch_size: int = 32, image_size: int = 224, val_split: float = 0.2):
    """
    Loads the ADNI dataset and returns PyTorch dataloaders for train, val, and test.
    - Uses ImageNet normalization to match ConvNeXt pretrained weights
    - Adds stronger train-time augments for better generalization
    - Converts grayscale MRI to 3 channels (needed by ConvNeXt)
    - Splits train -> (train/val) if no explicit val exists

    Expected folder structure (yours is ADNI/AD_NC/...; we auto-handle that):
        root/
          train/AD, train/NC
          test/AD,  test/NC
    """

    # Resolve paths and auto-handle nested AD_NC folder
    data_dir = Path(data_dir)
    if (data_dir / "AD_NC").exists():
        data_dir = data_dir / "AD_NC"

    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    # ImageNet stats to align with ConvNeXt pretrained weights
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # Stronger train-time pipeline (random crop/scale + flip) + force 3 channels
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),   # MRI is 1ch; ConvNeXt expects 3ch
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Evaluation pipeline: resize -> center crop (standard ImageNet eval) + 3ch + normalize
    val_test_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),     # e.g., 256 when image_size=224
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Load the full training dataset
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

    # Split train -> (train/val)
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Validation should not use data augmentation
    val_dataset.dataset.transform = val_test_transforms

    # Test dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    # DataLoaders
    # pin_memory helps on GPU; num_workers=2 is safe default for most setups
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to ADNI dataset root (e.g. ./ADNI or ./ADNI/AD_NC)")
    ap.add_argument("--batch_size", type=int, default=16, help="Number of samples per batch")
    ap.add_argument("--image_size", type=int, default=224, help="Input image size (match model)")
    ap.add_argument("--val_split", type=float, default=0.2, help="Fraction of train set to use for validation")
    args = ap.parse_args()

    # Example: python dataset.py --data ./ADNI/AD_NC --batch_size 16
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data, batch_size=args.batch_size, image_size=args.image_size, val_split=args.val_split
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
