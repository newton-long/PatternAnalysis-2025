# dataset.py
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(data_dir: str, batch_size: int = 32, image_size: int = 224, val_split: float = 0.2):
    """
    Loads ADNI and returns train/val/test loaders.
    - ImageNet normalization (matches ConvNeXt pretraining)
    - Stronger train augments (crop, flip, blur, random erasing)
    - Grayscale -> 3 channels
    - Splits train -> (train/val) if no explicit val exists
    """
    data_dir = Path(data_dir)
    if (data_dir / "AD_NC").exists():
        data_dir = data_dir / "AD_NC"

    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # Stronger train-time pipeline (wider crop range + blur + erasing)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), inplace=True),
    ])

    # Standard eval pipeline (resize->center crop) + 3ch + normalize
    val_test_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),   # e.g., 256 for 224
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Remove augments for val
    val_dataset.dataset.transform = val_test_transforms

    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to ADNI dataset root (e.g. ./ADNI or ./ADNI/AD_NC)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.2)
    args = ap.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data, batch_size=args.batch_size, image_size=args.image_size, val_split=args.val_split
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
