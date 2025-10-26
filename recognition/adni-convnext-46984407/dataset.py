# dataset.py — upgraded augmentation & dataloaders for ConvNeXt ADNI fine-tuning
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(data_dir: str, batch_size: int = 32, image_size: int = 224, val_split: float = 0.2):
    """
    Loads ADNI dataset and returns train/val/test dataloaders.

    Improvements:
    - Enhanced augmentation (rotation, color jitter, blur, random erase)
    - Handles grayscale -> RGB for ConvNeXt
    - ImageNet normalization for pretrained backbone
    - RandomResizedCrop scale (0.8–1.0) for better context balance
    - Safe random split into train/val if not predefined
    """
    data_dir = Path(data_dir)
    if (data_dir / "AD_NC").exists():
        data_dir = data_dir / "AD_NC"

    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # ---------- Train transforms (strong but stable) ----------
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        transforms.RandomApply([transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random')], p=0.5),
        transforms.Grayscale(num_output_channels=3),  # ensure 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # ---------- Validation / Test transforms (deterministic) ----------
    val_test_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),   # e.g., 256 for 224
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # ---------- Dataset splits ----------
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_test_transforms  # disable augments for val

    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    # ---------- Dataloaders ----------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader


# ---------- Quick check ----------
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
