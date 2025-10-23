# dataset.py
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(data_dir: str, batch_size: int = 32, image_size: int = 224, val_split: float = 0.2):
    """
    Loads the ADNI dataset and returns PyTorch dataloaders for train, val, and test.
    Automatically creates a validation split from the training set if one doesn’t exist.

    Expected folder structure:
        ADNI/
            train/
                AD/
                NC/
            test/
                AD/
                NC/
    """

    # Convert the input path to a Path object for easier manipulation
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # Data augmentation + normalization for training set
    # These transformations help the model generalize better
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # resize all images to same size
        transforms.RandomHorizontalFlip(),            # randomly flip images horizontally
        transforms.RandomRotation(10),                # small random rotations for variation
        transforms.ToTensor(),                        # convert to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5])   # normalize pixel values to [-1, 1]
    ])

    # For validation and test, we only normalize (no augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load the full training dataset
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

    # Split the training set into train and validation subsets
    # We use random_split for simplicity (80% train, 20% val by default)
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Validation shouldn’t have data augmentation applied
    # so we overwrite the transform to only use normalization
    val_dataset.dataset.transform = val_test_transforms

    # Load test set using the same normalization as validation
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    # Wrap all datasets with DataLoader for easy batch iteration during training
    # Shuffle only the training data to ensure random mini-batches every epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    # Allow command-line arguments for flexibility
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to ADNI dataset root (e.g. ./ADNI/AD_NC)")
    ap.add_argument("--batch_size", type=int, default=16, help="Number of samples per batch")
    args = ap.parse_args()

    # like this: python dataset.py --data ./ADNI/AD_NC --batch_size 16

    # Load the dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args.data, batch_size=args.batch_size)

    # Print some quick info to verify the setup
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
