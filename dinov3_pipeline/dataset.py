"""
dino_pipeline/dataset.py

Image dataset preparation for DINO vision transformer model.

This module:
- Loads physics-informed crystal structure images
- Implements PyTorch Dataset for image data
- Applies data augmentation for training
- Handles preprocessing (normalization, resizing)
- Creates DataLoaders for training and evaluation

Image specifications:
- Size: 224x224 (standard for vision transformers)
- Format: RGB
- Normalization: ImageNet mean/std (for pre-trained models)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SuperconductorImageDataset(Dataset):
    """
    PyTorch Dataset for superconductor crystal structure images.

    Each sample consists of:
    - A rendered 2D image of the crystal structure
    - The critical temperature (Tc) as the target
    """

    def __init__(self, metadata_df, transform=None, split="train"):
        """
        Initialize the dataset.

        Args:
            metadata_df (pd.DataFrame): DataFrame with image paths and labels
            transform: Optional torchvision transforms
            split (str): Data split ('train', 'val', or 'test')
        """
        self.metadata = metadata_df[metadata_df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.split = split

        if len(self.metadata) == 0:
            raise ValueError(f"No samples found for split: {split}")

    def __len__(self):
        """Returns the number of samples."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Gets a single sample.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (image, tc_value)
        """
        row = self.metadata.iloc[idx]
        image_path = row["image_path"]
        tc_value = row["tc"]

        # Fix relative path - prepend parent directory if path starts with "data/"
        if image_path.startswith("data/"):
            image_path = os.path.join("..", image_path)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(tc_value, dtype=torch.float32)


def get_transforms(split="train", image_size=224):
    """
    Gets appropriate transforms for each data split.

    Training transforms include augmentation:
    - Random horizontal/vertical flips
    - Random rotation
    - Color jittering
    - Normalization using ImageNet statistics

    Validation/test transforms only include:
    - Resizing
    - Normalization

    Args:
        split (str): Data split ('train', 'val', or 'test')
        image_size (int): Target image size

    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    # ImageNet normalization (for pre-trained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == "train":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])

    return transform


def get_dataloader(
    metadata_path,
    split="train",
    batch_size=32,
    shuffle=None,
    num_workers=4,
    image_size=224
):
    """
    Creates a DataLoader for the specified split.

    Args:
        metadata_path (str): Path to image metadata CSV
        split (str): Data split ('train', 'val', or 'test')
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data (default: True for train, False otherwise)
        num_workers (int): Number of worker processes
        image_size (int): Image size

    Returns:
        DataLoader: PyTorch DataLoader
    """
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)

    # Get transforms
    transform = get_transforms(split, image_size)

    # Create dataset
    dataset = SuperconductorImageDataset(
        metadata_df=metadata_df,
        transform=transform,
        split=split
    )

    # Default shuffle behavior
    if shuffle is None:
        shuffle = (split == "train")

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return loader


def load_metadata(metadata_path="data/images/image_metadata.csv"):
    """
    Loads image metadata from CSV.

    Args:
        metadata_path (str): Path to metadata file

    Returns:
        pd.DataFrame: Metadata with image paths and labels
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Please run 03_render_images.py first to generate images."
        )

    df = pd.read_csv(metadata_path)

    print(f"Loaded metadata for {len(df)} images")
    print(f"  Train: {len(df[df['split'] == 'train'])}")
    print(f"  Val:   {len(df[df['split'] == 'val'])}")
    print(f"  Test:  {len(df[df['split'] == 'test'])}")

    return df


if __name__ == "__main__":
    # Test dataset loading
    print("Testing image dataset...")

    try:
        # Load metadata
        metadata = load_metadata()

        # Create dataloaders
        train_loader = get_dataloader(
            "data/images/image_metadata.csv",
            split="train",
            batch_size=8,
            num_workers=0
        )

        print(f"\nTrain loader created with {len(train_loader)} batches")

        # Test loading a batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")

        print("\nImage dataset module ready!")

    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Dataset will be available after running 03_render_images.py")
