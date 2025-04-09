import os
from typing import Tuple

import cv2 as cv
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class PolypDataset(Dataset):
    def __init__(
        self,
        annotations_file: str = None,
        transform=None,
        mode: str = "train",  # 'train', 'valid', 'test'
        device=None,  # 'cuda' or 'cpu'
    ):
        self.annotations_file = pd.read_csv(annotations_file)
        self.mode = mode  # 'train', 'valid', 'test'
        self.device = device  # 'cuda' or 'cpu'
        self.transform = transform.to(self.device) if transform else None

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        return self._get_item_from_csv(index)

    def _get_item_from_csv(self, index):
        img_path = self.annotations_file.iloc[index, 0]
        mask_path = self.annotations_file.iloc[index, 1]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"The image file {img_path} does not exist.")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"The mask file {mask_path} does not exist.")

        # Load image and mask from paths
        image = cv.imread(img_path).to(self.device)
        mask = cv.imread(mask_path, 0)  # Load mask in grayscale mode

        # Apply transformations if any
        if self.transform:
            image = self.transform(image).to(self.device)  # Ensure image is on the correct device
            mask = self.transform(mask).to(self.device)  # Ensure mask is on the correct device

        return image, mask


def get_data_loaders(
    annotations_path: str = None,
    transform=None,  # transforms.Compose([transforms.ToTensor()]) or None
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = 'cuda'
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    """Returns a dataloader for the given dataset."""
    train_dataset = PolypDataset(annotations_path, transform=transform, device=device)
    valid_dataset = PolypDataset(annotations_path, transform=transform, device=device)
    test_dataset = PolypDataset(annotations_path, transform=transform, device=device)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_data_loader, valid_data_loader, test_data_loader


if __name__ == "__main__":
    # Example usage
    annotations_file_path = "/Users/macbook/Desktop/MagaDiplom/DMSS/data/external/data.csv"

    transforms = v2.Compose([
        v2.Resize(640, 640),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # data = PolypDataset(annotations_file=annotations_file_path, device="cpu")
    train_loader, val_loader, test_loader = get_data_loaders(
        annotations_path=annotations_file_path,
        transform=transforms,
        batch_size=16,
        num_workers=2,
        device="cpu",
    )


