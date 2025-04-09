from dataclasses import dataclass

import torch
from torch.optim import Adam
import albumentations as A
from torchvision.transforms import v2

from dmss.dataset import PolypDataset, get_data_loaders
from dmss.models import PolypModel


# ----------------------------
# Define the hyperparameters
# ----------------------------
@dataclass
class Config:
    arch = "Unet"
    encoder_name = "resnet34"  # example model name
    in_channels = 3
    out_classes = 1
    epochs = 50
    batch_size = 32
    num_workers = 4
    shuffle = False  # Set to True if you want to shuffle the dataset during training
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "/Users/macbook/Desktop/MagaDiplom/DMSS/data/external/data.csv"  # Path to your annotations
    # Add other configuration parameters as needed


# ----------------------------
# Train model
# ----------------------------
def main(conf: Config):
    # Initialize model, optimizer, and data loaders
    model = PolypModel(
        arch=conf.arch,
        encoder_name=conf.encoder_name,
        in_channels=conf.in_channels,
        out_classes=conf.out_classes,
        device=conf.device,
    )

    model.to(conf.device)

    # Add code to initialize the optimizer and data loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate)
    transforms = v2.Compose([
        v2.Resize(640, 640),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader, val_loader, test_loader = get_data_loaders(
        annotations_path=conf.data_path,
        transform=transforms,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        device=conf.device,
    )


if __name__ == "__main__":
    config = Config()
    main(config)
