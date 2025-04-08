from dataclasses import dataclass

import torch
from torch.optim import Adam

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
    data_path = {"train": (path,)}  # Path to your dataset
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

    model.to(config.device)

    # Add code to initialize the optimizer and data loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=conf.data_path,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
    )


if __name__ == "__main__":
    config = Config()
    main(config)
