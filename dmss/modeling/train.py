from dataclasses import dataclass

import torch
from torch.optim import Adam

from dmss.models import PolypModel
from dmss.dataset import get_data_loaders, PolypDataset


# ----------------------------
# Define the hyperparameters
# ----------------------------
@dataclass
class Config:
    arch: str = "Unet"
    encoder_name: str = "resnet34"  # example model name
    in_channels: int = 3
    out_classes = 1
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Add other configuration parameters as needed
    pass


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
    train_loader, val_loader = get_data_loaders()


if __name__ == "__main__":
    config = Config()
    main(config)
