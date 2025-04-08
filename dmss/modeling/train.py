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
    arch: str = "Unet"
    encoder_name: str = "resnet34"  # example model name
    in_channels: int = 3
    out_classes = 1
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = False  # Set to True if you want to shuffle the dataset during training
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_path: str = "path/to/data"  # Path to your dataset
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
    train_loader, val_loader, test_loader  = get_data_loaders(data_dir=conf.data_path,
                                                              batch_size=conf.batch_size,
                                                              num_workers=conf.num_workers,
                                                              )


if __name__ == "__main__":
    config = Config()
    main(config)
