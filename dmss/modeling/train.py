from dataclasses import dataclass
import os

import torch
from torchvision.transforms import v2
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss

from dmss.dataset import get_data_loaders
from dmss.models import PolypModel
from dmss.train_utils import SegmentationTrainer



# ----------------------------
# Define the hyperparameters
# ----------------------------
@dataclass
class Config:
    # ---------- General parameters------------
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))

    # ---------- Model parameters------------
    arch = "Unet"
    encoder_name = "resnet34"
    in_channels = 3
    out_classes = 1

    # ---------- Dataset parameters------------
    epochs = 50
    batch_size = 32
    num_workers = 4
    data_path = os.path.join(project_dir, "data/external/data.csv")  # Path to your annotations

    #---------- Training parameters------------
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patience = 10  # Patience for early stopping

    #---------- Loss parameters----------------
    alpha = 1.0
    beta = 1.0


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
    # noinspection PyTypeChecker
    loss = conf.alpha*DiceLoss(mode='binary') + conf.beta*SoftBCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    transforms = v2.Compose([
        v2.Resize(size=(640, 640)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    train_loader, val_loader, test_loader = get_data_loaders(
        annotations_path=conf.data_path,
        transform=transforms,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        device=conf.device,
    )
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=conf.device,
        num_epochs=conf.epochs,
        patience=conf.patience,
    )
    trainer.train()



if __name__ == "__main__":
    config = Config()
    main(config)