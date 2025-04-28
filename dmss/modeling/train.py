from dataclasses import dataclass
import os
import random
import string
import sys

from clearml import Logger, Task
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
import torch
from torchvision.transforms import v2

from dmss.dataset import get_data_loaders
from dmss.models import PolypModel
from dmss.train_utils import SegmentationTrainer


# ----------------------------
# Define the hyperparameters
# ----------------------------
@dataclass
class Config:
    # ---------- General parameters------------
    project_dir: str = os.path.dirname(os.path.dirname(os.getcwd()))

    # ---------- Model parameters------------
    arch: str = "Unet"
    encoder_name: str = "resnet34"
    in_channels: int = 3
    out_classes: int = 1

    # ---------- Dataset parameters------------
    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 4
    data_path: str = os.path.join(
        project_dir, "data/external/data.csv"
    )  # Path to your annotations

    # ---------- Training parameters------------
    learning_rate: float = 1e-3
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    patience: int = 10  # Patience for early stopping

    # ---------- Loss parameters----------------
    alpha: float = 1.0
    beta: float = 1.0


def generate_random_string(length):
    # Объединяем все буквы (строчные и прописные) и цифры
    characters = string.ascii_letters + string.digits
    # Генерируем строку заданной длины
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


# Пример использования: генерируем строку длиной 10 символов
class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        :param alpha: вес для Dice Loss
        :param beta: вес для SoftBCEWithLogitsLoss
        :param mode: режим для DiceLoss
        :param smooth: параметр сглаживания для DiceLoss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(mode="binary")
        self.soft_bce_loss = SoftBCEWithLogitsLoss()

    def forward(self, preds, targets):
        # Вычисляем значения каждого лосса
        loss_dice = self.dice_loss(preds, targets)
        loss_bce = self.soft_bce_loss(preds, targets)
        # Возвращаем взвешенную сумму
        total_loss = self.alpha * loss_dice + self.beta * loss_bce
        return total_loss


def main(conf: Config, curr_task: Task):
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
    loss = CombinedLoss(alpha=conf.alpha, beta=conf.beta)
    # conf.alpha * DiceLoss(mode="binary") + conf.beta * SoftBCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Длина начального цикла (в шагах)
        T_mult=1,  # Увеличение длины цикла после каждого перезапуска
        eta_min=1e-7,  # Минимальный learning rate
    )

    curr_task.set_parameter("Optimizer", optimizer.__class__.__name__)
    curr_task.set_parameter("Scheduler", scheduler.__class__.__name__)

    # logger.report_text(
    #     f"Optimizer Name {optimizer.__class__.__name__}"
    # )  # TODO: дописать логирование clearml

    transforms = v2.Compose(
        [
            v2.Resize(size=(640, 640)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

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
    print(os.getcwd())
    task_name = generate_random_string(20)
    task = Task.init(
        project_name="dmss",
        task_name=task_name,
    )
    logger = task.get_logger()
    config = Config()
    task.connect(config)
    main(config, task)
    print("Training completed.")
    print(f"Task {task_name} has been closed.")
    task.close()
