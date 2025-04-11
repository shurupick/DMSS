import os

import segmentation_models_pytorch
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping class that stops training when a specified number of epochs have passed without improvement.

    Attributes:
        best_fitness (float): Best fitness value observed.
        best_epoch (int): Epoch where best fitness was observed.
        patience (int): Number of epochs to wait after fitness stops improving before stopping.
        possible_stop (bool): Flag indicating if stopping may occur next epoch.
    """

    def __init__(self, patience):
        """
        Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float(
            "inf"
        )  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:  # check if fitness=None (happens when val=False)
            return False

        if (
            fitness > self.best_fitness or self.best_fitness == 0
        ):  # allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded

        return stop


class SegmentationTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: segmentation_models_pytorch.losses,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        device: str = None,
        num_epochs: int = 10,
        patience: int = 50,
    ):
        """
        :param model: модель сегментации
        :param train_loader: DataLoader для обучающего набора
        :param val_loader: DataLoader для валидационного набора
        :param loss_fn: функция потерь (например, CrossEntropyLoss)
        :param optimizer: оптимизатор
        :param device: устройство для тренировки ('cuda' или 'cpu')
        :param num_epochs: количество эпох обучения
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.patience = patience
        self.stopper, self.stop = EarlyStopping(patience=self.patience), False
        self.checkpoint_dir = "./models"

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            # Переносим данные на выбранное устройство
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            # Получаем предсказания модели. Выход имеет размер [B, num_classes, H, W]
            outputs = self.model(images)
            # Для CrossEntropyLoss ожидаются выходы с размером [B, num_classes, H, W] и маски [B, H, W]
            loss = self.loss_fn(outputs, masks)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch: {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                running_loss += loss.item()

        val_loss = running_loss / len(self.val_loader)
        return val_loss

    def train(self):
        self.best_val_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs}:")

            # ------- Train -------
            train_loss = self.train_epoch()

            # ------- Validate -------
            final_epoch = epoch + 1 >= self.num_epochs
            val_loss = self.validate()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model("best")
                print(f"Best validation loss updated to {val_loss:.4f}")

            self.stop |= self.stopper(epoch + 1, val_loss)

            # ------- Save model -------
            if final_epoch:
                self._save_model("last")
                print(f"Final epoch reached. Model saved.")

            # ------- Early stopping -------
            if self.stop:
                print("Early stopping triggered.")
                break

            print(
                f"Epoch {epoch}  Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}\n"
            )

    def _save_model(self, mode: str):
        """
        mode: 'best' or 'last'
        """
        if mode == "best":
            torch.save(
                self.model.state_dict(), os.path.join(self.checkpoint_dir, "best/best_model.pth")
            )
            print(f"Best model saved.")
        elif mode == "last":
            torch.save(
                self.model.state_dict(), os.path.join(self.checkpoint_dir, "last/last_model.pth")
            )
            print(f"Final epoch reached. Last model saved.")

    def _setup_dirs(self):
        """
        import os
        os.getcwd() - выводит абсолютный путь до проекта
        '/Users/macbook/Desktop/MagaDiplom/DMSS'
        """
        # self.best_checkpoint_dir = f'{self.checkpoint_dir}/best'
        # self.last_checkpoint_dir = f'{self.checkpoint_dir}/last'
        pass

