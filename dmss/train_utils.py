import os

from clearml import Logger, Task
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
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


def visualize(output_dir, image_filename, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)

    plt.show()
    plt.savefig(os.path.join(output_dir, image_filename))
    plt.close()


def _calculate_metrics(tp, fp, fn, tn):
    tp = (torch.tensor([tp]),)
    fp = (torch.tensor([fp]),)
    fn = (torch.tensor([fn]),)
    tn = torch.tensor([tn])

    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    recall = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro")
    precision = smp.metrics.positive_predictive_value(tp, fp, fn, tn, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

    return {
        "iou_score": iou_score,
        "f1_score": f1_score,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
    }


class SegmentationTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: smp.losses,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        device: str = None,
        num_epochs: int = 10,
        patience: int = 50,
        logger: Logger = None,
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
        self.logger = logger if logger else None
        self.output_dir = "./reports"

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
        tp, fp, fn, tn = 0, 0, 0, 0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)

                for i, output in enumerate(outputs):
                    input = images[i].cpu().numpy().transpose(1, 2, 0)
                    output = output.squeeze().cpu().numpy()

                    visualize(
                        self.output_dir,
                        f"output_{i}.png",
                        input_image=input,
                        output_mask=output,
                        binary_mask=output > 0.5,
                    )

                running_loss += loss.item()

                prob_mask = outputs.sigmoid().squeeze(1)
                pred_mask = (prob_mask > 0.5).long()
                batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                    pred_mask, masks, mode="binary"
                )

                tp += batch_tp.sum().item()
                fp += batch_fp.sum().item()
                fn += batch_fn.sum().item()
                tn += batch_tn.sum().item()

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
            if (
                val_loss < self.best_val_loss
            ):  # Остановку может быть стоит делать не по лоссу, а по метрике
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
            os.makedirs(os.path.join(self.checkpoint_dir, "best"), exist_ok=True)
            torch.save(
                self.model.state_dict(), os.path.join(self.checkpoint_dir, "best/best_model.pth")
            )
            print(f"Best model saved.")
        elif mode == "last":
            os.makedirs(os.path.join(self.checkpoint_dir, "last"), exist_ok=True)
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
