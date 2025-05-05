import os

from clearml import Logger
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
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
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float("inf")
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training.

        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch

        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None:
            return False

        if fitness > self.best_fitness or self.best_fitness == 0:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        stop = delta >= self.patience

        return stop


def visualize(output_dir, image_filename, task_name, **images):
    """
    Plot and save a row of images.
    Ключи **images: название→numpy-массив.
    """
    os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
    n = len(images)
    plt.figure(figsize=(5 * n, 5))

    for idx, (name, img) in enumerate(images.items(), start=1):
        ax = plt.subplot(1, n, idx)
        ax.set_title(name.replace("_", " ").title())
        ax.axis("off")

        # если 2D — ставим градации серого
        if img.ndim == 2:
            plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            plt.imshow(img)

    save_path = os.path.join(output_dir, task_name, image_filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def _calculate_metrics(tp, fp, fn, tn):
    tp = torch.tensor([tp])
    fp = torch.tensor([fp])
    fn = torch.tensor([fn])
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


def log_metrics_to_clearml(metrics: dict, epoch: int, logger: Logger):
    for metric_name, value in metrics.items():
        if torch.is_tensor(value):
            value = value.item()  # превращаем в float
        logger.report_scalar(
            title="Validation metrics",  # это название группы графиков
            series=metric_name,  # подпись линии/сериала
            value=value,  # само значение
            iteration=epoch,  # номер эпохи
        )


class SegmentationTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler,
        device: str = None,
        num_epochs: int = 10,
        patience: int = 50,
        logger: Logger = None,
        name_tsk: str = None,
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
        self.output_dir = "./reports"  # todo: проверить пути
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.name_task = name_tsk

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {self.epoch}/{self.num_epochs} [Train]"
        )
        for images, masks in progress_bar:
            # Переносим данные на выбранное устройство
            images, masks = images.to(self.device), masks.to(self.device)
            # print(images)
            # print(masks)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()
            progress_bar.set_postfix(train_loss=running_loss / (progress_bar.n + 1e-7))

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        tp = fp = fn = tn = 0

        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader, desc=f"Epoch {self.epoch}/{self.num_epochs} [Valid]"
            )
            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                running_loss += loss.item()

                # подготовим ground-truth
                masks = masks.clone()
                masks[masks == -1] = 0
                masks_int = masks.long().squeeze(1)  # [B,H,W]

                # вероятностная карта и бинарная маска
                prob_map = outputs.sigmoid().squeeze(1)  # [B,H,W]
                pred_mask = (prob_map > 0.5).long()

                # для каждого 3 примера в батче — визуализация
                for i in range(images.size(0)):
                    if i % 3 == 0:
                        # назад к HWC float [0,1]
                        inp = images[i].cpu().numpy().transpose(1, 2, 0)
                        inp = np.clip(inp * self.std + self.mean, 0, 1)
                        inp = inp[..., ::-1]

                        prob = prob_map[i].cpu().numpy()  # H,W
                        pred = pred_mask[i].cpu().numpy()  # H,W {0,1}
                        true = masks_int[i].cpu().numpy()  # H,W {0,1}

                        visualize(
                            self.output_dir,
                            f"epoch{self.epoch}_sample{i}.png",
                            task_name=self.name_task,
                            input_image=inp,
                            prob_map=prob,
                            pred_mask=pred,
                            true_mask=true,
                        )

                # подсчёт статистик
                batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                    pred_mask, masks_int, mode="binary"
                )
                tp += batch_tp.sum().item()
                fp += batch_fp.sum().item()
                fn += batch_fn.sum().item()
                tn += batch_tn.sum().item()

                progress_bar.set_postfix(val_loss=running_loss / (progress_bar.n + 1e-7))

        metrics = _calculate_metrics(tp, fp, fn, tn)
        print(f"Validation F1 = {metrics['f1_score']}")
        return running_loss / len(self.val_loader), metrics

    def train(self):
        # self.best_val_loss = float("inf")
        self.best_f1 = 0
        self.epoch = 0

        for epoch in range(1, self.num_epochs + 1):
            self.epoch = epoch
            # ------- Train -------
            train_loss = self.train_epoch()
            # ------- Validate -------
            final_epoch = epoch >= self.num_epochs
            val_loss, metrics = self.validate()
            # --------Log losses--------
            self.logger.report_scalar("Train", "loss", iteration=self.epoch, value=train_loss)
            self.logger.report_scalar("Valid", "loss", iteration=self.epoch, value=val_loss)
            log_metrics_to_clearml(metrics, self.epoch, self.logger)

            # if (
            #     val_loss < self.best_val_loss
            # ):  # Остановку может быть стоит делать не по лоссу, а по метрике
            #     self.best_val_loss = val_loss
            #     self._save_model("best")
            #     print(f"Best validation loss updated to {val_loss:.4f}")

            if metrics["f1_score"] > self.best_f1:
                self.best_f1 = metrics["f1_score"]
                self._save_model("best", task_name=self.name_task)
                print(f"Best f1 updated to {metrics['f1_score']:.4f}")
            self.stop |= self.stopper(epoch, metrics["f1_score"])

            # ------- Save model -------
            if final_epoch:
                self._save_model("last", task_name=self.name_task)
                print("Final epoch reached. Model saved.")

            # ------- Early stopping -------
            if self.stop:
                self._save_model("last", task_name=self.name_task)
                print("Early stopping triggered.")
                break

            print(
                f"Epoch {epoch}  Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}\n"
            )

    def _save_model(self, mode: str, task_name):
        """
        mode: 'best' or 'last'
        """
        if mode == "best":
            os.makedirs(os.path.join(self.checkpoint_dir, f"best_{task_name}"), exist_ok=True)
            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_dir, f"best_{task_name}/best_model.pth"),
            )
            print("Best model saved.")
        elif mode == "last":
            os.makedirs(os.path.join(self.checkpoint_dir, f"last_{task_name}"), exist_ok=True)
            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_dir, f"last_{task_name}/last_model.pth"),
            )
            print("Final epoch reached. Last model saved.")
