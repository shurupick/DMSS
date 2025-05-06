import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToPILImage
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from peft import get_peft_model, LoraConfig
from dmss.train_utils import (
    visualize,
    _calculate_metrics,
    log_metrics_to_clearml,
    EarlyStopping
)
import segmentation_models_pytorch as smp


class DiffusionSegmentationTrainer:
    def __init__(
        self,
        train_loader,
        val_loader,
        cfg,
        logger,
        task_name="diffusion_lora_seg",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.task_name = task_name
        self.device = cfg.device
        self.output_dir = "./reports"
        self.checkpoint_dir = "./models"
        self.prompt = cfg.prompt

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Load base pipeline
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-seg",
            torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)

        # Apply LoRA
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=["to_q", "to_k"]
        )
        self.pipe.unet = get_peft_model(self.pipe.unet, lora_config)
        self.pipe.unet.train()

        self.optimizer = torch.optim.Adam(self.pipe.unet.parameters(), lr=cfg.learning_rate)
        self.epochs = cfg.epochs
        self.guidance_scale = cfg.guidance_scale
        self.early_stopper = EarlyStopping(patience=cfg.patience)
        self.best_f1 = 0.0

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            self.pipe.unet.train()
            train_loss = 0
            for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                preds = []
                for img in images:
                    pil_img = ToPILImage()(img.cpu())
                    out = self.pipe(
                        prompt=self.prompt,
                        image=pil_img,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=20,
                    ).images[0]

                    mask_tensor = torch.tensor(np.array(out)).float() / 255.0  # [H, W]
                    if mask_tensor.ndim == 3:
                        mask_tensor = mask_tensor.mean(dim=-1)  # grayscale
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                    mask_tensor = F.interpolate(mask_tensor, size=img.shape[1:], mode="bilinear")
                    preds.append(mask_tensor)

                preds = torch.cat(preds, dim=0).to(self.device)  # [B,1,H,W]
                loss = F.binary_cross_entropy(preds, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            self.logger.report_scalar("Train", "loss", iteration=epoch, value=train_loss)

            val_loss, metrics = self.validate()
            self.logger.report_scalar("Valid", "loss", iteration=epoch, value=val_loss)
            log_metrics_to_clearml(metrics, epoch, self.logger)

            if metrics["f1_score"] > self.best_f1:
                self.best_f1 = metrics["f1_score"]
                self._save_model("best")
                print(f"Best F1 updated to {self.best_f1:.4f}")

            if epoch == self.epochs or self.early_stopper(epoch, metrics["f1_score"]):
                self._save_model("last")
                print("Early stopping triggered or final epoch reached.")
                break

    def validate(self):
        self.pipe.unet.eval()
        val_loss = 0
        tp = fp = fn = tn = 0

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Valid]"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                preds = []
                for img in images:
                    pil_img = ToPILImage()(img.cpu())
                    out = self.pipe(
                        prompt=self.prompt,
                        image=pil_img,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=20,
                    ).images[0]

                    mask_tensor = torch.tensor(np.array(out)).float() / 255.0
                    if mask_tensor.ndim == 3:
                        mask_tensor = mask_tensor.mean(dim=-1)
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                    mask_tensor = F.interpolate(mask_tensor, size=img.shape[1:], mode="bilinear")
                    preds.append(mask_tensor)

                preds = torch.cat(preds, dim=0).to(self.device)
                loss = F.binary_cross_entropy(preds, masks)
                val_loss += loss.item()

                prob_map = preds.sigmoid().squeeze(1)
                pred_mask = (prob_map > 0.5).long()
                true_mask = (masks > 0.5).long().squeeze(1)

                for i in range(images.size(0)):
                    if i % 3 == 0:
                        inp = images[i].cpu().numpy().transpose(1, 2, 0)
                        inp = np.clip(inp * self.std + self.mean, 0, 1)[..., ::-1]

                        prob = prob_map[i].cpu().numpy()
                        pred = pred_mask[i].cpu().numpy()
                        true = true_mask[i].cpu().numpy()

                        visualize(
                            self.output_dir,
                            f"epoch{self.epoch}_sample{i}.png",
                            task_name=self.task_name,
                            input_image=inp,
                            prob_map=prob,
                            pred_mask=pred,
                            true_mask=true,
                        )

                stats = smp.metrics.get_stats(pred_mask, true_mask, mode="binary")
                tp += stats[0].sum().item()
                fp += stats[1].sum().item()
                fn += stats[2].sum().item()
                tn += stats[3].sum().item()

        metrics = _calculate_metrics(tp, fp, fn, tn)
        return val_loss / len(self.val_loader), metrics

    def _save_model(self, mode: str):
        path = os.path.join(self.checkpoint_dir, f"{mode}_{self.task_name}")
        os.makedirs(path, exist_ok=True)
        self.pipe.unet.save_pretrained(path)
        print(f"{mode.capitalize()} model saved to {path}")
