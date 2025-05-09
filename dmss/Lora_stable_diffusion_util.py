import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import get_peft_model, LoraConfig
from transformers import CLIPTokenizer, CLIPTextModel
from dmss.train_utils import (
    visualize,
    _calculate_metrics,
    log_metrics_to_clearml,
    EarlyStopping,
)
import segmentation_models_pytorch as smp


class DiffusionSegmentationTrainer:
    def __init__(self,
                 train_loader,
                 val_loader,
                 cfg,
                 loss_fn,
                 logger,
                 task_name="diffusion_lora_seg"):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.logger = logger
        self.task_name = task_name
        self.device = cfg.device
        self.output_dir = "./reports"
        self.checkpoint_dir = "./models"
        self.loss_fn = loss_fn
        self.prompt = cfg.promt
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

        # Components from Stable Diffusion
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(self.device)
        base_unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(self.device)
        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            target_modules=["to_q", "to_k"]
        )
        self.unet = get_peft_model(base_unet, lora_config)
        self.scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        # Use default inference steps (e.g. 50)
        self.scheduler.set_timesteps(num_inference_steps=50)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=cfg.learning_rate)
        self.epochs = cfg.epochs
        self.early_stopper = EarlyStopping(patience=cfg.patience)
        self.best_f1 = 0.0
        self.image_cond_proj = torch.nn.Linear(4, 768).to(self.device)

    def _encode_prompt(self, prompt, batch_size):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(self.device)
        embeddings = self.text_encoder(tokens)[0]
        return embeddings.repeat(batch_size, 1, 1)

    def train(self):
        self.vae.eval()
        self.unet.train()

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            total_loss = 0.0

            for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]"):
                images = images.to(self.device)
                masks = masks.to(self.device)

                with torch.no_grad():
                    cond_latents = self.vae.encode(images).latent_dist.sample() * 0.18215  # z_c
                    masks_rgb = masks if masks.shape[1] == 3 else masks.repeat(1, 3, 1, 1)  # [B, 1, H, W] → [B, 3, H, W]
                    target_latents = self.vae.encode(masks_rgb).latent_dist.sample() * 0.18215  # z_0

                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (images.shape[0],), device=self.device).long()
                noisy_latents = self.scheduler.add_noise(target_latents, noise, timesteps)

                noise_pred = self.unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=torch.cat([
                        self._encode_prompt(self.prompt, images.size(0)),  # [B, 77, 768]
                        self.image_cond_proj(cond_latents.flatten(2).transpose(1, 2))  # [B, HxW, 4]
                    ], dim=1)
                ).sample

                alpha_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                z0_pred = (noisy_latents - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

                loss = F.l1_loss(z0_pred, target_latents) + F.l1_loss(noise_pred, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"[Epoch {epoch}] Train loss: {avg_loss:.4f}")
            self.logger.report_scalar("Train", "loss", iteration=epoch, value=avg_loss)

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
        self.unet.eval()
        self.vae.eval()
        val_loss = 0.0
        tp = fp = fn = tn = 0

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Valid]"):
                images = images.to(self.device)
                cond_latents = self.vae.encode(images).latent_dist.sample() * 0.18215

                latent_shape = cond_latents.shape
                noisy_latents = torch.randn(latent_shape, device=self.device)

                for t in self.scheduler.timesteps:
                    t_tensor = torch.tensor([t], dtype=torch.long, device=self.device)
                    noise_pred = self.unet(
                        sample=noisy_latents,
                        timestep=t_tensor,
                        encoder_hidden_states=torch.cat([
                            self._encode_prompt(self.prompt, images.size(0)),
                            self.image_cond_proj(cond_latents.flatten(2).transpose(1, 2))], dim=1)).sample
                    noisy_latents = self.scheduler.step(noise_pred, t_tensor, noisy_latents).prev_sample

                z0_pred = noisy_latents
                decoded = self.vae.decode(z0_pred / 0.18215).sample  # [B, C, H, W]
                prob_map = torch.sigmoid(decoded[:, 0])  # [B, H, W]
                pred_mask = (prob_map > 0.5).long()
                masks = masks.to(self.device)
                true_mask = (masks > 0.5).long().squeeze(1)  # no GT
                val_loss += self.loss_fn(prob_map.unsqueeze(1), masks).item()

                for i in range(images.size(0)):
                    if i % 3 == 0:
                        visualize(
                            self.output_dir,
                            f"epoch{self.epoch}_sample{i}.png",
                            task_name=self.task_name,
                            input_image=np.clip(images[i].cpu().permute(1, 2, 0).numpy(), 0, 1),
                            prob_map=prob_map[i].cpu().numpy(),
                            pred_mask=pred_mask[i].cpu().numpy(),
                            true_mask=true_mask[i].cpu().numpy(),
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
        self.unet.save_pretrained(path)
        print(f"{mode.capitalize()} model saved to {path}")
