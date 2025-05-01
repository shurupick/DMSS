import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class PolypModel(nn.Module):
    def __init__(
        self,
        arch: str = "Unet",
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        out_classes: int = 1,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.std = torch.tensor(params["std"]).view(1, 3, 1, 1).to(device)
        self.mean = torch.tensor(params["mean"]).view(1, 3, 1, 1).to(device)

    def forward(self, image):
        mask = self.model(image)
        return mask
