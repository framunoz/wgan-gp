from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from wgan_gp_vae.model import Encoder, Generator

_models_option = "encoder", "generator"
TypeModels = Literal["encoder", "generator"]

def gradient_penalty(critic, real, fake, device="cpu"):
    device = torch.device(device)
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def load_checkpoint(
        model: nn.Module, 
        root_path: str | Path, 
        type_model: TypeModels, 
        device=None
        ):
    if type_model not in _models_option:
        raise ValueError(f"Choose one of the options: {_models_option}.")
    source_path = Path(root_path) / type_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(
        torch.load(source_path, map_location=device)
    )
    model.eval()

    return model


def alexnet_norm(x):
    assert (
        x.max() <= 1 or x.min() >= 0
    ), f"Alexnet received input outside of range [0,1]: {x.min(),x.max()}"
    out = x - torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).type_as(x)
    out = out / torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).type_as(x)
    return out


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class ProjectorOnManifold:
    def __init__(
        self,
        encoder: Encoder,
        generator: Generator,
        image_size: tuple[int, int, int] = (1, 28, 28),
        image_size_net: tuple[int, int, int] = (1, 64, 64),
        transform_in=None,
        transform_out=None,
    ):
        self.encoder = encoder
        self.generator = generator
        self.img_size = image_size
        self._img_size_net = image_size_net
        channels_img = image_size_net[0]
        self._transform_in = transform_in or transforms.Compose([
            # From pdf to grayscale
            transforms.Lambda(lambda x: x / torch.max(x)),
            # transforms.Lambda(lambda x: x),
            transforms.ToPILImage(),
            transforms.Resize(image_size_net[1:]),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)]
            ),
        ])
        self._transform_out = transform_out or transforms.Compose([
            # Ensure the range is in [0, 1]
            transforms.Lambda(lambda x: x - torch.min(x)),
            transforms.Lambda(lambda x: x / torch.max(x)),
            transforms.ToPILImage(),
            transforms.Resize(image_size[1:]),
            # transforms.ToTensor(),
            transforms.Lambda(lambda x: x / torch.sum(x)),
        ])

    def forward(self, x):
        x = torch.unsqueeze(self._transform_in(x), 0)
        x = torch.squeeze(self.generator(self.encoder(x)))
        x = self._transform_out(x)
        return x

    def __call__(self, x):
        return self.forward(x)
    