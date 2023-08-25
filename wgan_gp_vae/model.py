from typing import Iterable, Literal

import torch
import torch.nn as nn


def _unif(n_dim, z_dim, device="cpu"):
    return 2 * torch.rand(n_dim, z_dim, 1, 1).to(device) - 1


def _norm(n_dim, z_dim, device="cpu"):
    return torch.randn(n_dim, z_dim, 1, 1).to(device)


_LATENT_DISTR = {
    "unif": _unif,
    "norm": _norm,
}


# Generator model
class Generator(nn.Module):
    latent_distr: Literal["unif", "norm"] = "norm"

    def __init__(
        self,
        latent_dim,
        channels_img=3,
        num_filters: Iterable[int] = (256, 128, 64, 32),
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # Hidden layers
        self.hidden_layer = nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer:
            if i == 0:
                deconv = nn.ConvTranspose2d(
                    in_channels=latent_dim,
                    out_channels=num_filters[i],
                    kernel_size=4,
                    stride=1,
                    padding=0,
                )
            else:
                deconv = nn.ConvTranspose2d(
                    in_channels=num_filters[i - 1],
                    out_channels=num_filters[i],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )

            deconv_name = f"deconv_{i+1}"
            self.hidden_layer.add_module(deconv_name, deconv)

            nn.init.normal_(deconv.weight.data, mean=0.0, std=0.02)
            nn.init.constant_(deconv.bias.data, 0.0)

            # Batch normalization
            batch_norm = nn.BatchNorm2d(num_filters[i])
            bn_name = f"bn_{i+1}"
            self.hidden_layer.add_module(bn_name, batch_norm)
            nn.init.normal_(batch_norm.weight.data, mean=0.0, std=0.02)
            nn.init.constant_(batch_norm.bias.data, 0.0)

            # Activation
            act_name = f"act_{i+1}"
            self.hidden_layer.add_module(act_name, nn.ReLU())

        # Output layer
        self.output_layer = nn.Sequential()

        # Deconvolutional layer
        out = nn.ConvTranspose2d(
            in_channels=num_filters[i],
            out_channels=channels_img,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.output_layer.add_module("out", out)
        nn.init.normal_(out.weight.data, mean=0.0, std=0.02)
        nn.init.constant_(out.bias.data, 0.0)

        # Activation
        self.output_layer.add_module("act_out", nn.Tanh())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x):
        x = self.hidden_layer(x)
        out = self.output_layer(x)
        return out

    def sample_noise(self, n: int) -> torch.Tensor:
        return _LATENT_DISTR[self.latent_distr](n, self.latent_dim, self.device)


# critic model
class Critic(nn.Module):
    def __init__(self, channels_img=3, num_filters: Iterable[int] = (32, 64, 128, 256)):
        super().__init__()

        # Hidden layers
        self.hidden_layer = nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                conv = nn.Conv2d(
                    in_channels=channels_img,
                    out_channels=num_filters[i],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            else:
                conv = nn.Conv2d(
                    in_channels=num_filters[i - 1],
                    out_channels=num_filters[i],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )

            conv_name = f"conv_{i+1}"
            self.hidden_layer.add_module(conv_name, conv)

            # Initializer
            nn.init.normal_(conv.weight.data, mean=0.0, std=0.02)
            nn.init.constant_(conv.bias.data, 0.0)

            # Batch normalization
            if i > 0:
                inst_norm = nn.InstanceNorm2d(num_filters[i], affine=True)
                in_name = f"in_{i+1}"
                self.hidden_layer.add_module(in_name, inst_norm)
                nn.init.normal_(inst_norm.weight.data, mean=0.0, std=0.02)
                nn.init.constant_(inst_norm.bias.data, 0.0)

            # Activation
            act_name = f"act_{i+1}"
            self.hidden_layer.add_module(act_name, nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = nn.Sequential()
        # Convolutional layer
        out = nn.Conv2d(
            in_channels=num_filters[i],
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=0,
        )
        self.output_layer.add_module("out", out)
        # Initializer
        nn.init.normal_(out.weight.data, mean=0.0, std=0.02)
        nn.init.constant_(out.bias.data, 0.0)

    def forward(self, x):
        x = self.hidden_layer(x)
        out = self.output_layer(x)
        return out


# Encoder model
class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        channels_img=3,
        num_filters: Iterable[int] = (32, 64, 128, 256),
    ):
        super().__init__()

        # Hidden layers
        self.hidden_layer = nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                conv = nn.Conv2d(
                    in_channels=channels_img,
                    out_channels=num_filters[i],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            else:
                conv = nn.Conv2d(
                    in_channels=num_filters[i - 1],
                    out_channels=num_filters[i],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )

            conv_name = f"conv_{i+1}"
            self.hidden_layer.add_module(conv_name, conv)

            # Initializer
            nn.init.normal_(conv.weight.data, mean=0.0, std=0.02)
            nn.init.constant_(conv.bias.data, 0.0)

            # Batch normalization
            if i > 0:
                batch_norm = nn.BatchNorm2d(num_filters[i])
                bn_name = f"bn_{i+1}"
                self.hidden_layer.add_module(bn_name, batch_norm)
                nn.init.normal_(batch_norm.weight.data, mean=0.0, std=0.02)
                nn.init.constant_(batch_norm.bias.data, 0.0)

            # Activation
            act_name = f"act_{i+1}"
            self.hidden_layer.add_module(act_name, nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = nn.Sequential()
        # Convolutional layer
        out = nn.Conv2d(
            in_channels=num_filters[i],
            out_channels=latent_dim,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.output_layer.add_module("out", out)
        # Initializer
        nn.init.normal_(out.weight.data, mean=0.0, std=0.02)
        nn.init.constant_(out.bias.data, 0.0)
        # Activation
        batch_norm = nn.BatchNorm2d(latent_dim)
        self.output_layer.add_module("bn_out", batch_norm)
        nn.init.normal_(batch_norm.weight.data, mean=0.0, std=0.02)
        nn.init.constant_(batch_norm.bias.data, 0.0)

    def forward(self, x):
        x = self.hidden_layer(x)
        out = self.output_layer(x)
        return out


def _test():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, in_channels, H, W = 8, 3, 64, 64
    latent_dim = 100
    x = torch.randn((N, in_channels, H, W)).to(dev)
    z = torch.randn((N, latent_dim, 1, 1)).to(dev)

    # Generator test
    gen = Generator(latent_dim, in_channels, [128, 64, 32, 16]).to(dev)
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"

    # Critic test
    critic = Critic(in_channels, [8, 16, 32, 64]).to(dev)
    assert critic(x).shape == (N, 1, 1, 1), "Critic test failed"

    # Encoder
    enc = Encoder(latent_dim, in_channels, [16, 32, 64, 128]).to(dev)
    assert enc(x).shape == (N, latent_dim, 1, 1), "Encoder test failed"

    print("Tests passed")


if __name__ == "__main__":
    _test()
