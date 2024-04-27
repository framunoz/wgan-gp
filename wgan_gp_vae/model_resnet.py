import abc
import functools
from typing import Callable, Iterable, Literal, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

_norm_layer_t = Optional[Literal["instance_norm", "batch_norm"]]
_device_t = str | torch.device
_resample_t = Optional[Literal["up", "down"]]
type seed_t = int | None


class UpsampleConv2d(nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return F.interpolate(output, scale_factor=2)


class AvgPoolConv2d(nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return F.avg_pool2d(output, kernel_size=2)


def norm_layer_factory(norm_layer: _norm_layer_t) -> Type[nn.Module]:
    # Create norm layer
    match norm_layer:
        case "instance_norm":
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)

        case "batch_norm":
            norm_layer = nn.BatchNorm2d

        case None:
            norm_layer = None

        case str(value):
            raise ValueError(
                f"Please choose an option between 'instance_norm' or 'batch_norm', not '{value}'."
            )

        case _:
            raise TypeError(
                "Please provide a string from the list ['instance_norm', 'batch_norm']."
            )

    return norm_layer


def resample_factory(
    resample: _resample_t, in_channels: int, out_channels: int
) -> tuple[nn.Module, nn.Module, Callable[..., nn.Module]]:
    kwargs = dict(kernel_size=3, padding=1, bias=False)
    match resample:
        case "up":
            # (C_in, H_in, W_in) -> (C_out, H_out, W_out)
            conv1 = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            # (C_out, H_out, W_out) -> (C_out, H_out, W_out)
            conv2 = nn.Conv2d(out_channels, out_channels, **kwargs)
            resample = nn.Upsample(scale_factor=2)

        case "down":
            # (C_in, H_in, W_in) -> (C_out, H_in, W_in)
            conv1 = nn.Conv2d(in_channels, out_channels, **kwargs)
            # (C_out, H_in, W_in) -> (C_out, H_out, W_out)
            conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )  # This parameters
            resample = nn.AvgPool2d(kernel_size=2)

        case None:
            conv1 = nn.Conv2d(in_channels, out_channels, **kwargs)
            conv2 = nn.Conv2d(out_channels, out_channels, **kwargs)
            resample = None

        case str(value):
            raise ValueError(
                f"Please choose an option between 'up', 'down' or None, not '{value}'."
            )

        case _:
            raise TypeError(
                "Please provide a string from the list ['up', 'down', None]."
            )

    return conv1, conv2, resample


class ResidualBlock(nn.Module):
    """Original Residual Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: _resample_t = None,
        norm_layer: _norm_layer_t = "batch_norm",
        act_func=nn.ReLU(),
    ):
        super().__init__()

        # Parameters
        self.in_channels, self.out_channels = in_channels, out_channels

        # Create norm layer
        self.norm_layer = norm_layer_factory(norm_layer)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Create convolutional layers and resample
        conv1, conv2, self.resample = resample_factory(
            resample, in_channels, out_channels
        )

        # Modules
        self.act_func = act_func
        self.norm1 = self.norm_layer(out_channels) if self.norm_layer else None
        self.conv1 = conv1
        self.norm2 = self.norm_layer(out_channels) if self.norm_layer else None
        self.conv2 = conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute identity
        identity = x  # (C_in, H_in, W_in)
        if self.shortcut is not None:
            # (C_in, H_in, W_in) -> (C_out, H_in, W_in)
            identity = self.shortcut(identity)
        if self.resample is not None:
            # (C_out, H_in, W_in) -> (C_out, H_out, W_out)
            identity = self.resample(identity)

        out = x

        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        out += identity
        out = self.act_func(out)

        return out


class ResidualBlockV2(ResidualBlock):
    """Residual Block for ResNet v2 https://arxiv.org/pdf/1603.05027.pdf"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: _resample_t = None,
        norm_layer: _norm_layer_t = "batch_norm",
        act_func=nn.ReLU(),
    ):
        super().__init__(in_channels, out_channels, resample, norm_layer, act_func)

        self.norm1 = self.norm_layer(in_channels) if self.norm_layer else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute identity
        identity = x  # (C_in, H_in, W_in)
        if self.shortcut is not None:
            # (C_in, H_in, W_in) -> (C_out, H_in, W_in)
            identity = self.shortcut(identity)
        if self.resample is not None:
            # (C_out, H_in, W_in) -> (C_out, H_out, W_out)
            identity = self.resample(identity)

        out = x

        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.act_func(out)
        out = self.conv1(out)

        if self.norm2 is not None:
            out = self.norm2(out)
        out = self.act_func(out)
        out = self.conv2(out)

        out += identity

        return out


_Block = ResidualBlock


def set_generator(seed: seed_t = None, device: _device_t = "cpu") -> torch.Generator:
    """
    Set the generator for the random number generator.

    :param seed: The seed for the random number generator.
    :param device: The device to set the generator.
    :return: The generator.
    """
    gen = torch.Generator(device=device)
    if seed is None:
        gen.seed()
        return gen
    gen.manual_seed(seed)
    return gen


class latent_distr(abc.ABC):
    gen: torch.Generator
    device: _device_t
    dtype: torch.dtype

    @abc.abstractmethod
    def __call__(self, n_dim: int, device: _device_t = "cpu"):
        """Return a tensor of shape (n_dim, z_dim, 1, 1)"""
        pass

    @abc.abstractmethod
    def log_prob(self, z: torch.Tensor, device: _device_t = "cpu") -> torch.Tensor:
        """Return the log probability of z."""
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__


class unif(latent_distr):
    def __init__(
        self,
        z_dim: int,
        device: _device_t,
        dtype: torch.dtype = torch.float32,
        a: float = -1,
        b: float = 1,
        seed: seed_t = None
    ):
        self.z_dim = z_dim
        self.a = torch.tensor(float(a), device=device)
        self.b = torch.tensor(float(b), device=device)
        self.gen = set_generator(seed=seed, device=device)
        self.device = device
        self.dtype = dtype

    def __call__(self, n_dim: int):
        unif_ = torch.rand(
            (n_dim, self.z_dim, 1, 1), generator=self.gen,
            device=self.device, dtype=self.dtype
        )
        return self.a + (self.b - self.a) * unif_
        # return self.distr.sample((n_dim, self.z_dim, 1, 1))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        restrict = ((z >= self.a) & (z <= self.b)).bool()
        restrict = torch.all(restrict, dim=1)
        return torch.where(
            restrict,
            0,
            torch.finfo(z.dtype).min,
        ).unsqueeze(1)


class norm(latent_distr):
    def __init__(
        self,
        z_dim: int,
        device: _device_t,
        dtype: torch.dtype = torch.float32,
        loc: float = 0,
        scale: float = 1,
        seed: seed_t = None
    ):
        self.z_dim = z_dim
        self.loc = torch.tensor(float(loc), device=device)
        self.scale = torch.tensor(float(scale), device=device)
        self.gen = set_generator(seed=seed, device=device)
        self.device = device
        self.dtype = dtype

    def __call__(self, n_dim: int):
        randn = torch.randn(
            (n_dim, self.z_dim, 1, 1), generator=self.gen,
            device=self.device, dtype=self.dtype
        )
        return self.loc + self.scale * randn
        # return self.distr.sample((n_dim, self.z_dim, 1, 1))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        norm_z_2 = torch.sum(z ** 2, dim=1)
        return -0.5 * norm_z_2.unsqueeze(1)


_LATENT_DISTR = {
    "unif": unif,
    "norm": norm,
}


class LatentDistribution:
    def __init__(
        self,
        name: Literal["unif", "norm"],
        z_dim: int,
        device: _device_t = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: seed_t = None,
        **kwargs,
    ):
        self.name = name
        self.device = device
        self.dtype = dtype
        self.z_dim = z_dim
        self._lat_distr: latent_distr = _LATENT_DISTR[name](
            z_dim, device, dtype, seed=seed, **kwargs
        )

    def __call__(self, n_dim: int):
        return self._lat_distr(n_dim)

    def __repr__(self):
        to_return = f"{self.__class__.__name__}"
        to_return += "("
        to_return += f"name={self.name}, "
        to_return += f"z_dim={self.z_dim}, "
        to_return += f"device={self.device}, "
        to_return += f"dtype={self.dtype}, "
        to_return += f"seed={self._lat_distr.gen.initial_seed()}"
        to_return += ")"
        return to_return


# Generator model
class Generator(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        channels_img: int = 3,
        num_filters: Iterable[int] = (128, 128, 128, 128, 128),
        resample_list: Iterable[Optional[str]] = ("up", "up", "up", None),
        Block=_Block,
        latent_distr: str | latent_distr = "norm",
    ) -> None:
        super().__init__()
        resample_list = (
            ["up"] * (len(num_filters) - 1) if resample_list is None else resample_list
        )
        if len(num_filters) - 1 != len(resample_list):
            raise ValueError(
                f"Iterables 'num_filters' and 'resample_list' must have same length. {len(num_filters)} - 1 != {len(resample_list)}"
            )

        self.latent_dim = latent_dim

        # Hidden Layers
        # (latent_dim, 1, 1) -> (latent_dim, 4, 4)
        self.upsample = nn.ConvTranspose2d(latent_dim, num_filters[0], 4)

        self.res_layers = nn.Sequential()
        for i in range(len(num_filters) - 1):
            res = Block(num_filters[i], num_filters[i + 1], resample=resample_list[i])
            self.res_layers.add_module(f"res{i + 1}", res)

        self.conv_out = nn.Conv2d(
            num_filters[-1], channels_img, kernel_size=3, padding=1
        )
        self.latent_distr_name: Literal["unif", "norm"] = latent_distr

    def forward(self, x):
        x = self.upsample(x)
        x = self.res_layers(x)
        x = self.conv_out(x)
        x = F.tanh(x)
        return x


# critic model
class Critic(nn.Module):
    def __init__(
        self,
        channels_img=3,
        num_filters: Iterable[int] = (128, 128, 128, 128),
        resample_list=("down", "down", None, None),
        Block=_Block,
    ):
        super().__init__()
        if len(num_filters) != len(resample_list):
            raise ValueError(
                "Iterables 'num_filters' and 'resample_list' must have same length."
            )
        num_filters = [channels_img] + list(num_filters)

        # Hidden Layers
        self.res_layers = nn.Sequential()
        for i in range(len(num_filters) - 1):
            norm_layer = "instance_norm" if i > 0 else None
            res = Block(
                num_filters[i],
                num_filters[i + 1],
                resample=resample_list[i],
                norm_layer=norm_layer,
                act_func=nn.LeakyReLU(0.2),
            )
            self.res_layers.add_module(f"res{i + 1}", res)

        self.down_sample = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(num_filters[-1], 1)

    def forward(self, x):
        x = self.res_layers(x)
        x = self.down_sample(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x[:, :, None, None]
        return x


# Encoder model
class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        channels_img=3,
        num_filters: Iterable[int] = (128, 128, 128, 128),
        resample_list=("down", "down", "down", None),
        Block=_Block,
    ):
        super().__init__()
        if len(num_filters) != len(resample_list):
            raise ValueError(
                "Iterables 'num_filters' and 'resample_list' must have same length."
            )
        num_filters = [channels_img] + list(num_filters)

        # Hidden Layers
        self.res_layers = nn.Sequential()
        for i in range(len(num_filters) - 1):
            norm_layer = "batch_norm" if i > 0 else None
            res = Block(
                num_filters[i],
                num_filters[i + 1],
                resample=resample_list[i],
                norm_layer=norm_layer,
                act_func=nn.LeakyReLU(0.2),
            )
            self.res_layers.add_module(f"res{i + 1}", res)

        self.output_layer = nn.Sequential()
        # Convolutional Layer
        out = nn.Conv2d(
            num_filters[-1],
            latent_dim,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.output_layer.add_module("out", out)

        # Normalization
        batch_norm = nn.BatchNorm2d(latent_dim)
        self.output_layer.add_module("bn_out", batch_norm)

    def forward(self, x):
        x = self.res_layers(x)
        x = self.output_layer(x)
        return x


def _test():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, in_channels, H, W = 8, 3, 32, 32
    latent_dim = 100
    x = torch.randn((N, in_channels, H, W)).to(dev)
    z = torch.randn((N, latent_dim, 1, 1)).to(dev)

    out_channels = 32
    conv = UpsampleConv2d(in_channels, out_channels, kernel_size=3, padding=1).to(dev)
    assert conv(x).shape == (
        N,
        out_channels,
        H * 2,
        W * 2,
    ), "UpsampleConv2d test failed"

    out_channels = 128
    conv = AvgPoolConv2d(in_channels, out_channels, kernel_size=3, padding=1).to(dev)
    assert conv(x).shape == (N, out_channels, H / 2, W / 2), "AvgPoolConv2d test failed"

    out_channels = 256
    # Check norm layer is a correct instance
    rb = ResidualBlock(
        in_channels,
        out_channels,
        norm_layer="instance_norm",
    ).to(dev)
    assert isinstance(
        rb.norm1, nn.InstanceNorm2d
    ), "norm_layer parameter does not works."

    rb = ResidualBlock(in_channels, out_channels, resample="up").to(dev)
    assert rb(x).shape == (
        N,
        out_channels,
        H * 2,
        W * 2,
    ), "ResidualBlock test failed with 'up'"

    rb = ResidualBlock(in_channels, out_channels, resample="down").to(dev)
    assert rb(x).shape == (
        N,
        out_channels,
        H / 2,
        W / 2,
    ), "ResidualBlock test failed with 'down'"

    rb = ResidualBlock(in_channels, out_channels).to(dev)
    assert rb(x).shape == (N, out_channels, H, W), "ResidualBlock test failed"

    # Generator test
    gen = Generator(
        latent_dim, in_channels, [128, 64, 32, 16, 8], latent_distr="norm"
    ).to(dev)
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print(gen)

    # Critic test
    critic = Critic(in_channels, [8, 16, 32, 64]).to(dev)
    assert critic(x).shape == (N, 1, 1, 1), "Critic test failed"
    print(critic)

    # Encoder
    enc = Encoder(latent_dim, in_channels, [16, 32, 64, 128]).to(dev)
    assert enc(x).shape == (N, latent_dim, 1, 1), "Encoder test failed"
    print(enc)

    z_ = gen.sample_noise(N)
    # print(z_.device)
    # print(gen.latent_distr.log_prob(z_).shape)
    z_ = gen.sample_noise(N)
    # z_ = torch.ones_like(z_) * 2
    # z_[0, 0, 0, 0] = 2
    print(z_.shape)
    print(gen.latent_distr.log_prob(z_))
    print(gen.latent_distr.log_prob(z_).shape)
    print(gen.latent_distr.name)

    print("Tests passed")


if __name__ == "__main__":
    _test()
