import math

from matplotlib.pyplot import isinteractive
from torchvision import disable_beta_transforms_warning

disable_beta_transforms_warning()

import os
import typing as t
from copy import deepcopy
from pathlib import Path
from typing import Literal

import dataloader
import numpy as np
import quick_torch as qt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as T
import utils
from model_resnet import (Critic, Encoder, Generator, LatentDistribution,
                          ResidualBlock)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from utils import (alexnet_norm, bures_wass_dist, denorm, gradient_penalty,
                   imq_kernel)

NUM_WORKERS = 10
print(f"Using {NUM_WORKERS} workers.")
NOISE_NAME = "norm"
type_models = "resnet"

torch.backends.cudnn.benchmark = True


class CriticIterations:
    def __init__(
        self,
        critic_iterations,
        patience=5,
        max_diff_loss=10.0,
        min_critic_iter=2,
    ):
        self.k = 0
        self.last_loss = -float("inf")
        self.max_loss = -float("inf")
        self.new_loss = -float("inf")
        self.diff_loss = -float("inf")
        self.max_diff_loss = max_diff_loss or 5.0
        self.patience = patience or 5
        self.min_critic_iter = min_critic_iter or 2

        if callable(critic_iterations):
            try:
                _critic_iterations = critic_iterations
                critic_iterations(19, 1.0)

            except TypeError:

                def _critic_iterations(epoch, *args, **kwargs):
                    return critic_iterations(epoch)

            self.critic_iterations = _critic_iterations
        else:
            self.critic_iterations = critic_iterations

    def register_new_loss(self, new_loss):
        if new_loss > self.max_loss:
            self.max_loss = new_loss
        self.diff_loss = self.max_loss - new_loss
        self.last_loss = self.new_loss
        self.new_loss = new_loss

        if (
            callable(self.critic_iterations)
            or self.critic_iterations <= self.min_critic_iter
        ):
            return

        # If the difference is greater than the max_diff_loss, then decrease the critic_iterations
        decrese_by_difference = self.diff_loss > self.max_diff_loss
        if decrese_by_difference:
            self.max_loss = new_loss

        # If the new loss is less than the last loss, then increase the k
        if self.new_loss < self.last_loss:
            self.k += 1
        else:
            self.k = 0

        # If the k is greater than the patience, or the difference is
        #  greater than the max_diff_loss, then decrease the critic_iterations
        if self.k >= self.patience or decrese_by_difference:
            self.critic_iterations = max(
                self.min_critic_iter, self.critic_iterations - 1
            )
            self.k = 0

    def __call__(self, epoch):
        if callable(self.critic_iterations):
            return self.critic_iterations(epoch, self.new_loss)

        return self.critic_iterations

    def __repr__(self):
        tab = " " * 2
        to_return = self.__class__.__name__
        to_return += "("
        if callable(self.critic_iterations):
            to_return += f"critic_iterations={self.critic_iterations}"
            to_return += ")"
        else:
            to_return += (
                f"\n{tab}"
                + f"critic_iterations={self.critic_iterations}, "
                + f"min_critic_iter={self.min_critic_iter}, "
            )
            to_return += (
                f"\n{tab}"
                + f"last_loss={self.last_loss:.4f}, "
                + f"new_loss={self.new_loss:.4f}, "
            )
            to_return += f"\n{tab}" + f"k={self.k}, " + f"patience={self.patience}, "
            to_return += (
                f"\n{tab}"
                + f"max_loss={self.max_loss:.4f}, "
                + f"diff_loss={self.diff_loss:.4f}, "
                + f"max_diff_loss={self.max_diff_loss}, "
            )

            if to_return.endswith(", "):
                to_return = to_return[:-2]

            to_return += "\n)"

        return to_return


def sample_noise(n_dim, z_dim, device="cpu"):
    return torch.randn(n_dim, z_dim, 1, 1).to(device)


def get_device(device=None) -> torch.device:
    """Gets the device, or set a device if None is provided. Always prefer cuda."""
    # Define device
    if device is None:  # Default
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    return device


def get_dataset(transform, ds_type: str, file_path: str = None):
    dataset = None
    test_dataset = None

    if ds_type == "mnist":
        dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    elif ds_type == "celeba":
        dataset = datasets.ImageFolder(root="celeb_dataset", transform=transform)
    elif ds_type == "quickdraw":
        categories = [qt.Category.FACE]
        dataset_ = qt.QuickDraw(
            root="dataset",
            categories=categories,
            transform=transform,
            download=True,
            recognized=True,
            train_percentage=0.95,
        )
        path_dataset = Path(file_path)
        dataset_.data = np.load(
            path_dataset
            # / "cleaned"
            # / "data.npy"
            # path_dataset / "cleaned" / "data_sin_contorno.npy"
            # path_dataset / "cleaned" / "data_sin_contorno_arriba.npy"
        ).reshape(-1, 28, 28)
        dataset_.targets = np.ones(len(dataset_.data), dtype=int)
        dataset = dataset_.get_train_data()
        test_dataset = dataset_.get_test_data()

    file_path = Path(file_path)
    if dataset is not None:
        pass
    elif file_path.suffix == ".npy":
        dataset = dataloader.NpyDataset(root=file_path, transform=transform)
    elif file_path.suffix == ".h5py":
        dataset = dataloader.H5Loader(file_path, transform=transform)
    else:
        raise FileNotFoundError("El archivo buscado no fue encontrado.")

    return dataset, test_dataset


# Training methods
def train_wgan_and_wae(
    latent_dim,  # =100
    nn_kwargs=None,
    channels_img=3,
    learning_rate_E=1e-3,
    learning_rate_G=1e-4,
    learning_rate_C=1e-4,
    batch_size=128,
    image_size=64,
    num_epochs=100,
    critic_iterations=5,
    penalty_wgan_lp=10,
    penalty_wgan_gp=0.1,
    penalty_wae=10,
    criterion: Literal["mse", "l1"] = "l1",
    betas_wgan=(0.5, 0.9),
    betas_wae=(0.5, 0.999),
    weight_decay=1e-5,
    milestones=(25, 50, 75),
    patience=10,  # Number of epochs to wait for improvement
    min_delta=0,  # Minimum change in validation loss to be considered as an improvement
    device=None,
    transform=None,
    file_path: str | Literal["mnist", "celeba"] = "shoes_images/shoes.hdf5",
    save_dir="networks/",
    summary_writer_dir="logs",
    verbose=True,
    report_every=100,
):
    device = get_device(device)

    # Transforms
    transform = transform or T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    # Dataset and Dataloader
    dataset, val_dataset = get_dataset(file_path=file_path, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    if val_dataset:
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    # Models
    if nn_kwargs:
        G = Generator(latent_dim, channels_img, **nn_kwargs["generator"]).to(device)
        C = Critic(channels_img, **nn_kwargs["critic"]).to(device)
        E = Encoder(latent_dim, channels_img, **nn_kwargs["encoder"]).to(device)
    else:
        G = Generator(latent_dim, channels_img).to(device)
        C = Critic(channels_img).to(device)
        E = Encoder(latent_dim, channels_img).to(device)

    print(C)
    print(G)
    print(E)

    # Loss function for WAE
    match criterion:
        case "mse":
            criterion = nn.MSELoss()
        case "l1":
            criterion = nn.L1Loss()
        case _:
            criterion = criterion

    # Optimizers
    G_optimizer = optim.Adam(
        G.parameters(),
        lr=learning_rate_G,
        betas=betas_wgan,
        weight_decay=weight_decay,
    )
    C_optimizer = optim.Adam(
        C.parameters(),
        lr=learning_rate_C,
        betas=betas_wgan,
        weight_decay=weight_decay,
    )
    # Optimizer
    E_optimizer = optim.Adam(
        E.parameters(),
        lr=learning_rate_E,
        betas=betas_wae,
        weight_decay=weight_decay,
    )

    # for tensorboard plotting
    fixed_noise = G.sample_noise(32)
    summary_writer_dir = Path(summary_writer_dir)
    summary_writer_dir.mkdir(exist_ok=True, parents=True)
    writer_real = SummaryWriter(summary_writer_dir / "real")
    writer_fake = SummaryWriter(summary_writer_dir / "fake")
    writer_loss = SummaryWriter(summary_writer_dir / "loss")
    step = 0

    # Directory to save
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Schedulers
    G_scheduler = optim.lr_scheduler.MultiStepLR(
        G_optimizer, milestones=milestones, gamma=0.5
    )
    C_scheduler = optim.lr_scheduler.MultiStepLR(
        C_optimizer, milestones=milestones, gamma=0.5
    )
    E_scheduler = optim.lr_scheduler.MultiStepLR(
        E_optimizer, milestones=milestones, gamma=0.5
    )

    C.train()
    G.train()
    E.train()

    # Early stop
    current_patience = 0
    best_validation_loss = float("inf")

    real_val, _ = next(iter(val_data_loader))
    real_val = real_val.to(device)
    _penalty_wgan_gp = math.sqrt(
        penalty_wgan_gp / penalty_wgan_lp
    )  # the penalty that use the leaky relu

    for epoch in range(1, num_epochs + 1):
        # Free memory
        torch.cuda.empty_cache()

        epoch_wass_dist_WGAN = []
        epoch_wass_dist_WAE = []
        C_epoch_losses = []
        G_epoch_losses = []
        E_epoch_losses = []
        critic_real_values = []
        critic_fake_values = []
        gradient_norm_values = []

        if verbose:
            iterable = enumerate(
                tqdm(
                    data_loader,
                    desc=(
                        f"Epoch [{epoch}/{num_epochs}], Patience" f" {current_patience}"
                    ),
                )
            )
        else:
            iterable = enumerate(data_loader)

        # Training loop
        for batch_idx, (real, _) in iterable:
            n = real.shape[0]
            real = real.to(device)

            for _ in range(critic_iterations):
                # Train Critic: min E[critic(fake)] - E[critic(real)] + penalization
                z = G.sample_noise(n, type_as=real)
                fake = G(z)
                c_real = C(real).reshape(-1)
                c_fake = C(fake).reshape(-1)
                gradient, gradient_norm = gradient_penalty(
                    C, real, fake, device=device, penalty_gr=_penalty_wgan_gp
                )
                wasserstein_dist_WGAN = torch.mean(c_real) - torch.mean(c_fake)
                C_loss = -wasserstein_dist_WGAN + penalty_wgan_lp * gradient

                # Update parameters of the critic
                C.zero_grad()
                C_loss.backward(retain_graph=True)
                C_optimizer.step()

            for _ in range(critic_iterations):
                # Train encoder: min |real - generator(encoder(real))| + penalization
                z = G.sample_noise(n, type_as=real)
                z_tilde = E(real)  # Generate \tilde{z}_i from Q(Z|x_i) for i = 1:n
                x_recon = G(z_tilde)

                # According to the paper, this is the Wasserstein distance
                wasserstein_dist_WAE = criterion(real, x_recon)
                mmd_loss = imq_kernel(
                    z, z_tilde, p_distr=G.latent_distr.name
                )  # Compute MMD
                E_loss = wasserstein_dist_WAE + penalty_wae * mmd_loss

                # Update parameters of the encoder
                E.zero_grad()
                E_loss.backward(retain_graph=True)
                E_optimizer.step()

            # Train Generator: min Wasserstein distance
            z = G.sample_noise(n, type_as=real)
            wasserstein_dist_WGAN_ = torch.mean(C(real)) - torch.mean(C(G(z)))
            wasserstein_dist_WAE_ = criterion(real, G(E(real)))
            G_loss = wasserstein_dist_WGAN_ + wasserstein_dist_WAE_

            # Update parameters of the generator
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # loss values
            epoch_wass_dist_WGAN.append(wasserstein_dist_WGAN.data.item())
            epoch_wass_dist_WAE.append(wasserstein_dist_WAE.data.item())

            C_epoch_losses.append(C_loss.data.item())
            G_epoch_losses.append(G_loss.data.item())
            E_epoch_losses.append(E_loss.data.item())
            critic_real_value = torch.mean(c_real)
            critic_fake_value = torch.mean(c_fake)
            critic_real_values.append(critic_real_value.data.item())
            critic_fake_values.append(critic_fake_value.data.item())
            gradient_norm_mean = torch.mean(gradient_norm)
            gradient_norm_values.append(gradient_norm_mean.data.item())

            # Print losses occasionally and print to tensorboard
            if batch_idx % report_every == 0:
                with torch.no_grad():
                    fake_WGAN = G(fixed_noise)
                    fake_WAE = G(E(real))
                    fake_val_WAE = G(E(real_val))

                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_real_val = torchvision.utils.make_grid(
                        real_val[:32], normalize=True
                    )
                    img_grid_fake_WGAN = torchvision.utils.make_grid(
                        fake_WGAN[:32], normalize=True
                    )
                    img_grid_fake_WAE = torchvision.utils.make_grid(
                        fake_WAE[:32], normalize=True
                    )
                    img_grid_fake_val_WAE = torchvision.utils.make_grid(
                        fake_val_WAE[:32], normalize=True
                    )

                    loss_C_name = "Loss Discriminator"
                    loss_G_name = "Loss Generator"
                    loss_E_name = "Loss Encoder"
                    critic_values_name = "Critic Values"
                    wass_dist_name_WGAN = "Wasserstein Distance WGAN"
                    wass_dist_name_WAE = "Wasserstein Distance WAE"
                    gradient_norm_name = "Gradient Norm"
                    writer_real.add_image("1 Real", img_grid_real, global_step=step)
                    writer_fake.add_image(
                        "2 Decoded", img_grid_fake_WAE, global_step=step
                    )
                    writer_real.add_image(
                        "3 Real Val", img_grid_real_val, global_step=step
                    )
                    writer_fake.add_image(
                        "4 Decoded Val", img_grid_fake_val_WAE, global_step=step
                    )
                    writer_fake.add_image(
                        "5 Generated", img_grid_fake_WGAN, global_step=step
                    )
                    writer_loss.add_scalars(
                        loss_C_name, {loss_C_name: C_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        loss_G_name, {loss_G_name: G_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        loss_E_name, {loss_E_name: E_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        wass_dist_name_WGAN,
                        {wass_dist_name_WGAN: wasserstein_dist_WGAN},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        wass_dist_name_WAE,
                        {wass_dist_name_WAE: wasserstein_dist_WAE},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        gradient_norm_name,
                        {gradient_norm_name: gradient_norm_mean},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        critic_values_name,
                        {
                            "Real": critic_real_value,
                            "Fake": critic_fake_value,
                        },
                        global_step=step,
                    )

                step += 1

        # Validation loop
        if verbose:
            iterable = tqdm(val_data_loader, desc=f"Val. Epoch [{epoch}/{num_epochs}]")
        else:
            iterable = val_data_loader

        epoch_wass_dist_WAE_val = []
        epoch_wass_dist_WGAN_val = []
        with torch.no_grad():
            for real, _ in iterable:
                real = real.to(device)
                n = real.shape[0]

                # WAE Validation Loss
                recon = G(E(real))
                loss = criterion(real, recon)
                epoch_wass_dist_WAE_val.append(loss.data.item())

                # WGAN Validation Loss
                z = G.sample_noise(n, type_as=real)
                fake = G(z)
                c_real, c_fake = C(real).reshape(-1), C(fake).reshape(-1)
                loss = torch.mean(c_real) - torch.mean(c_fake)
                epoch_wass_dist_WGAN_val.append(loss.data.item())

        # Calculate the average validation loss
        avg_wass_dist_WAE_val = torch.mean(
            torch.FloatTensor(epoch_wass_dist_WAE_val)
        ).item()
        avg_wass_dist_WGAN_val = torch.mean(
            torch.FloatTensor(epoch_wass_dist_WGAN_val)
        ).item()

        if avg_wass_dist_WAE_val < best_validation_loss - min_delta:
            best_validation_loss = avg_wass_dist_WAE_val
            current_patience = 0
            # Save models
            torch.save(G.state_dict(), save_dir / "generator")
            torch.save(C.state_dict(), save_dir / "discriminator")
            torch.save(E.state_dict(), save_dir / "encoder")
        else:
            current_patience += 1

        global_epoch = step - 1
        avg_wass_dist_WGAN = torch.mean(torch.FloatTensor(epoch_wass_dist_WGAN)).item()
        avg_wass_dist_WAE = torch.mean(torch.FloatTensor(epoch_wass_dist_WAE)).item()
        C_avg_loss = torch.mean(torch.FloatTensor(C_epoch_losses)).item()
        G_avg_loss = torch.mean(torch.FloatTensor(G_epoch_losses)).item()
        E_avg_loss = torch.mean(torch.FloatTensor(E_epoch_losses)).item()
        critic_real_avg_loss = torch.mean(torch.FloatTensor(critic_real_values)).item()
        critic_fake_avg_loss = torch.mean(torch.FloatTensor(critic_fake_values)).item()
        gradient_norm_values_avg = torch.mean(
            torch.FloatTensor(gradient_norm_values)
        ).item()
        writer_loss.add_scalars(
            wass_dist_name_WGAN,
            {"Avg. " + wass_dist_name_WGAN: avg_wass_dist_WGAN},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WAE,
            {"Avg. " + wass_dist_name_WAE: avg_wass_dist_WAE},
            global_epoch,
        )
        writer_loss.add_scalars(
            gradient_norm_name,
            {"Avg. " + gradient_norm_name: gradient_norm_values_avg},
            global_epoch,
        )
        writer_loss.add_scalars(
            loss_C_name, {"Avg. " + loss_C_name: C_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            loss_G_name, {"Avg. " + loss_G_name: G_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            loss_E_name, {"Avg. " + loss_E_name: E_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            critic_values_name,
            {
                "Avg. Real": critic_real_avg_loss,
                "Avg. Fake": critic_fake_avg_loss,
            },
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WAE,
            {"Avg. " + wass_dist_name_WAE + " Validation": avg_wass_dist_WAE_val},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WGAN,
            {"Avg. " + wass_dist_name_WGAN + " Validation": avg_wass_dist_WGAN_val},
            global_epoch,
        )

        if current_patience >= patience:
            print("Early stopping triggered. Training halted.")
            break

        # Decrease learning-rate
        G_scheduler.step()
        C_scheduler.step()
        E_scheduler.step()


# Training methods
# MARK: - WGAN and WAE with optimized training
def train_wgan_and_wae_optimized(
    latent_dim,  # =100
    nn_kwargs=None,
    channels_img=3,
    learning_rate_E=1e-3,
    learning_rate_G=1e-4,
    learning_rate_C=1e-4,
    batch_size=128,
    image_size: int | tuple[int, int] = 64,
    num_epochs=100,
    critic_iterations=5,
    crit_iter_patience=None,
    penalty_wgan_lp=10,
    penalty_wgan_gp=0.05,
    penalty_wae=10,
    criterion: Literal["mse", "l1"] = "l1",
    betas_wgan=(0.5, 0.9),
    betas_wae=(0.5, 0.999),
    weight_decay=1e-5,
    milestones=(25, 50, 75),
    patience=20,  # Number of epochs to wait for improvement
    min_delta=0,  # Minimum change in validation loss to be considered as an improvement
    device=None,
    transform=None,
    file_path: str | Path = None,
    ds_type="quickdraw",
    save_dir: str | Path = "networks/",
    summary_writer_dir: str | Path = "logs",
    verbose=True,
    report_every=100,
):
    device = get_device(device)

    critic_iterations = CriticIterations(critic_iterations, patience=crit_iter_patience)

    # Transforms
    transform = transform or T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    # Dataset and Dataloader
    dataset, val_dataset = get_dataset(
        ds_type=ds_type, file_path=file_path, transform=transform
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if val_dataset:
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    # Models
    if nn_kwargs:
        G = Generator(latent_dim, channels_img, **nn_kwargs["generator"]).to(device)
        C = Critic(channels_img, **nn_kwargs["critic"]).to(device)
        E = Encoder(latent_dim, channels_img, **nn_kwargs["encoder"]).to(device)
    else:
        G = Generator(latent_dim, channels_img).to(device)
        C = Critic(channels_img).to(device)
        E = Encoder(latent_dim, channels_img).to(device)

    noise_sampler = LatentDistribution(
        name=G.latent_distr_name, z_dim=latent_dim, device=device
    )

    print(C)
    print(G)
    print(E)

    # Loss function for WAE
    match criterion:
        case "mse":
            criterion = nn.MSELoss()
        case "l1":
            criterion = nn.L1Loss()
        case _:
            criterion = criterion

    # Optimizers
    G_optimizer = optim.Adam(
        G.parameters(),
        lr=learning_rate_G,
        betas=betas_wgan,
        weight_decay=weight_decay,
    )
    C_optimizer = optim.Adam(
        C.parameters(),
        lr=learning_rate_C,
        betas=betas_wgan,
        weight_decay=weight_decay,
    )
    # Optimizer
    E_optimizer = optim.Adam(
        E.parameters(),
        lr=learning_rate_E,
        betas=betas_wae,
        weight_decay=weight_decay,
    )

    # for tensorboard plotting
    fixed_noise = noise_sampler(32)
    # fixed_noise = G.sample_noise(32)
    summary_writer_dir = Path(summary_writer_dir)
    summary_writer_dir.mkdir(exist_ok=True, parents=True)
    writer_real = SummaryWriter(summary_writer_dir / "real")
    writer_fake = SummaryWriter(summary_writer_dir / "fake")
    writer_loss = SummaryWriter(summary_writer_dir / "loss")
    step = 0

    # Directory to save
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Schedulers
    G_scheduler = optim.lr_scheduler.MultiStepLR(
        G_optimizer, milestones=milestones, gamma=0.5
    )
    C_scheduler = optim.lr_scheduler.MultiStepLR(
        C_optimizer, milestones=milestones, gamma=0.5
    )
    E_scheduler = optim.lr_scheduler.MultiStepLR(
        E_optimizer, milestones=milestones, gamma=0.5
    )

    C.train()
    G.train()
    E.train()

    # Early stop
    current_patience = 0
    best_validation_loss = float("inf")

    real_val, _ = next(iter(val_data_loader))
    real_val = real_val.to(device)
    # fixed_noise = fixed_noise.to(device)
    _penalty_wgan_gp = math.sqrt(
        penalty_wgan_gp / penalty_wgan_lp
    )  # the penalty that use the leaky relu

    print(critic_iterations)
    print(noise_sampler)
    print(f"{save_dir = }")

    mov_avg_critic_val = None  # Moving average of the critic values
    time_window = 10  # Time window for the moving average
    smth_fact = 2 / (time_window + 1)  # Smoothing factor for the moving average
    print(f"{time_window = }; smoothing factor = {smth_fact:.6f}")

    for epoch in range(1, num_epochs + 1):
        # Free memory
        torch.cuda.empty_cache()

        epoch_wass_dist_WGAN = []
        epoch_wass_dist_WAE = []
        C_epoch_losses = []
        G_epoch_losses = []
        E_epoch_losses = []
        critic_real_values = []
        critic_fake_values = []
        critic_mean_values = []
        mov_avg_critic_vals = []
        gradient_norm_values = []

        if verbose:
            iterable = enumerate(
                tqdm(
                    data_loader,
                    desc=(
                        f"Epoch [{epoch}/{num_epochs}], " f"Patience {current_patience}"
                    ),
                )
            )
        else:
            iterable = enumerate(data_loader)

        # Training loop
        C_optimizer.zero_grad(set_to_none=True)
        G_optimizer.zero_grad(set_to_none=True)
        E_optimizer.zero_grad(set_to_none=True)

        for batch_idx, (real, _) in iterable:
            n = real.shape[0]
            real = real.to(device, non_blocking=True)

            n_critic_iterations = critic_iterations(epoch)
            for _ in range(n_critic_iterations):
                # Train Critic: min E[critic(fake)] - E[critic(real)] + penalization
                z = noise_sampler(n)
                fake = G(z)
                c_real = C(real).reshape(-1)
                c_fake = C(fake).reshape(-1)
                gradient, gradient_norm = gradient_penalty(
                    C, real, fake, device=device, penalty_gr=_penalty_wgan_gp
                )
                wasserstein_dist_WGAN = torch.mean(c_real) - torch.mean(c_fake)
                C_loss = -wasserstein_dist_WGAN + penalty_wgan_lp * gradient

                C_optimizer.zero_grad(set_to_none=True)
                C_loss.backward(retain_graph=True)
                C_optimizer.step()

            # Train Generator: min Wasserstein distance
            z = noise_sampler(n)
            G_loss = -torch.mean(C(G(z)))  # Generator loss

            # Update parameters of the generator
            G_optimizer.zero_grad(set_to_none=True)
            G_loss.backward()
            G_optimizer.step()

            # Train encoder: min |real - generator(encoder(real))| + penalization
            z = noise_sampler(n)
            z_tilde = E(real)  # Generate \tilde{z}_i from Q(Z|x_i) for i = 1:n
            x_recon = G(z_tilde)

            # According to the paper, this is the Wasserstein distance
            wasserstein_dist_WAE = criterion(real, x_recon)
            mmd_loss = imq_kernel(
                z, z_tilde, p_distr=G.latent_distr_name
            )  # Compute MMD
            E_loss = wasserstein_dist_WAE + penalty_wae * mmd_loss

            # Update parameters of the encoder
            E_optimizer.zero_grad(set_to_none=True)
            G_optimizer.zero_grad(set_to_none=True)
            E_loss.backward()
            E_optimizer.step()
            G_optimizer.step()

            # loss values
            # Register occasionally
            if batch_idx % report_every == 0:
                epoch_wass_dist_WGAN.append(wasserstein_dist_WGAN.data.item())
                epoch_wass_dist_WAE.append(wasserstein_dist_WAE.data.item())

                C_epoch_losses.append(C_loss.data.item())
                G_epoch_losses.append(G_loss.data.item())
                E_epoch_losses.append(E_loss.data.item())

                critic_real_value = c_real.mean().data.item()
                critic_fake_value = c_fake.mean().data.item()

                mean_critic_value = (critic_real_value + critic_fake_value) / 2
                mov_avg_critic_val = (
                    (
                        smth_fact * mean_critic_value
                        + (1 - smth_fact) * mov_avg_critic_val
                    )
                    if mov_avg_critic_val is not None
                    else mean_critic_value
                )

                critic_real_values.append(critic_real_value)
                critic_fake_values.append(critic_fake_value)
                critic_mean_values.append(mean_critic_value)
                mov_avg_critic_vals.append(mov_avg_critic_val)

                gradient_norm_mean = torch.mean(gradient_norm)
                gradient_norm_values.append(gradient_norm_mean.data.item())

                with torch.no_grad():
                    fake_WGAN = G(fixed_noise)
                    fake_WAE = G(E(real))
                    fake_val_WAE = G(E(real_val))

                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_real_val = torchvision.utils.make_grid(
                        real_val[:32], normalize=True
                    )
                    img_grid_fake_WGAN = torchvision.utils.make_grid(
                        fake_WGAN[:32], normalize=True
                    )
                    img_grid_fake_WAE = torchvision.utils.make_grid(
                        fake_WAE[:32], normalize=True
                    )
                    img_grid_fake_val_WAE = torchvision.utils.make_grid(
                        fake_val_WAE[:32], normalize=True
                    )

                    loss_C_name = "Loss Discriminator"
                    loss_G_name = "Loss Generator"
                    loss_E_name = "Loss Encoder"
                    critic_values_name = "Critic Values"
                    wass_dist_name_WGAN = "Wasserstein Distance WGAN"
                    wass_dist_name_WAE = "Wasserstein Distance WAE"
                    gradient_norm_name = "Gradient Norm"
                    writer_real.add_image("1 Real", img_grid_real, global_step=step)
                    writer_fake.add_image(
                        "2 Decoded", img_grid_fake_WAE, global_step=step
                    )
                    writer_real.add_image(
                        "3 Real Val", img_grid_real_val, global_step=step
                    )
                    writer_fake.add_image(
                        "4 Decoded Val", img_grid_fake_val_WAE, global_step=step
                    )
                    writer_fake.add_image(
                        "5 Generated", img_grid_fake_WGAN, global_step=step
                    )
                    writer_loss.add_scalars(
                        loss_C_name, {loss_C_name: C_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        loss_G_name, {loss_G_name: G_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        loss_E_name, {loss_E_name: E_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        wass_dist_name_WGAN,
                        {wass_dist_name_WGAN: wasserstein_dist_WGAN},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        wass_dist_name_WAE,
                        {wass_dist_name_WAE: wasserstein_dist_WAE},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        gradient_norm_name,
                        {gradient_norm_name: gradient_norm_mean},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        critic_values_name,
                        {
                            "Real": critic_real_value,
                            "Fake": critic_fake_value,
                            "Mean": mean_critic_value,
                            "Mov. Avg.": mov_avg_critic_val,
                        },
                        global_step=step,
                    )

                step += 1

        # Register the average loss
        critic_iterations.register_new_loss(mov_avg_critic_val)

        # Validation loop
        if verbose:
            iterable = tqdm(val_data_loader, desc=f"Val. Epoch [{epoch}/{num_epochs}]")
        else:
            iterable = val_data_loader

        epoch_wass_dist_WAE_val = []
        epoch_wass_dist_WGAN_val = []
        with torch.no_grad():
            for real, _ in iterable:
                real = real.to(device, non_blocking=True)
                n = real.shape[0]

                # WAE Validation Loss
                recon = G(E(real))
                loss = criterion(real, recon)
                epoch_wass_dist_WAE_val.append(loss.data.item())

                # WGAN Validation Loss
                z = noise_sampler(n)
                # z = G.sample_noise(n, type_as=real)
                fake = G(z)
                c_real, c_fake = C(real).reshape(-1), C(fake).reshape(-1)
                loss = torch.mean(c_real) - torch.mean(c_fake)
                epoch_wass_dist_WGAN_val.append(loss.data.item())

        # Calculate the average validation loss
        avg_wass_dist_WAE_val = torch.mean(
            torch.FloatTensor(epoch_wass_dist_WAE_val)
        ).item()
        avg_wass_dist_WGAN_val = torch.mean(
            torch.FloatTensor(epoch_wass_dist_WGAN_val)
        ).item()

        if avg_wass_dist_WAE_val < best_validation_loss - min_delta:
            best_validation_loss = avg_wass_dist_WAE_val
            current_patience = 0
            # Save models
            torch.save(deepcopy(G.state_dict()), save_dir / "generator.pt")
            torch.save(deepcopy(C.state_dict()), save_dir / "discriminator.pt")
            torch.save(deepcopy(E.state_dict()), save_dir / "encoder.pt")
        else:
            current_patience += 1
            # Save models
            torch.save(deepcopy(G.state_dict()), save_dir / "generator_overfitted.pt")
            torch.save(
                deepcopy(C.state_dict()),
                save_dir / "discriminator_overfitted.pt",
            )
            torch.save(deepcopy(E.state_dict()), save_dir / "encoder_overfitted.pt")

        global_epoch = step - 1
        avg_wass_dist_WGAN = torch.mean(torch.FloatTensor(epoch_wass_dist_WGAN)).item()
        avg_wass_dist_WAE = torch.mean(torch.FloatTensor(epoch_wass_dist_WAE)).item()
        C_avg_loss = torch.mean(torch.FloatTensor(C_epoch_losses)).item()
        G_avg_loss = torch.mean(torch.FloatTensor(G_epoch_losses)).item()
        E_avg_loss = torch.mean(torch.FloatTensor(E_epoch_losses)).item()
        critic_real_avg_loss = torch.mean(torch.FloatTensor(critic_real_values)).item()
        critic_fake_avg_loss = torch.mean(torch.FloatTensor(critic_fake_values)).item()
        critic_mean_avg_loss = torch.mean(torch.FloatTensor(critic_mean_values)).item()
        mov_avg_critic_val_avg = torch.mean(
            torch.FloatTensor(mov_avg_critic_vals)
        ).item()
        print(f"{critic_iterations = }")
        gradient_norm_values_avg = torch.mean(
            torch.FloatTensor(gradient_norm_values)
        ).item()
        writer_loss.add_scalars(
            wass_dist_name_WGAN,
            {"Avg. " + wass_dist_name_WGAN: avg_wass_dist_WGAN},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WAE,
            {"Avg. " + wass_dist_name_WAE: avg_wass_dist_WAE},
            global_epoch,
        )
        writer_loss.add_scalars(
            gradient_norm_name,
            {"Avg. " + gradient_norm_name: gradient_norm_values_avg},
            global_epoch,
        )
        writer_loss.add_scalars(
            loss_C_name, {"Avg. " + loss_C_name: C_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            loss_G_name, {"Avg. " + loss_G_name: G_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            loss_E_name, {"Avg. " + loss_E_name: E_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            critic_values_name,
            {
                "Avg. Real": critic_real_avg_loss,
                "Avg. Fake": critic_fake_avg_loss,
                "Avg. Mean": critic_mean_avg_loss,
                "Avg. Mov. Avg.": mov_avg_critic_val_avg,
            },
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WAE,
            {"Avg. " + wass_dist_name_WAE + " Validation": avg_wass_dist_WAE_val},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WGAN,
            {"Avg. " + wass_dist_name_WGAN + " Validation": avg_wass_dist_WGAN_val},
            global_epoch,
        )

        if current_patience >= patience:
            print("Early stopping triggered. Training halted.")
            break

        # Decrease learning-rate
        G_scheduler.step()
        C_scheduler.step()
        E_scheduler.step()


# MARK: Train DWAE Optimized
def train_dwae_optimized(
    latent_dim,  # =100
    nn_kwargs=None,
    channels_img=3,
    learning_rate_E=1e-3,
    learning_rate_G=1e-4,
    batch_size=128,
    image_size: int | tuple[int, int] = 64,
    num_epochs=100,
    penalty_wae=1 / 5,
    criterion: Literal["mse", "l1"] = "l1",
    betas_wgan=(0.5, 0.9),
    betas_wae=(0.5, 0.999),
    weight_decay=1e-5,
    milestones=(25, 50, 75),
    patience=20,  # Number of epochs to wait for improvement
    min_delta=0,  # Minimum change in validation loss to be considered as an improvement
    device=None,
    transform=None,
    file_path: str | Path = None,
    ds_type="quickdraw",
    save_dir: str | Path = "networks/",
    summary_writer_dir: str | Path = "logs",
    verbose=True,
    report_every=100,
    double_recon=False,
    denoise=True,
):
    device = get_device(device)

    # Transforms
    transform = transform or T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    sigma = 3
    k_size = 2 * int(3 * sigma) + 1
    gaussian_blur = T.GaussianBlur(k_size, (0.1, sigma))

    # Dataset and Dataloader
    dataset, val_dataset = get_dataset(
        ds_type=ds_type, file_path=file_path, transform=transform
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if val_dataset:
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    # Models
    if nn_kwargs:
        G = Generator(latent_dim, channels_img, **nn_kwargs["generator"]).to(device)
        E = Encoder(latent_dim, channels_img, **nn_kwargs["encoder"]).to(device)
    else:
        G = Generator(latent_dim, channels_img).to(device)
        E = Encoder(latent_dim, channels_img).to(device)

    noise_sampler = LatentDistribution(
        name=G.latent_distr_name, z_dim=latent_dim, device=device
    )

    print(G)
    print(E)

    # Loss function for WAE
    match criterion:
        case "mse":
            criterion = nn.MSELoss()
        case "l1":
            criterion = nn.L1Loss()
        case _:
            criterion = criterion

    # Optimizers
    G_optimizer = optim.Adam(
        G.parameters(),
        lr=learning_rate_G,
        betas=betas_wgan,
        weight_decay=weight_decay,
    )
    # Optimizer
    E_optimizer = optim.Adam(
        E.parameters(),
        lr=learning_rate_E,
        betas=betas_wae,
        weight_decay=weight_decay,
    )

    # for tensorboard plotting
    fixed_noise = noise_sampler(32)
    summary_writer_dir = Path(summary_writer_dir)
    summary_writer_dir.mkdir(exist_ok=True, parents=True)
    writer_real = SummaryWriter(str(summary_writer_dir / "real"))
    writer_fake = SummaryWriter(str(summary_writer_dir / "fake"))
    writer_loss = SummaryWriter(str(summary_writer_dir / "loss"))
    step = 0

    # Directory to save
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Schedulers
    G_scheduler = optim.lr_scheduler.MultiStepLR(
        G_optimizer, milestones=milestones, gamma=0.5
    )
    E_scheduler = optim.lr_scheduler.MultiStepLR(
        E_optimizer, milestones=milestones, gamma=0.5
    )

    G.train()
    E.train()

    # Early stop
    current_patience = 0
    best_validation_loss = float("inf")

    real_val, _ = next(iter(val_data_loader))
    real_val = real_val.to(device)

    print(noise_sampler)
    print(f"{save_dir = }")

    for epoch in range(1, num_epochs + 1):
        # Free memory
        torch.cuda.empty_cache()

        epoch_wass_dist_WAE = []
        E_epoch_losses = []
        penalization_epoch = []

        if verbose:
            iterable = enumerate(
                tqdm(
                    data_loader,
                    desc=(
                        f"Epoch [{epoch}/{num_epochs}], " f"Patience {current_patience}"
                    ),
                )
            )
        else:
            iterable = enumerate(data_loader)

        # Training loop
        G_optimizer.zero_grad(set_to_none=True)
        E_optimizer.zero_grad(set_to_none=True)

        for batch_idx, (real, _) in iterable:
            n: int = real.shape[0]
            real_ = real = real.to(device, non_blocking=True)

            # Train encoder: min |real - generator(encoder(real))| + penalization
            z = noise_sampler(n)
            if denoise:
                real_ = gaussian_blur(real)
            z_tilde = E(real_)  # Generate \tilde{z}_i from Q(Z|x_i) for i = 1:n
            x_recon = G(z_tilde)
            if double_recon:
                z_recon = E(x_recon)  
                x_recon_recon = G(z_recon)
                


            # According to the paper, this is the Wasserstein distance
            wasserstein_dist_WAE = criterion(real, x_recon)
            z_tilde = z_tilde.squeeze()
            m, C = torch.mean(z_tilde, dim=0), torch.cov(z_tilde.T)
            penalization = bures_wass_dist(m, C)
            # mmd_loss = imq_kernel(
            #     z, z_tilde, p_distr=G.latent_distr_name
            # )  # Compute MMD
            if double_recon:
                recon_loss = criterion(real, x_recon_recon)
                loss_weight = max(0.3, 1 - 0.07 * epoch)
                E_loss = (
                    loss_weight * wasserstein_dist_WAE
                    + (1 - loss_weight) * recon_loss
                    + penalty_wae * penalization
                )
            else:
                E_loss = wasserstein_dist_WAE + penalty_wae * penalization

            # Update parameters of the encoder
            E_optimizer.zero_grad(set_to_none=True)
            G_optimizer.zero_grad(set_to_none=True)
            E_loss.backward()
            E_optimizer.step()
            G_optimizer.step()

            # loss values
            # Register occasionally
            if batch_idx % report_every == 0:
                epoch_wass_dist_WAE.append(wasserstein_dist_WAE.data.item())
                E_epoch_losses.append(E_loss.data.item())
                penalization_epoch.append(penalization.data.item())

                with torch.no_grad():
                    corrupted_real_val = gaussian_blur(real_val)
                    fake_WGAN = G(fixed_noise)
                    fake_WAE = G(E(real))
                    denoised_real = G(E(real_))
                    fake_val_WAE = G(E(real_val))
                    denoised_real_val = G(E(corrupted_real_val))

                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_real_ = torchvision.utils.make_grid(
                        real_[:32], normalize=True
                    )
                    img_grid_real_val = torchvision.utils.make_grid(
                        real_val[:32], normalize=True
                    )
                    img_grid_real_val_blur = torchvision.utils.make_grid(
                        corrupted_real_val[:32], normalize=True
                    )
                    img_grid_fake_WGAN = torchvision.utils.make_grid(
                        fake_WGAN[:32], normalize=True
                    )
                    img_grid_fake_WAE = torchvision.utils.make_grid(
                        fake_WAE[:32], normalize=True
                    )
                    img_grid_den_real = torchvision.utils.make_grid(
                        denoised_real[:32], normalize=True
                    )
                    img_grid_fake_val_WAE = torchvision.utils.make_grid(
                        fake_val_WAE[:32], normalize=True
                    )
                    img_grid_den_real_val = torchvision.utils.make_grid(
                        denoised_real_val[:32], normalize=True
                    )

                    writer_real.add_image(
                        "1 Real", img_grid_real, global_step=step
                    )  # fmt: skip
                    writer_fake.add_image(
                        "2 Decoded", img_grid_fake_WAE, global_step=step
                    )
                    writer_real.add_image(
                        "3 Real Blurred", img_grid_real_, global_step=step
                    )
                    writer_fake.add_image(
                        "4 Denoise", img_grid_den_real, global_step=step
                    )
                    writer_real.add_image(
                        "5 Real Val", img_grid_real_val, global_step=step
                    )
                    writer_fake.add_image(
                        "6 Decoded Val", img_grid_fake_val_WAE, global_step=step
                    )
                    writer_real.add_image(
                        "7 Real Val Blurred", img_grid_real_val_blur, global_step=step
                    )
                    writer_fake.add_image(
                        "8 Denoise Val", img_grid_den_real_val, global_step=step
                    )
                    writer_fake.add_image(
                        "9 Generated", img_grid_fake_WGAN, global_step=step
                    )

                    loss_E_name = "Loss Encoder"
                    wass_dist_name_WAE = "Wasserstein Distance WAE"
                    penalization_name = "Penalization"

                    writer_loss.add_scalars(
                        loss_E_name,
                        {loss_E_name: E_loss},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        wass_dist_name_WAE,
                        {wass_dist_name_WAE: wasserstein_dist_WAE},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        penalization_name,
                        {penalization_name: penalization},
                        global_step=step,
                    )

                step += 1

        # Validation loop
        if verbose:
            iterable = tqdm(val_data_loader, desc=f"Val. Epoch [{epoch}/{num_epochs}]")
        else:
            iterable = val_data_loader

        epoch_wass_dist_WAE_val = []
        with torch.no_grad():
            for real, _ in iterable:
                real = real.to(device, non_blocking=True)

                # WAE Validation Loss
                recon = G(E(real))
                loss = criterion(real, recon)
                epoch_wass_dist_WAE_val.append(loss.data.item())

        # Calculate the average validation loss
        avg_wass_dist_WAE_val = torch.mean(
            torch.FloatTensor(epoch_wass_dist_WAE_val)
        ).item()

        if avg_wass_dist_WAE_val < best_validation_loss - min_delta:
            best_validation_loss = avg_wass_dist_WAE_val
            current_patience = 0
            # Save models
            torch.save(deepcopy(G.state_dict()), save_dir / "generator.pt")
            torch.save(deepcopy(E.state_dict()), save_dir / "encoder.pt")
        else:
            current_patience += 1
            # Save models
            torch.save(deepcopy(G.state_dict()), save_dir / "generator_overfitted.pt")
            torch.save(deepcopy(E.state_dict()), save_dir / "encoder_overfitted.pt")

        global_epoch = step - 1
        avg_wass_dist_WAE = torch.mean(torch.FloatTensor(epoch_wass_dist_WAE)).item()
        E_avg_loss = torch.mean(torch.FloatTensor(E_epoch_losses)).item()
        penalization_avg = torch.mean(torch.FloatTensor(penalization_epoch)).item()

        writer_loss.add_scalars(
            wass_dist_name_WAE,
            {"Avg. " + wass_dist_name_WAE: avg_wass_dist_WAE},
            global_epoch,
        )
        writer_loss.add_scalars(
            loss_E_name,
            {"Avg. " + loss_E_name: E_avg_loss},
            global_epoch,
        )
        writer_loss.add_scalars(
            penalization_name,
            {"Avg. " + penalization_name: penalization_avg},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WAE,
            {"Avg. " + wass_dist_name_WAE + " Validation": avg_wass_dist_WAE_val},
            global_epoch,
        )

        if current_patience >= patience:
            print("Early stopping triggered. Training halted.")
            break

        # Decrease learning-rate
        G_scheduler.step()
        E_scheduler.step()


# MARK: DWAE WGAN
def train_wgan_and_dwae_optimized(
    latent_dim,  # =100
    nn_kwargs=None,
    channels_img=3,
    learning_rate_E=1e-3,
    learning_rate_G=1e-4,
    learning_rate_C=1e-4,
    batch_size=128,
    image_size: int | tuple[int, int] = 64,
    num_epochs=100,
    critic_iterations=5,
    crit_iter_patience=None,
    penalty_wgan_lp=10,
    penalty_wgan_gp=0.1,
    penalty_wae=10,
    criterion: Literal["mse", "l1"] = "l1",
    betas_wgan=(0.5, 0.9),
    betas_wae=(0.5, 0.999),
    weight_decay=1e-5,
    milestones=(25, 50, 75),
    patience=20,  # Number of epochs to wait for improvement
    min_delta=0,  # Minimum change in validation loss to be considered as an improvement
    device=None,
    transform=None,
    file_path: str | Path = None,
    ds_type="quickdraw",
    save_dir: str | Path = "networks/",
    summary_writer_dir: str | Path = "logs",
    verbose=True,
    report_every=100,
):
    device = get_device(device)

    critic_iterations = CriticIterations(critic_iterations, patience=crit_iter_patience)

    # Transforms
    transform = transform or T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    sigma = 3
    k_size = 2 * int(3 * sigma) + 1
    gaussian_blur = T.GaussianBlur(k_size, (0.1, sigma))

    # Dataset and Dataloader
    dataset, val_dataset = get_dataset(
        ds_type=ds_type, file_path=file_path, transform=transform
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if val_dataset:
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    # Models
    if nn_kwargs:
        G = Generator(latent_dim, channels_img, **nn_kwargs["generator"]).to(device)
        C = Critic(channels_img, **nn_kwargs["critic"]).to(device)
        E = Encoder(latent_dim, channels_img, **nn_kwargs["encoder"]).to(device)
    else:
        G = Generator(latent_dim, channels_img).to(device)
        C = Critic(channels_img).to(device)
        E = Encoder(latent_dim, channels_img).to(device)

    noise_sampler = LatentDistribution(
        name=G.latent_distr_name, z_dim=latent_dim, device=device
    )

    print(C)
    print(G)
    print(E)

    # Loss function for WAE
    match criterion:
        case "mse":
            criterion = nn.MSELoss()
        case "l1":
            criterion = nn.L1Loss()
        case _:
            criterion = criterion

    # Optimizers
    G_optimizer = optim.Adam(
        G.parameters(),
        lr=learning_rate_G,
        betas=betas_wgan,
        weight_decay=weight_decay,
    )
    C_optimizer = optim.Adam(
        C.parameters(),
        lr=learning_rate_C,
        betas=betas_wgan,
        weight_decay=weight_decay,
    )
    # Optimizer
    E_optimizer = optim.Adam(
        E.parameters(),
        lr=learning_rate_E,
        betas=betas_wae,
        weight_decay=weight_decay,
    )

    # for tensorboard plotting
    fixed_noise = noise_sampler(32)
    summary_writer_dir = Path(summary_writer_dir)
    summary_writer_dir.mkdir(exist_ok=True, parents=True)
    writer_real = SummaryWriter(summary_writer_dir / "real")
    writer_fake = SummaryWriter(summary_writer_dir / "fake")
    writer_loss = SummaryWriter(summary_writer_dir / "loss")
    step = 0

    # Directory to save
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Schedulers
    G_scheduler = optim.lr_scheduler.MultiStepLR(
        G_optimizer, milestones=milestones, gamma=0.5
    )
    C_scheduler = optim.lr_scheduler.MultiStepLR(
        C_optimizer, milestones=milestones, gamma=0.5
    )
    E_scheduler = optim.lr_scheduler.MultiStepLR(
        E_optimizer, milestones=milestones, gamma=0.5
    )

    C.train()
    G.train()
    E.train()

    # Early stop
    current_patience = 0
    best_validation_loss = float("inf")

    real_val, _ = next(iter(val_data_loader))
    real_val = real_val.to(device)
    _penalty_wgan_gp = math.sqrt(
        penalty_wgan_gp / penalty_wgan_lp
    )  # the penalty that use the leaky relu

    print(critic_iterations)
    print(noise_sampler)
    print(f"{save_dir = }")

    mov_avg_critic_val = None  # Moving average of the critic values
    time_window = 10  # Time window for the moving average
    smth_fact = 2 / (time_window + 1)  # Smoothing factor for the moving average
    print(f"{time_window = }; smoothing factor = {smth_fact:.6f}")

    for epoch in range(1, num_epochs + 1):
        # Free memory
        torch.cuda.empty_cache()

        epoch_wass_dist_WGAN = []
        epoch_wass_dist_WAE = []
        C_epoch_losses = []
        G_epoch_losses = []
        E_epoch_losses = []
        critic_real_values = []
        critic_fake_values = []
        critic_mean_values = []
        penalization_epoch = []
        mov_avg_critic_vals = []
        gradient_norm_values = []

        if verbose:
            iterable = enumerate(
                tqdm(
                    data_loader,
                    desc=(f"Epoch [{epoch}/{num_epochs}], Patience {current_patience}"),
                )
            )
        else:
            iterable = enumerate(data_loader)

        # Training loop
        C_optimizer.zero_grad(set_to_none=True)
        G_optimizer.zero_grad(set_to_none=True)
        E_optimizer.zero_grad(set_to_none=True)

        for batch_idx, (real, _) in iterable:
            n = real.shape[0]
            real = real.to(device, non_blocking=True)

            n_critic_iterations = critic_iterations(epoch)
            for _ in range(n_critic_iterations):
                # Train Critic: min E[critic(fake)] - E[critic(real)] + penalization
                z = noise_sampler(n)
                fake = G(z)
                c_real = C(real).reshape(-1)
                c_fake = C(fake).reshape(-1)
                gradient, gradient_norm = gradient_penalty(
                    C, real, fake, device=device, penalty_gr=_penalty_wgan_gp
                )
                wasserstein_dist_WGAN = torch.mean(c_real) - torch.mean(c_fake)
                C_loss = -wasserstein_dist_WGAN + penalty_wgan_lp * gradient

                C_optimizer.zero_grad(set_to_none=True)
                C_loss.backward(retain_graph=True)
                C_optimizer.step()

            # Train Generator: min Wasserstein distance
            z = noise_sampler(n)
            G_loss = -torch.mean(C(G(z)))  # Generator loss

            # Update parameters of the generator
            G_optimizer.zero_grad(set_to_none=True)
            G_loss.backward()
            G_optimizer.step()

            # Train encoder: min |real - generator(encoder(real))| + penalization
            z = noise_sampler(n)
            real_ = gaussian_blur(real)
            z_tilde = E(real_)  # Generate \tilde{z}_i from Q(Z|x_i) for i = 1:n
            x_recon = G(z_tilde)

            # According to the paper, this is the Wasserstein distance
            wasserstein_dist_WAE = criterion(real, x_recon)
            z_tilde = z_tilde.squeeze()
            m, cov = torch.mean(z_tilde, dim=0), torch.cov(z_tilde.T)
            penalization = bures_wass_dist(m, cov)
            E_loss = wasserstein_dist_WAE + penalty_wae * penalization

            # Update parameters of the encoder
            E_optimizer.zero_grad(set_to_none=True)
            G_optimizer.zero_grad(set_to_none=True)
            E_loss.backward()
            E_optimizer.step()
            G_optimizer.step()

            # loss values
            # Register occasionally
            if batch_idx % report_every == 0:
                epoch_wass_dist_WGAN.append(wasserstein_dist_WGAN.data.item())
                epoch_wass_dist_WAE.append(wasserstein_dist_WAE.data.item())

                C_epoch_losses.append(C_loss.data.item())
                G_epoch_losses.append(G_loss.data.item())
                E_epoch_losses.append(E_loss.data.item())

                critic_real_value = c_real.mean().data.item()
                critic_fake_value = c_fake.mean().data.item()

                mean_critic_value = (critic_real_value + critic_fake_value) / 2
                mov_avg_critic_val = (
                    (
                        smth_fact * mean_critic_value
                        + (1 - smth_fact) * mov_avg_critic_val
                    )
                    if mov_avg_critic_val is not None
                    else mean_critic_value
                )

                critic_real_values.append(critic_real_value)
                critic_fake_values.append(critic_fake_value)
                critic_mean_values.append(mean_critic_value)
                mov_avg_critic_vals.append(mov_avg_critic_val)

                penalization_epoch.append(penalization.data.item())

                gradient_norm_mean = torch.mean(gradient_norm)
                gradient_norm_values.append(gradient_norm_mean.data.item())

                with torch.no_grad():
                    corrupted_real_val = gaussian_blur(real_val)
                    fake_WGAN = G(fixed_noise)
                    fake_WAE = G(E(real))
                    denoised_real = G(E(real_))
                    fake_val_WAE = G(E(real_val))
                    denoised_real_val = G(E(corrupted_real_val))

                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_real_ = torchvision.utils.make_grid(
                        real_[:32], normalize=True
                    )
                    img_grid_real_val = torchvision.utils.make_grid(
                        real_val[:32], normalize=True
                    )
                    img_grid_real_val_blur = torchvision.utils.make_grid(
                        corrupted_real_val[:32], normalize=True
                    )
                    img_grid_fake_WGAN = torchvision.utils.make_grid(
                        fake_WGAN[:32], normalize=True
                    )
                    img_grid_fake_WAE = torchvision.utils.make_grid(
                        fake_WAE[:32], normalize=True
                    )
                    img_grid_den_real = torchvision.utils.make_grid(
                        denoised_real[:32], normalize=True
                    )
                    img_grid_fake_val_WAE = torchvision.utils.make_grid(
                        fake_val_WAE[:32], normalize=True
                    )
                    img_grid_den_real_val = torchvision.utils.make_grid(
                        denoised_real_val[:32], normalize=True
                    )

                    loss_C_name = "Loss Discriminator"
                    loss_G_name = "Loss Generator"
                    loss_E_name = "Loss Encoder"
                    critic_values_name = "Critic Values"
                    wass_dist_name_WGAN = "Wasserstein Distance WGAN"
                    wass_dist_name_WAE = "Wasserstein Distance WAE"
                    gradient_norm_name = "Gradient Norm"
                    penalization_name = "Penalization"

                    writer_real.add_image(
                        "1 Real", img_grid_real, global_step=step
                    )  # fmt: skip
                    writer_fake.add_image(
                        "2 Decoded", img_grid_fake_WAE, global_step=step
                    )
                    writer_real.add_image(
                        "3 Real Blurred", img_grid_real_, global_step=step
                    )
                    writer_fake.add_image(
                        "4 Denoise", img_grid_den_real, global_step=step
                    )
                    writer_real.add_image(
                        "5 Real Val", img_grid_real_val, global_step=step
                    )
                    writer_fake.add_image(
                        "6 Decoded Val", img_grid_fake_val_WAE, global_step=step
                    )
                    writer_real.add_image(
                        "7 Real Val Blurred", img_grid_real_val_blur, global_step=step
                    )
                    writer_fake.add_image(
                        "8 Denoise Val", img_grid_den_real_val, global_step=step
                    )
                    writer_fake.add_image(
                        "9 Generated", img_grid_fake_WGAN, global_step=step
                    )

                    writer_loss.add_scalars(
                        loss_C_name, {loss_C_name: C_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        loss_G_name, {loss_G_name: G_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        loss_E_name, {loss_E_name: E_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        wass_dist_name_WGAN,
                        {wass_dist_name_WGAN: wasserstein_dist_WGAN},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        wass_dist_name_WAE,
                        {wass_dist_name_WAE: wasserstein_dist_WAE},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        gradient_norm_name,
                        {gradient_norm_name: gradient_norm_mean},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        penalization_name,
                        {penalization_name: penalization},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        critic_values_name,
                        {
                            "Real": critic_real_value,
                            "Fake": critic_fake_value,
                            "Mean": mean_critic_value,
                            "Mov. Avg.": mov_avg_critic_val,
                        },
                        global_step=step,
                    )

                step += 1

        # Register the average loss
        critic_iterations.register_new_loss(mov_avg_critic_val)

        # Validation loop
        if verbose:
            iterable = tqdm(val_data_loader, desc=f"Val. Epoch [{epoch}/{num_epochs}]")
        else:
            iterable = val_data_loader

        epoch_wass_dist_WAE_val = []
        epoch_wass_dist_WGAN_val = []
        with torch.no_grad():
            for real, _ in iterable:
                real = real.to(device, non_blocking=True)
                n = real.shape[0]

                # WAE Validation Loss
                recon = G(E(real))
                loss = criterion(real, recon)
                epoch_wass_dist_WAE_val.append(loss.data.item())

                # WGAN Validation Loss
                z = noise_sampler(n)
                fake = G(z)
                c_real, c_fake = C(real).reshape(-1), C(fake).reshape(-1)
                loss = torch.mean(c_real) - torch.mean(c_fake)
                epoch_wass_dist_WGAN_val.append(loss.data.item())

        # Calculate the average validation loss
        avg_wass_dist_WAE_val = torch.mean(
            torch.FloatTensor(epoch_wass_dist_WAE_val)
        ).item()
        avg_wass_dist_WGAN_val = torch.mean(
            torch.FloatTensor(epoch_wass_dist_WGAN_val)
        ).item()

        if avg_wass_dist_WAE_val < best_validation_loss - min_delta:
            best_validation_loss = avg_wass_dist_WAE_val
            current_patience = 0
            # Save models
            torch.save(deepcopy(G.state_dict()), save_dir / "generator.pt")
            torch.save(deepcopy(C.state_dict()), save_dir / "discriminator.pt")
            torch.save(deepcopy(E.state_dict()), save_dir / "encoder.pt")
        else:
            current_patience += 1
            # Save models
            torch.save(
                deepcopy(G.state_dict()),
                save_dir / "generator_overfitted.pt",
            )  # fmt: skip
            torch.save(
                deepcopy(C.state_dict()),
                save_dir / "discriminator_overfitted.pt",
            )
            torch.save(
                deepcopy(E.state_dict()), 
                save_dir / "encoder_overfitted.pt",
            )  # fmt: skip

        global_epoch = step - 1
        avg_wass_dist_WGAN = torch.mean(torch.FloatTensor(epoch_wass_dist_WGAN)).item()
        avg_wass_dist_WAE = torch.mean(torch.FloatTensor(epoch_wass_dist_WAE)).item()
        C_avg_loss = torch.mean(torch.FloatTensor(C_epoch_losses)).item()
        G_avg_loss = torch.mean(torch.FloatTensor(G_epoch_losses)).item()
        E_avg_loss = torch.mean(torch.FloatTensor(E_epoch_losses)).item()
        critic_real_avg_loss = torch.mean(torch.FloatTensor(critic_real_values)).item()
        critic_fake_avg_loss = torch.mean(torch.FloatTensor(critic_fake_values)).item()
        critic_mean_avg_loss = torch.mean(torch.FloatTensor(critic_mean_values)).item()
        penalization_avg = torch.mean(torch.FloatTensor(penalization_epoch)).item()
        mov_avg_critic_val_avg = torch.mean(
            torch.FloatTensor(mov_avg_critic_vals)
        ).item()
        print(f"{critic_iterations = }")
        gradient_norm_values_avg = torch.mean(
            torch.FloatTensor(gradient_norm_values)
        ).item()

        writer_loss.add_scalars(
            wass_dist_name_WGAN,
            {"Avg. " + wass_dist_name_WGAN: avg_wass_dist_WGAN},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WAE,
            {"Avg. " + wass_dist_name_WAE: avg_wass_dist_WAE},
            global_epoch,
        )
        writer_loss.add_scalars(
            gradient_norm_name,
            {"Avg. " + gradient_norm_name: gradient_norm_values_avg},
            global_epoch,
        )
        writer_loss.add_scalars(
            loss_C_name, {"Avg. " + loss_C_name: C_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            loss_G_name, {"Avg. " + loss_G_name: G_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            loss_E_name, {"Avg. " + loss_E_name: E_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            critic_values_name,
            {
                "Avg. Real": critic_real_avg_loss,
                "Avg. Fake": critic_fake_avg_loss,
                "Avg. Mean": critic_mean_avg_loss,
                "Avg. Mov. Avg.": mov_avg_critic_val_avg,
            },
            global_epoch,
        )
        writer_loss.add_scalars(
            penalization_name,
            {"Avg. " + penalization_name: penalization_avg},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WAE,
            {"Avg. " + wass_dist_name_WAE + " Validation": avg_wass_dist_WAE_val},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WGAN,
            {"Avg. " + wass_dist_name_WGAN + " Validation": avg_wass_dist_WGAN_val},
            global_epoch,
        )

        if current_patience >= patience:
            print("Early stopping triggered. Training halted.")
            break

        # Decrease learning-rate
        G_scheduler.step()
        C_scheduler.step()
        E_scheduler.step()


# MARK: DWAE WGAN Double Reconstruction
def train_wgan_and_dwae_double_recon_optimized(
    latent_dim,  # =100
    nn_kwargs=None,
    channels_img=3,
    learning_rate_E=1e-3,
    learning_rate_G=1e-4,
    learning_rate_C=1e-4,
    batch_size=128,
    image_size: int | tuple[int, int] = 64,
    num_epochs=100,
    critic_iterations=5,
    crit_iter_patience=None,
    penalty_wgan_lp=10,
    penalty_wgan_gp=0.1,
    penalty_wae=10,
    criterion: Literal["mse", "l1"] = "l1",
    betas_wgan=(0.5, 0.9),
    betas_wae=(0.5, 0.999),
    weight_decay=1e-5,
    milestones=(25, 50, 75),
    patience=20,  # Number of epochs to wait for improvement
    min_delta=0,  # Minimum change in validation loss to be considered as an improvement
    device=None,
    transform=None,
    file_path: str | Path = None,
    ds_type="quickdraw",
    save_dir: str | Path = "networks/",
    summary_writer_dir: str | Path = "logs",
    verbose=True,
    report_every=100,
    denoising=True,
    double_recon=True,
    strategy_double_recon: t.Literal["from_denoising", "from_real", "proj_vs_proj2", "from_gen"] = "from_denoising",
):
    device = get_device(device)

    critic_iterations = CriticIterations(critic_iterations, patience=crit_iter_patience)

    # Transforms
    transform = transform or T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    sigma = 3
    k_size = 2 * int(3 * sigma) + 1
    gaussian_blur = T.GaussianBlur(k_size, (0.1, sigma))

    # Dataset and Dataloader
    dataset, val_dataset = get_dataset(
        ds_type=ds_type, file_path=file_path, transform=transform
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if val_dataset:
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    # Models
    if nn_kwargs:
        G = Generator(latent_dim, channels_img, **nn_kwargs["generator"]).to(device)
        C = Critic(channels_img, **nn_kwargs["critic"]).to(device)
        E = Encoder(latent_dim, channels_img, **nn_kwargs["encoder"]).to(device)
    else:
        G = Generator(latent_dim, channels_img).to(device)
        C = Critic(channels_img).to(device)
        E = Encoder(latent_dim, channels_img).to(device)

    noise_sampler = LatentDistribution(
        name=G.latent_distr_name, z_dim=latent_dim, device=device
    )

    print(C)
    print(G)
    print(E)

    # Loss function for WAE
    match criterion:
        case "mse":
            criterion = nn.MSELoss()
        case "l1":
            criterion = nn.L1Loss()
        case _:
            criterion = criterion

    # Optimizers
    G_optimizer = optim.Adam(
        G.parameters(),
        lr=learning_rate_G,
        betas=betas_wgan,
        weight_decay=weight_decay,
    )
    C_optimizer = optim.Adam(
        C.parameters(),
        lr=learning_rate_C,
        betas=betas_wgan,
        weight_decay=weight_decay,
    )
    # Optimizer
    E_optimizer = optim.Adam(
        E.parameters(),
        lr=learning_rate_E,
        betas=betas_wae,
        weight_decay=weight_decay,
    )

    # for tensorboard plotting
    fixed_noise = noise_sampler(32)
    summary_writer_dir = Path(summary_writer_dir)
    summary_writer_dir.mkdir(exist_ok=True, parents=True)
    writer_real = SummaryWriter(summary_writer_dir / "real")
    writer_fake = SummaryWriter(summary_writer_dir / "fake")
    writer_loss = SummaryWriter(summary_writer_dir / "loss")
    step = 0

    # Directory to save
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Schedulers
    G_scheduler = optim.lr_scheduler.MultiStepLR(
        G_optimizer, milestones=milestones, gamma=0.5
    )
    C_scheduler = optim.lr_scheduler.MultiStepLR(
        C_optimizer, milestones=milestones, gamma=0.5
    )
    E_scheduler = optim.lr_scheduler.MultiStepLR(
        E_optimizer, milestones=milestones, gamma=0.5
    )

    C.train()
    G.train()
    E.train()

    # Early stop
    current_patience = 0
    best_validation_loss = float("inf")

    real_val, _ = next(iter(val_data_loader))
    real_val = real_val.to(device)
    _penalty_wgan_gp = math.sqrt(
        penalty_wgan_gp / penalty_wgan_lp
    )  # the penalty that use the leaky relu

    print(critic_iterations)
    print(noise_sampler)
    print(f"{save_dir = }")

    mov_avg_critic_val = None  # Moving average of the critic values
    time_window = 10  # Time window for the moving average
    smth_fact = 2 / (time_window + 1)  # Smoothing factor for the moving average
    print(f"{time_window = }; smoothing factor = {smth_fact:.6f}")

    for epoch in range(1, num_epochs + 1):
        # Free memory
        torch.cuda.empty_cache()

        epoch_wass_dist_WGAN = []
        epoch_wass_dist_WAE = []
        C_epoch_losses = []
        G_epoch_losses = []
        E_epoch_losses = []
        critic_real_values = []
        critic_fake_values = []
        critic_mean_values = []
        penalization_epoch = []
        mov_avg_critic_vals = []
        gradient_norm_values = []
        if double_recon:
            double_recon_loss_values = []

        if verbose:
            iterable = enumerate(
                tqdm(
                    data_loader,
                    desc=(f"Epoch [{epoch}/{num_epochs}], Patience {current_patience}"),
                )
            )
        else:
            iterable = enumerate(data_loader)

        # Training loop
        C_optimizer.zero_grad(set_to_none=True)
        G_optimizer.zero_grad(set_to_none=True)
        E_optimizer.zero_grad(set_to_none=True)

        for batch_idx, (real, _) in iterable:
            n = real.shape[0]
            real_ = real = real.to(device, non_blocking=True)

            n_critic_iterations = critic_iterations(epoch)
            for _ in range(n_critic_iterations):
                # Train Critic: min E[critic(fake)] - E[critic(real)] + penalization
                z = noise_sampler(n)
                fake = G(z)
                c_real = C(real).reshape(-1)
                c_fake = C(fake).reshape(-1)
                gradient, gradient_norm = gradient_penalty(
                    C, real, fake, device=device, penalty_gr=_penalty_wgan_gp
                )
                wasserstein_dist_WGAN = torch.mean(c_real) - torch.mean(c_fake)
                C_loss = -wasserstein_dist_WGAN + penalty_wgan_lp * gradient

                C_optimizer.zero_grad(set_to_none=True)
                C_loss.backward(retain_graph=True)
                C_optimizer.step()

            # Train Generator: min Wasserstein distance
            z = noise_sampler(n)
            G_loss = -torch.mean(C(G(z)))  # Generator loss

            # Update parameters of the generator
            G_optimizer.zero_grad(set_to_none=True)
            G_loss.backward()
            G_optimizer.step()

            # Train encoder: min |real - generator(encoder(real))| + penalization
            z = noise_sampler(n)
            if denoising:
                real_ = gaussian_blur(real)

            z_tilde = E(real_)  # Generate \tilde{z}_i from Q(Z|x_i) for i = 1:n
            x_recon = G(z_tilde)

            # Double reconstruction
            if double_recon:
                real_recon = real
                match strategy_double_recon:
                    case "from_denoising":
                        z_recon = E(x_recon)
                        x_recon_recon = G(z_recon)
                    case "from_real":
                        z_tilde_ = E(real)
                        x_recon_ = G(z_tilde_)
                        z_recon = E(x_recon_)
                        x_recon_recon = G(z_recon)
                    case "proj_vs_proj2":
                        real_recon = x_recon
                        z_recon = E(x_recon)
                        x_recon_recon = G(z_recon)
                    case "from_gen":
                        z = noise_sampler(n)
                        real_recon = G(z)
                        z_recon = E(real_recon)
                        x_recon_recon = G(z_recon)


            # According to the paper, this is the Wasserstein distance
            wasserstein_dist_WAE = criterion(real, x_recon)

            z_tilde = z_tilde.squeeze()
            m, cov = torch.mean(z_tilde, dim=0), torch.cov(z_tilde.T)
            penalization = bures_wass_dist(m, cov)

            if double_recon:
                double_recon_loss = criterion(real_recon, x_recon_recon)
                # The loss should be 1 at epoch 0 and 0.5 at epoch 10, then must be constant
                loss_weight = (0.5 - 1.0) / (10 - 0) * epoch + 1.0
                loss_weight = max(0.5, loss_weight)
                E_loss = (
                    loss_weight * wasserstein_dist_WAE
                    + (1 - loss_weight) * double_recon_loss
                    + penalty_wae * penalization
                )
            else:
                E_loss = wasserstein_dist_WAE + penalty_wae * penalization

            # Update parameters of the encoder
            E_optimizer.zero_grad(set_to_none=True)
            G_optimizer.zero_grad(set_to_none=True)
            E_loss.backward()
            E_optimizer.step()
            G_optimizer.step()

            # loss values
            # Register occasionally
            if batch_idx % report_every == 0:
                epoch_wass_dist_WGAN.append(wasserstein_dist_WGAN.data.item())
                epoch_wass_dist_WAE.append(wasserstein_dist_WAE.data.item())

                if double_recon:
                    double_recon_loss_values.append(double_recon_loss.data.item())

                C_epoch_losses.append(C_loss.data.item())
                G_epoch_losses.append(G_loss.data.item())
                E_epoch_losses.append(E_loss.data.item())

                critic_real_value = c_real.mean().data.item()
                critic_fake_value = c_fake.mean().data.item()

                mean_critic_value = (critic_real_value + critic_fake_value) / 2
                mov_avg_critic_val = (
                    (
                        smth_fact * mean_critic_value
                        + (1 - smth_fact) * mov_avg_critic_val
                    )
                    if mov_avg_critic_val is not None
                    else mean_critic_value
                )

                critic_real_values.append(critic_real_value)
                critic_fake_values.append(critic_fake_value)
                critic_mean_values.append(mean_critic_value)
                mov_avg_critic_vals.append(mov_avg_critic_val)

                penalization_epoch.append(penalization.data.item())

                gradient_norm_mean = torch.mean(gradient_norm)
                gradient_norm_values.append(gradient_norm_mean.data.item())

                with torch.no_grad():
                    corrupted_real_val = gaussian_blur(real_val)
                    fake_WGAN = G(fixed_noise)
                    fake_WAE = G(E(real))
                    denoised_real = G(E(real_))
                    fake_val_WAE = G(E(real_val))
                    denoised_real_val = G(E(corrupted_real_val))

                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_real_ = torchvision.utils.make_grid(
                        real_[:32], normalize=True
                    )
                    img_grid_real_val = torchvision.utils.make_grid(
                        real_val[:32], normalize=True
                    )
                    img_grid_real_val_blur = torchvision.utils.make_grid(
                        corrupted_real_val[:32], normalize=True
                    )
                    img_grid_fake_WGAN = torchvision.utils.make_grid(
                        fake_WGAN[:32], normalize=True
                    )
                    img_grid_fake_WAE = torchvision.utils.make_grid(
                        fake_WAE[:32], normalize=True
                    )
                    img_grid_den_real = torchvision.utils.make_grid(
                        denoised_real[:32], normalize=True
                    )
                    img_grid_fake_val_WAE = torchvision.utils.make_grid(
                        fake_val_WAE[:32], normalize=True
                    )
                    img_grid_den_real_val = torchvision.utils.make_grid(
                        denoised_real_val[:32], normalize=True
                    )

                    loss_C_name = "Loss Discriminator"
                    loss_G_name = "Loss Generator"
                    loss_E_name = "Loss Encoder"
                    critic_values_name = "Critic Values"
                    wass_dist_name_WGAN = "Wasserstein Distance WGAN"
                    wass_dist_name_WAE = "Wasserstein Distance WAE"
                    double_recon_name = "Double Reconstruction Loss" if double_recon else None
                    gradient_norm_name = "Gradient Norm"
                    penalization_name = "Penalization"

                    writer_real.add_image(
                        "1 Real", img_grid_real, global_step=step
                    )  # fmt: skip
                    writer_fake.add_image(
                        "2 Decoded", img_grid_fake_WAE, global_step=step
                    )
                    writer_real.add_image(
                        "3 Real Blurred", img_grid_real_, global_step=step
                    )
                    writer_fake.add_image(
                        "4 Denoise", img_grid_den_real, global_step=step
                    )
                    writer_real.add_image(
                        "5 Real Val", img_grid_real_val, global_step=step
                    )
                    writer_fake.add_image(
                        "6 Decoded Val", img_grid_fake_val_WAE, global_step=step
                    )
                    writer_real.add_image(
                        "7 Real Val Blurred", img_grid_real_val_blur, global_step=step
                    )
                    writer_fake.add_image(
                        "8 Denoise Val", img_grid_den_real_val, global_step=step
                    )
                    writer_fake.add_image(
                        "9 Generated", img_grid_fake_WGAN, global_step=step
                    )

                    writer_loss.add_scalars(
                        loss_C_name, {loss_C_name: C_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        loss_G_name, {loss_G_name: G_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        loss_E_name, {loss_E_name: E_loss}, global_step=step
                    )
                    writer_loss.add_scalars(
                        wass_dist_name_WGAN,
                        {wass_dist_name_WGAN: wasserstein_dist_WGAN},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        wass_dist_name_WAE,
                        {wass_dist_name_WAE: wasserstein_dist_WAE},
                        global_step=step,
                    )
                    if double_recon:
                        writer_loss.add_scalars(
                            double_recon_name,
                            {double_recon_name: double_recon_loss},
                            global_step=step,
                        )
                    writer_loss.add_scalars(
                        gradient_norm_name,
                        {gradient_norm_name: gradient_norm_mean},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        penalization_name,
                        {penalization_name: penalization},
                        global_step=step,
                    )
                    writer_loss.add_scalars(
                        critic_values_name,
                        {
                            "Real": critic_real_value,
                            "Fake": critic_fake_value,
                            "Mean": mean_critic_value,
                            "Mov. Avg.": mov_avg_critic_val,
                        },
                        global_step=step,
                    )

                step += 1

        # Register the average loss
        critic_iterations.register_new_loss(mov_avg_critic_val)

        # Validation loop
        if verbose:
            iterable = tqdm(val_data_loader, desc=f"Val. Epoch [{epoch}/{num_epochs}]")
        else:
            iterable = val_data_loader

        epoch_wass_dist_WAE_val = []
        epoch_wass_dist_WGAN_val = []
        with torch.no_grad():
            for real, _ in iterable:
                real = real.to(device, non_blocking=True)
                n = real.shape[0]

                # WAE Validation Loss
                recon = G(E(real))
                loss = criterion(real, recon)
                epoch_wass_dist_WAE_val.append(loss.data.item())

                # WGAN Validation Loss
                z = noise_sampler(n)
                fake = G(z)
                c_real, c_fake = C(real).reshape(-1), C(fake).reshape(-1)
                loss = torch.mean(c_real) - torch.mean(c_fake)
                epoch_wass_dist_WGAN_val.append(loss.data.item())

        # Calculate the average validation loss
        avg_wass_dist_WAE_val = torch.mean(
            torch.FloatTensor(epoch_wass_dist_WAE_val)
        ).item()
        avg_wass_dist_WGAN_val = torch.mean(
            torch.FloatTensor(epoch_wass_dist_WGAN_val)
        ).item()

        if avg_wass_dist_WAE_val < best_validation_loss - min_delta:
            best_validation_loss = avg_wass_dist_WAE_val
            current_patience = 0
            # Save models
            torch.save(deepcopy(G.state_dict()), save_dir / "generator.pt")
            torch.save(deepcopy(C.state_dict()), save_dir / "discriminator.pt")
            torch.save(deepcopy(E.state_dict()), save_dir / "encoder.pt")
        else:
            current_patience += 1
            # Save models
            torch.save(
                deepcopy(G.state_dict()),
                save_dir / "generator_overfitted.pt",
            )  # fmt: skip
            torch.save(
                deepcopy(C.state_dict()),
                save_dir / "discriminator_overfitted.pt",
            )
            torch.save(
                deepcopy(E.state_dict()), 
                save_dir / "encoder_overfitted.pt",
            )  # fmt: skip

        global_epoch = step - 1
        avg_wass_dist_WGAN = torch.mean(torch.FloatTensor(epoch_wass_dist_WGAN)).item()
        avg_wass_dist_WAE = torch.mean(torch.FloatTensor(epoch_wass_dist_WAE)).item()
        avg_double_recon_loss = (
            torch.mean(torch.FloatTensor(double_recon_loss_values)).item()
            if double_recon
            else None
        )
        C_avg_loss = torch.mean(torch.FloatTensor(C_epoch_losses)).item()
        G_avg_loss = torch.mean(torch.FloatTensor(G_epoch_losses)).item()
        E_avg_loss = torch.mean(torch.FloatTensor(E_epoch_losses)).item()
        critic_real_avg_loss = torch.mean(torch.FloatTensor(critic_real_values)).item()
        critic_fake_avg_loss = torch.mean(torch.FloatTensor(critic_fake_values)).item()
        critic_mean_avg_loss = torch.mean(torch.FloatTensor(critic_mean_values)).item()
        penalization_avg = torch.mean(torch.FloatTensor(penalization_epoch)).item()
        mov_avg_critic_val_avg = torch.mean(
            torch.FloatTensor(mov_avg_critic_vals)
        ).item()
        print(f"{critic_iterations = }")
        gradient_norm_values_avg = torch.mean(
            torch.FloatTensor(gradient_norm_values)
        ).item()

        writer_loss.add_scalars(
            wass_dist_name_WGAN,
            {"Avg. " + wass_dist_name_WGAN: avg_wass_dist_WGAN},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WAE,
            {"Avg. " + wass_dist_name_WAE: avg_wass_dist_WAE},
            global_epoch,
        )
        if double_recon:
            writer_loss.add_scalars(
                double_recon_name,
                {"Avg. " + double_recon_name: avg_double_recon_loss},
                global_epoch,
            )
        writer_loss.add_scalars(
            gradient_norm_name,
            {"Avg. " + gradient_norm_name: gradient_norm_values_avg},
            global_epoch,
        )
        writer_loss.add_scalars(
            loss_C_name, {"Avg. " + loss_C_name: C_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            loss_G_name, {"Avg. " + loss_G_name: G_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            loss_E_name, {"Avg. " + loss_E_name: E_avg_loss}, global_epoch
        )
        writer_loss.add_scalars(
            critic_values_name,
            {
                "Avg. Real": critic_real_avg_loss,
                "Avg. Fake": critic_fake_avg_loss,
                "Avg. Mean": critic_mean_avg_loss,
                "Avg. Mov. Avg.": mov_avg_critic_val_avg,
            },
            global_epoch,
        )
        writer_loss.add_scalars(
            penalization_name,
            {"Avg. " + penalization_name: penalization_avg},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WAE,
            {"Avg. " + wass_dist_name_WAE + " Validation": avg_wass_dist_WAE_val},
            global_epoch,
        )
        writer_loss.add_scalars(
            wass_dist_name_WGAN,
            {"Avg. " + wass_dist_name_WGAN + " Validation": avg_wass_dist_WGAN_val},
            global_epoch,
        )

        if current_patience >= patience:
            print("Early stopping triggered. Training halted.")
            break

        # Decrease learning-rate
        G_scheduler.step()
        C_scheduler.step()
        E_scheduler.step()


# Training methods
def train_gan(
    latent_dim,  # =100
    num_filters,  # =[1024, 512, 256, 128]
    channels_img=3,
    learning_rate=1e-4,
    batch_size=128,
    image_size=64,
    num_epochs=100,
    disc_iterations=5,
    lambda_gp=10,
    betas=(0.0, 0.9),
    weight_decay=1e-5,
    milestones=[25, 50, 75],
    device=None,
    transform=None,
    file_path: str | Literal["mnist", "celeba"] = "shoes_images/shoes.hdf5",
    save_dir="networks/",
    summary_writer_dir="logs",
    verbose=True,
    report_every=100,
):
    device = get_device(device)

    # Transforms
    transform = transform or T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    # Dataset and Dataloader
    dataset = get_dataset(file_path=file_path, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # Models
    G = Generator(latent_dim, channels_img, num_filters).to(device)
    C = Critic(channels_img, num_filters[::-1]).to(device)

    # Optimizers
    G_optimizer = optim.Adam(
        G.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )
    C_optimizer = optim.Adam(
        C.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )

    # for tensorboard plotting
    fixed_noise = G.sample_noise(32)
    summary_writer_dir = Path(summary_writer_dir)
    summary_writer_dir.mkdir(exist_ok=True, parents=True)
    writer_real = SummaryWriter(summary_writer_dir / "real")
    writer_fake = SummaryWriter(summary_writer_dir / "fake")
    writer_loss = SummaryWriter(summary_writer_dir / "loss")
    step = 0

    # Schedulers
    G_scheduler = optim.lr_scheduler.MultiStepLR(
        G_optimizer, milestones=milestones, gamma=0.75
    )
    C_scheduler = optim.lr_scheduler.MultiStepLR(
        C_optimizer, milestones=milestones, gamma=0.75
    )

    C.train()
    G.train()

    for epoch in range(num_epochs):
        epoch_wass_dist = []
        C_epoch_losses = []
        G_epoch_losses = []

        if verbose:
            iterable = enumerate(
                tqdm(data_loader, desc=f"Epoch [{epoch}/{num_epochs}]")
            )
        else:
            iterable = enumerate(data_loader)

        for batch_idx, (real, _) in iterable:
            cur_batch_size = real.shape[0]
            real = real.to(device)

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(disc_iterations):
                noise = G.sample_noise(cur_batch_size)
                fake = G(noise)
                critic_real = C(real).reshape(-1)
                critic_fake = C(fake).reshape(-1)
                gp = gradient_penalty(C, real, fake, device=device)
                wasserstein_dist = torch.mean(critic_real) - torch.mean(critic_fake)
                C_loss = -wasserstein_dist + lambda_gp * gp
                C.zero_grad()
                C_loss.backward(retain_graph=True)
                C_optimizer.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = C(fake).reshape(-1)
            G_loss = -torch.mean(gen_fake)
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # loss values
            epoch_wass_dist.append(wasserstein_dist.data.item())
            C_epoch_losses.append(C_loss.data.item())
            G_epoch_losses.append(G_loss.data.item())

            # Print losses occasionally and print to tensorboard
            if batch_idx % report_every == 0 and batch_idx > 0:
                with torch.no_grad():
                    fake = G(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    loss_C_name, loss_G_name = (
                        "Loss Discriminator",
                        "Loss Generator",
                    )
                    wass_dist_name = "Wasserstein Distance"
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    writer_loss.add_scalar(loss_C_name, C_loss, global_step=step)
                    writer_loss.add_scalar(loss_G_name, G_loss, global_step=step)
                    writer_loss.add_scalar(
                        wass_dist_name, wasserstein_dist, global_step=step
                    )

                step += 1

        avg_wass_dist = torch.mean(torch.FloatTensor(epoch_wass_dist)).item()
        C_avg_loss = torch.mean(torch.FloatTensor(C_epoch_losses)).item()
        G_avg_loss = torch.mean(torch.FloatTensor(G_epoch_losses)).item()
        writer_loss.add_scalar("Average Wasserstein Distance", avg_wass_dist, epoch)
        writer_loss.add_scalar("Average loss Discriminator", C_avg_loss, epoch)
        writer_loss.add_scalar("Average loss Generator", G_avg_loss, epoch)

        # Save models
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        torch.save(G.state_dict(), save_dir / "generator")
        torch.save(C.state_dict(), save_dir / "discriminator")

        # Decrease learning-rate
        G_scheduler.step()
        C_scheduler.step()


def _train_encoder_with_noise(
    latent_dim,  # =100
    num_filters,  # =[1024, 512, 256, 128]
    channels_img=3,
    learning_rate=2e-4,
    batch_size=256,
    image_size=64,
    num_epochs=100,
    betas=(0.5, 0.999),
    device=None,
    transform=None,
    file_path="shoes_images/shoes.hdf5",
    save_dir="networks/",
    summary_writer_dir="logs",
    verbose=True,
    report_every=100,
):
    device = get_device(device)

    # Transforms
    transform = transform or T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    # Dataset and Dataloader
    dataset = get_dataset(file_path=file_path, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # Load generator and fix weights
    G = Generator(latent_dim, channels_img, num_filters).to(device)
    save_dir = Path(save_dir)
    generator_path = save_dir / "generator"
    G.load_state_dict(torch.load(generator_path))
    G.eval()
    for param in G.parameters():
        param.requires_grad = False

    E = Encoder(latent_dim, channels_img, num_filters[::-1]).to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    E_optimizer = optim.Adam(
        E.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=1e-5,
    )

    # for tensorboard plotting
    summary_writer_dir = Path(summary_writer_dir)
    summary_writer_dir.mkdir(exist_ok=True, parents=True)
    writer_real = SummaryWriter(summary_writer_dir / "real_encoder")
    writer_fake = SummaryWriter(summary_writer_dir / "fake_encoder")
    writer_loss = SummaryWriter(summary_writer_dir / "loss_encoder")
    step = 0

    E.train()

    for epoch in range(num_epochs):
        E_losses = []

        if verbose:
            iterable = enumerate(
                tqdm(data_loader, desc=f"Epoch [{epoch}/{num_epochs}]")
            )
        else:
            iterable = enumerate(data_loader)

        # minibatch training
        for batch_idx, (real, _) in iterable:
            # generate_noise
            z = G.sample_noise(real.shape[0])
            fake = G(z)

            # Train Encoder
            out_latent = E(fake)
            E_loss = criterion(z, out_latent)

            # Back propagation
            E.zero_grad()
            E_loss.backward()
            E_optimizer.step()

            # loss values
            E_losses.append(E_loss.data.item())

            if batch_idx % report_every == 0 and batch_idx > 0:
                with torch.no_grad():
                    real = real.to(device)
                    fake = G(E(real))
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    real_name, fake_name = "Real", "Generated by Encoder"
                    loss_E_name = "Loss Encoder"
                    writer_real.add_image(real_name, img_grid_real, global_step=step)
                    writer_fake.add_image(fake_name, img_grid_fake, global_step=step)
                    writer_loss.add_scalar(loss_E_name, E_loss, global_step=step)

                step += 1

        E_avg_loss = torch.mean(torch.FloatTensor(E_losses)).item()
        writer_loss.add_scalar("Average loss Encoder", E_avg_loss, epoch)

        # Save models
        torch.save(E.state_dict(), save_dir / "encoder")


def train_encoder_with_wae(
    latent_dim,  # =100
    num_filters,  # =[1024, 512, 256, 128]
    channels_img=3,
    learning_rate=2e-4,
    batch_size=128,
    n=128,
    image_size=64,
    num_epochs=100,
    reg_mmd=10,
    betas=(0.5, 0.999),
    device=None,
    transform=None,
    file_path=None,
    ds_type="quickdraw",
    save_dir="networks/",
    summary_writer_dir="logs",
    verbose=True,
    report_every=100,
):
    device = get_device(device)

    # Transforms
    transform = transform or T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    # Dataset and Dataloader
    dataset, val_dataset = get_dataset(
        ds_type=ds_type, file_path=file_path, transform=transform
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if val_dataset:
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    # Load generator and fix weights
    G = Generator(latent_dim, channels_img, num_filters).to(device)
    save_dir = Path(save_dir)
    generator_path = save_dir / "generator"
    G.load_state_dict(torch.load(generator_path))
    G.eval()
    for param in G.parameters():
        param.requires_grad = False

    E = Encoder(latent_dim, channels_img, num_filters[::-1]).to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    E_optimizer = optim.Adam(
        E.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=1e-5,
    )

    # TODO: FALTA EL StepLR, aunque parece no ser importante

    # for tensorboard plotting
    summary_writer_dir = Path(summary_writer_dir)
    summary_writer_dir.mkdir(exist_ok=True, parents=True)
    writer_real = SummaryWriter(summary_writer_dir / "real_encoder")
    writer_fake = SummaryWriter(summary_writer_dir / "fake_encoder")
    writer_loss = SummaryWriter(summary_writer_dir / "loss_encoder")
    step = 0

    E.train()

    for epoch in range(num_epochs):
        epoch_wass_dist = []
        E_losses = []

        if verbose:
            iterable = enumerate(
                tqdm(data_loader, desc=f"Epoch [{epoch}/{num_epochs}]")
            )
        else:
            iterable = enumerate(data_loader)

        # minibatch training
        for batch_idx, (real, _) in iterable:
            real = real.to(device)
            n = real.shape[0]

            # generate noise
            z = G.sample_noise(
                n, type_as=real
            )  # Generate {z_1, ..., z_n} from the prior P_z
            z_tilde = E(real)  # Generate \tilde{z}_i from Q(Z|x_i) for i = 1:n
            x_recon = G(z_tilde)

            # According to the paper, this is the Wasserstein distance
            wasserstein_dist = criterion(real, x_recon)

            # Compute MMD
            mmd_loss = imq_kernel(z, z_tilde, p_distr=G.latent_distr_name)

            E_loss = wasserstein_dist + reg_mmd * mmd_loss

            # Back propagation
            E.zero_grad()
            E_loss.backward()
            E_optimizer.step()

            # loss values
            epoch_wass_dist.append(wasserstein_dist.data.item())
            E_losses.append(E_loss.data.item())

            if batch_idx % report_every == 0 and batch_idx > 0:
                with torch.no_grad():
                    fake = G(E(real))
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    real_name, fake_name = "Real", "Generated by Encoder"
                    loss_E_name = "Loss Encoder"
                    wass_dist_name = "Wasserstein Distance"
                    writer_real.add_image(real_name, img_grid_real, global_step=step)
                    writer_fake.add_image(fake_name, img_grid_fake, global_step=step)
                    writer_loss.add_scalar(loss_E_name, E_loss, global_step=step)
                    writer_loss.add_scalar(
                        wass_dist_name, wasserstein_dist, global_step=step
                    )

                step += 1

        avg_wass_dist = torch.mean(torch.FloatTensor(epoch_wass_dist)).item()
        E_avg_loss = torch.mean(torch.FloatTensor(E_losses)).item()
        writer_loss.add_scalar("Average Wasserstein Distance", avg_wass_dist, epoch)
        writer_loss.add_scalar("Average loss Encoder", E_avg_loss, epoch)

        # Save models
        torch.save(E.state_dict(), save_dir / "encoder")


def finetune_encoder_with_samples(
    latent_dim,  # =100
    num_filters,  # =[1024, 512, 256, 128]
    channels_img=3,
    learning_rate=2e-4,
    batch_size=128,
    image_size=64,
    num_epochs=100,
    betas=(0.5, 0.999),
    alpha=2e-3,
    device=None,
    transform=None,
    file_path="shoes_images/shoes.hdf5",
    save_dir="networks/",
    summary_writer_dir="logs",
    verbose=True,
    report_every=100,
):
    device = get_device(device)

    # Transforms
    transform = transform or T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    # Dataset and Dataloader
    dataset = get_dataset(file_path=file_path, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # load alexnet:
    alexnet = models.alexnet(pretrained=True).to(device)
    alexnet.eval()
    for param in alexnet.parameters():
        param.requires_grad = False

    # Load generator and fix weights
    G = Generator(latent_dim, channels_img, num_filters).to(device)
    save_dir = Path(save_dir)
    generator_path = save_dir / "generator"
    G.load_state_dict(torch.load(generator_path))
    G.eval()
    for param in G.parameters():
        param.requires_grad = False

    # Load encoder
    E = Encoder(latent_dim, channels_img, num_filters[::-1]).to(device)
    encoder_path = save_dir / "encoder"
    E.load_state_dict(torch.load(encoder_path))
    E.train()

    # Loss function
    criterion = nn.MSELoss()

    # Optimizers
    E_optimizer = optim.Adam(
        E.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=1e-5,
    )

    # for tensorboard plotting
    summary_writer_dir = Path(summary_writer_dir)
    writer_real = SummaryWriter(summary_writer_dir / "real_encoder_finetune")
    writer_fake = SummaryWriter(summary_writer_dir / "fake_encoder_finetune")
    writer_loss = SummaryWriter(summary_writer_dir / "loss_encoder_finetune")
    step = 0

    E_avg_losses = []

    interpolate = lambda x: F.interpolate(x, scale_factor=4, mode="bilinear")
    get_features = lambda x: alexnet.features(alexnet_norm(interpolate(denorm(x))))
    for epoch in range(num_epochs):
        E_losses = []

        if verbose:
            iterable = enumerate(
                tqdm(data_loader, desc=f"Epoch [{epoch}/{num_epochs}]")
            )
        else:
            iterable = enumerate(data_loader)

        # minibatch training
        for batch_idx, (real, _) in iterable:
            # generate_noise
            mini_batch = real.size()[0]
            x = real.to(device)

            # Train Encoder
            out_images = G(E(x))
            E_loss = criterion(x, out_images) + alpha * criterion(
                get_features(x), get_features(out_images)
            )

            # Backprop
            E.zero_grad()
            E_loss.backward()
            E_optimizer.step()

            # loss values
            E_losses.append(E_loss.data.item())

            if batch_idx % report_every == 0 and batch_idx > 0:
                with torch.no_grad():
                    real = real.to(device)
                    fake = G(E(real))
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    real_name, fake_name = "Real", "Generated by Encoder"
                    loss_E_name = "Loss Encoder"
                    writer_real.add_image(real_name, img_grid_real, global_step=step)
                    writer_fake.add_image(fake_name, img_grid_fake, global_step=step)
                    writer_loss.add_scalar(loss_E_name, E_loss, global_step=step)

                step += 1

        E_avg_loss = torch.mean(torch.FloatTensor(E_losses)).item()
        writer_loss.add_scalar("Average loss Encoder", E_avg_loss, epoch)

        # Save models
        torch.save(E.state_dict(), save_dir / "encoder")


def test_encoder(
    latent_dim,  # =100
    num_filters,  # =[1024, 512, 256, 128]
    channels_img=3,
    batch_size=128,
    image_size=64,
    device=None,
    transform=None,
    file_path="shoes_images/shoes.hdf5",
    save_dir="networks/",
    train_log_dir="dcgan_log_dir",
):
    device = get_device(device)

    # Transforms
    transform = transform or T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(
                [0.5 for _ in range(channels_img)],
                [0.5 for _ in range(channels_img)],
            ),
        ]
    )

    # Dataset and Dataloader
    dataset = get_dataset(file_path=file_path, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # load alexnet:
    alexnet = models.alexnet(pretrained=True).to(device)
    alexnet.eval()
    for param in alexnet.parameters():
        param.requires_grad = False

    G = Generator(latent_dim, channels_img, num_filters).to(device)
    save_dir = Path(save_dir)
    generator_path = save_dir / "generator"
    G.load_state_dict(torch.load(generator_path))
    G.eval()
    for param in G.parameters():
        param.requires_grad = False

    E = Encoder(latent_dim, channels_img, num_filters[::-1]).to(device)
    encoder_path = save_dir / "encoder"
    E.load_state_dict(torch.load(encoder_path))
    E.eval()
    for param in E.parameters():
        param.requires_grad = False

    interpolate = lambda x: F.interpolate(x, scale_factor=4, mode="bilinear")

    images, _ = next(iter(data_loader))
    mini_batch = images.size()[0]
    x = images.to(device)
    x_features = alexnet.features(alexnet_norm(interpolate(denorm(x))))

    # Encode
    z = E(x)
    out_images = torch.stack((denorm(x), denorm(G(z))), dim=1)

    z.requires_grad_(True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([z], lr=1e-3)

    for num_epoch in range(100):
        outputs = G(z)
        # loss = criterion(outputs, x_)
        loss = criterion(x, outputs) + 0.002 * criterion(
            x_features,
            alexnet.features(alexnet_norm(interpolate(denorm(outputs)))),
        )
        z.grad = None
        loss.backward()
        optimizer.step()
    out_images = torch.cat((out_images, denorm(G(z)).unsqueeze(1)), dim=1)

    nrow = out_images.shape[1]
    out_images = out_images.reshape(-1, *x.shape[1:])
    train_log_dir = Path(train_log_dir)
    train_log_dir.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    save_image(
        out_images,
        train_log_dir / "encoder_images.png",
        nrow=nrow,
        normalize=False,
        scale_each=False,
        range=(0, 1),
    )


if __name__ == "__main__":
    msg = "Press enter to continue."
    IMAGE_SIZE = (size_x, size_y) = (32, 32)
    CRITERION: Literal["mse", "l1"] = "l1"
    CHANNELS_IMG = 1
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    PATIENCE = 3
    REPORT_EVERY = 10
    # FILE_PATH = "/home/fmunoz/codeProjects/pythonProjects/wgan-gp/dataset/quick_draw/face_recognized.npy"
    # FILE_PATH = (
    #     "/home/fmunoz/codeProjects/pythonProjects/wgan-gp/dataset/cleaned/data.npy"
    # )
    TRANSFORM = T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomZoomOut(side_range=(1, 1.25), p=0.3),
            T.Resize(IMAGE_SIZE),
            T.RandomRotation(
                degrees=(-10, 10), interpolation=T.InterpolationMode.BILINEAR
            ),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
        ]
    )
    MILESTONES = [15, 30, 45, 60, 75, 90]

    LATENT_DIM = 128
    NN_KWARGS = {
        "encoder": dict(Block=ResidualBlock),
        "generator": dict(Block=ResidualBlock, latent_distr=NOISE_NAME),
        "critic": dict(Block=ResidualBlock),
    }

    DS_TYPE = "quickdraw"

    # # data sin contorno arriba
    # DATA_NAME = "data_sin_contorno_arriba"
    # DATA_PATH = Path("dataset") / "cleaned" / f"{DATA_NAME}.npy"
    # NAME_DIR = (
    #     f"cleaned_{DATA_NAME}_zDim{LATENT_DIM}_{NOISE_NAME}_bs_{BATCH_SIZE}"
    # )
    # SAVE_DIR = Path("networks") / NAME_DIR
    # SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    # NUM_EPOCHS = 900
    # train_wgan_and_wae_optimized(
    #     nn_kwargs=NN_KWARGS,
    #     latent_dim=LATENT_DIM,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate_E=3e-4,
    #     learning_rate_C=3e-4,
    #     learning_rate_G=3e-4,
    #     betas_wgan=(0.5, 0.9),
    #     betas_wae=(0.5, 0.9),
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=DATA_PATH,
    #     ds_type=DS_TYPE,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=2,
    #     milestones=MILESTONES,
    #     criterion=CRITERION,
    #     critic_iterations=5,
    #     # crit_iter_patience=3,
    #     patience=400,
    # )
    #
    # # data sin contorno
    # DATA_NAME = "data_sin_contorno"
    # DATA_PATH = Path("dataset") / "cleaned" / f"{DATA_NAME}.npy"
    # NAME_DIR = (
    #     f"cleaned_{DATA_NAME}_zDim{LATENT_DIM}_{NOISE_NAME}_bs_{BATCH_SIZE}"
    # )
    # SAVE_DIR = Path("networks") / NAME_DIR
    # SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    # NUM_EPOCHS = 50
    # train_wgan_and_wae_optimized(
    #     nn_kwargs=NN_KWARGS,
    #     latent_dim=LATENT_DIM,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate_E=3e-4,
    #     learning_rate_C=3e-4,
    #     learning_rate_G=3e-4,
    #     betas_wgan=(0.5, 0.9),
    #     betas_wae=(0.5, 0.9),
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=DATA_PATH,
    #     ds_type=DS_TYPE,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    #     milestones=MILESTONES,
    #     criterion=CRITERION,
    #     critic_iterations=5,
    #     # crit_iter_patience=3,
    #     patience=50,
    # )

    # # data
    # DATA_NAME = "data"
    # DATA_PATH = Path("dataset") / "cleaned" / f"{DATA_NAME}.npy"
    # NAME_DIR = (
    #     f"cleaned_{DATA_NAME}_zDim{LATENT_DIM}_{NOISE_NAME}_bs_{BATCH_SIZE}"
    # )
    # SAVE_DIR = Path("networks") / NAME_DIR
    # SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    # NUM_EPOCHS = 35
    # train_wgan_and_wae_optimized(
    #     nn_kwargs=NN_KWARGS,
    #     latent_dim=LATENT_DIM,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate_E=3e-4,
    #     learning_rate_C=3e-4,
    #     learning_rate_G=3e-4,
    #     betas_wgan=(0.5, 0.9),
    #     betas_wae=(0.5, 0.9),
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=DATA_PATH,
    #     ds_type=DS_TYPE,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    #     milestones=MILESTONES,
    #     criterion=CRITERION,
    #     critic_iterations=5,
    #     # crit_iter_patience=3,
    #     patience=50,
    # )

    # MARK: DWAE TRAINING
    # data
    # DATA_NAME = "data"
    # DATA_PATH = Path("dataset") / "cleaned" / f"{DATA_NAME}.npy"
    # NAME_DIR = f"dae_wwae_{DATA_NAME}_zDim{LATENT_DIM}_{NOISE_NAME}"
    # SAVE_DIR = Path("networks") / NAME_DIR
    # SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    # train_dwae_optimized(
    #     nn_kwargs=NN_KWARGS,
    #     latent_dim=LATENT_DIM,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate_E=3e-4,
    #     learning_rate_G=3e-4,
    #     betas_wgan=(0.5, 0.9),
    #     betas_wae=(0.5, 0.9),
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=DATA_PATH,
    #     ds_type=DS_TYPE,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    #     milestones=MILESTONES,
    #     criterion=CRITERION,
    #     patience=PATIENCE,
    # )

    # MARK: DWAE-WGAN TRAINING
    # data
    # DATA_NAME = "data"
    # DATA_PATH = Path("dataset") / "cleaned" / f"{DATA_NAME}.npy"
    # NAME_DIR = f"dwae_wgan_2_{DATA_NAME}_zDim{LATENT_DIM}_{NOISE_NAME}"
    # SAVE_DIR = Path("networks") / NAME_DIR
    # SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    # train_wgan_and_dwae_optimized(
    #     nn_kwargs=NN_KWARGS,
    #     latent_dim=LATENT_DIM,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate_E=3e-4,
    #     learning_rate_C=3e-4,
    #     learning_rate_G=3e-4,
    #     betas_wgan=(0.5, 0.9),
    #     betas_wae=(0.5, 0.9),
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=DATA_PATH,
    #     ds_type=DS_TYPE,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    #     milestones=MILESTONES,
    #     criterion=CRITERION,
    #     patience=PATIENCE,
    # )

    # MARK: DWAE-WGAN DOUBLE RECONSTRUCTION TRAINING
    # data
    # DATA_NAME = "data"
    # DATA_PATH = Path("dataset") / "cleaned" / f"{DATA_NAME}.npy"
    # NAME_DIR = f"dwae_wgan_3_{DATA_NAME}_zDim{LATENT_DIM}_{NOISE_NAME}"
    # SAVE_DIR = Path("networks") / NAME_DIR
    # SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    # train_wgan_and_dwae_double_recon_optimized(
    #     nn_kwargs=NN_KWARGS,
    #     latent_dim=LATENT_DIM,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate_E=3e-4,
    #     learning_rate_C=3e-4,
    #     learning_rate_G=3e-4,
    #     betas_wgan=(0.5, 0.9),
    #     betas_wae=(0.5, 0.9),
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=DATA_PATH,
    #     ds_type=DS_TYPE,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    #     milestones=MILESTONES,
    #     criterion=CRITERION,
    #     patience=PATIENCE,
    # )

    # MARK: DWAE DOUBLE RECONSTRUCTION TRAINING
    # data
    # DATA_NAME = "data"
    # DATA_PATH = Path("dataset") / "cleaned" / f"{DATA_NAME}.npy"
    # NAME_DIR = f"dae_wwae_double_recon_{DATA_NAME}_zDim{LATENT_DIM}_{NOISE_NAME}"
    # SAVE_DIR = Path("networks") / NAME_DIR
    # SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    # train_dwae_optimized(
    #     nn_kwargs=NN_KWARGS,
    #     latent_dim=LATENT_DIM,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate_E=3e-4,
    #     learning_rate_G=3e-4,
    #     betas_wgan=(0.5, 0.9),
    #     betas_wae=(0.5, 0.9),
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=DATA_PATH,
    #     ds_type=DS_TYPE,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    #     milestones=MILESTONES,
    #     criterion=CRITERION,
    #     patience=PATIENCE,
    #     denoise=True,
    #     double_recon=True,
    # )

    # MARK: WAE-WGAN DOUBLE RECONSTRUCTION TRAINING
    # data
    for strategy_double_recon in ["proj_vs_proj2", "from_gen", "from_real"]:
        DATA_NAME = "data"
        DATA_PATH = Path("dataset") / "cleaned" / f"{DATA_NAME}.npy"
        NAME_DIR = f"dwae_wgan_5_{strategy_double_recon}_{DATA_NAME}_zDim{LATENT_DIM}_{NOISE_NAME}"
        SAVE_DIR = Path("networks") / NAME_DIR
        SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
        train_wgan_and_dwae_double_recon_optimized(
            nn_kwargs=NN_KWARGS,
            latent_dim=LATENT_DIM,
            channels_img=CHANNELS_IMG,
            image_size=IMAGE_SIZE,
            learning_rate_E=3e-4,
            learning_rate_C=3e-4,
            learning_rate_G=3e-4,
            betas_wgan=(0.5, 0.9),
            betas_wae=(0.5, 0.9),
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            file_path=DATA_PATH,
            ds_type=DS_TYPE,
            save_dir=SAVE_DIR,
            summary_writer_dir=SUMMARY_WRITER_DIR,
            transform=TRANSFORM,
            report_every=REPORT_EVERY,
            milestones=MILESTONES,
            criterion=CRITERION,
            patience=PATIENCE,
            denoising=True,
            strategy_double_recon=strategy_double_recon
        )

    # LATENT_DIM = 64
    # NN_KWARGS = {
    #     "encoder": dict(Block=ResidualBlock),
    #     "generator": dict(Block=ResidualBlock, latent_distr=NOISE_NAME),
    #     "critic": dict(Block=ResidualBlock),
    # }
    # NAME_DIR = f"_resnet_face_zDim{LATENT_DIM}_{NOISE_NAME}_bs_{BATCH_SIZE}_cleaned_augmented_WAE_WGAN_loss_{CRITERION}_{size_x}p{size_y}"
    # SAVE_DIR = Path("networks") / NAME_DIR
    # SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    # train_wgan_and_wae(
    #     nn_kwargs=NN_KWARGS,
    #     latent_dim=LATENT_DIM,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate_E=3e-4,
    #     learning_rate_C=3e-4,
    #     learning_rate_G=3e-4,
    #     betas_wgan=(0.5, 0.9),
    #     betas_wae=(0.5, 0.9),
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=FILE_PATH,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    #     milestones=MILESTONES,
    #     criterion=CRITERION,
    # )

    # input(msg)
    # LATENT_DIM = 64
    # NN_KWARGS = {
    #     "encoder": dict(Block=ResidualBlockV2),
    #     "generator": dict(Block=ResidualBlockV2),
    #     "critic": dict(Block=ResidualBlockV2),
    # }
    # NAME_DIR = f"_resnetV2_face_zDim{LATENT_DIM}_{NOISE_NAME}_bs_{BATCH_SIZE}_recognized_augmented_WAE_WGAN_loss_{CRITERION}_{size_x}p{size_y}"
    # SAVE_DIR = Path("networks") / NAME_DIR
    # SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    # train_wgan_and_wae(
    #     nn_kwargs=NN_KWARGS,
    #     latent_dim=LATENT_DIM,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate_E=3e-4,
    #     learning_rate_C=3e-4,
    #     learning_rate_G=3e-4,
    #     betas_wgan=(0.5, 0.9),
    #     betas_wae=(0.5, 0.9),
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=FILE_PATH,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    #     milestones=MILESTONES,
    #     criterion=CRITERION,
    # )

    # LATENT_DIM = 128
    # NN_KWARGS = {
    #     "encoder": dict(Block=ResidualBlockV2),
    #     "generator": dict(Block=ResidualBlockV2),
    #     "critic": dict(Block=ResidualBlockV2),
    # }
    # NAME_DIR = f"_resnetV2_face_zDim{LATENT_DIM}_{NOISE_NAME}_bs_{BATCH_SIZE}_recognized_augmented_WAE_WGAN_loss_{CRITERION}_{size_x}p{size_y}"
    # SAVE_DIR = Path("networks") / NAME_DIR
    # SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    # train_wgan_and_wae(
    #     nn_kwargs=NN_KWARGS,
    #     latent_dim=LATENT_DIM,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate_E=3e-4,
    #     learning_rate_C=3e-4,
    #     learning_rate_G=3e-4,
    #     betas_wgan=(0.5, 0.9),
    #     betas_wae=(0.5, 0.9),
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=FILE_PATH,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    #     milestones=MILESTONES,
    #     criterion=CRITERION,
    # )

    # train_gan(
    #     latent_dim=LATENT_DIM,
    #     num_filters=NUM_FILTERS,
    #     channels_img=CHANNELS_IMG,
    #     image_size=IMAGE_SIZE,
    #     learning_rate=3e-4,
    #     batch_size=BATCH_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=FILE_PATH,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    #     milestones=MILESTONES,
    # )

    # train_encoder_with_wae(
    #     latent_dim=LATENT_DIM,
    #     num_filters=NUM_FILTERS,
    #     channels_img=CHANNELS_IMG,
    #     learning_rate=2e-4,
    #     n=BATCH_SIZE,
    #     image_size=IMAGE_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=FILE_PATH,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    # )

    # finetune_encoder_with_samples(
    #     latent_dim=LATENT_DIM,
    #     num_filters=NUM_FILTERS,
    #     channels_img=CHANNELS_IMG,
    #     learning_rate=2e-4,
    #     batch_size=BATCH_SIZE,
    #     image_size=IMAGE_SIZE,
    #     num_epochs=NUM_EPOCHS,
    #     file_path=FILE_PATH,
    #     save_dir=SAVE_DIR,
    #     summary_writer_dir=SUMMARY_WRITER_DIR,
    #     transform=TRANSFORM,
    #     report_every=REPORT_EVERY,
    # )

    # test_encoder(
    #     latent_dim=LATENT_DIM,
    #     num_filters=NUM_FILTERS,
    #     channels_img=CHANNELS_IMG,
    #     file_path=FILE_PATH,
    #     save_dir=SAVE_DIR,
    # )
