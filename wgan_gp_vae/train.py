from torchvision import disable_beta_transforms_warning

disable_beta_transforms_warning()

import multiprocessing as mp
import os
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
import torchvision.transforms.v2 as T
from model_resnet import Critic, Encoder, Generator, ResidualBlock, ResidualBlockV2
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from utils import alexnet_norm, denorm, gradient_penalty, imq_kernel

NUM_WORKERS = mp.cpu_count()
NOISE_NAME = "gauss"
type_models = "resnet"


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


def get_dataset(file_path: str, transform):
    dataset = None
    test_dataset = None

    if file_path == "mnist":
        dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    elif file_path == "celeba":
        dataset = datasets.ImageFolder(root="celeb_dataset", transform=transform)
    elif file_path == "quickdraw":
        categories = [qt.Category.FACE]
        dataset = qt.QuickDraw(
            root="dataset",
            categories=categories,
            train=True,
            transform=transform,
            download=True,
            recognized=True,
        )
        test_dataset = qt.QuickDraw(
            root="dataset",
            categories=categories,
            train=False,
            transform=transform,
            download=True,
            recognized=True,
        )

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
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
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

                    loss_C_name, loss_G_name = "Loss Discriminator", "Loss Generator"
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
    penalty_wgan=20,
    penalty_wae=10,
    criterion: Literal["mse", "l1"] = "l1",
    betas_wgan=(0.5, 0.9),
    betas_wae=(0.5, 0.999),
    weight_decay=1e-5,
    milestones=[25, 50, 75],
    patience=5,  # Number of epochs to wait for improvement
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
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
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

    print(E)
    print(G)
    print(C)

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

    for epoch in range(num_epochs):
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
                    desc=f"Epoch [{epoch}/{num_epochs}], Patience {current_patience}",
                )
            )
        else:
            iterable = enumerate(data_loader)

        # Training loop
        for batch_idx, (real, _) in iterable:
            n = real.shape[0]
            real = real.to(device)

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(critic_iterations):
                noise = G.sample_noise(n)
                fake = G(noise)
                c_real = C(real).reshape(-1)
                c_fake = C(fake).reshape(-1)
                gradient, gradient_norm = gradient_penalty(C, real, fake, device=device)
                wasserstein_dist_WGAN = torch.mean(c_real) - torch.mean(c_fake)
                C_loss = -wasserstein_dist_WGAN + penalty_wgan * gradient
                C.zero_grad()
                C_loss.backward(retain_graph=True)
                C_optimizer.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            c_fake = C(fake).reshape(-1)
            G_loss = -torch.mean(c_fake)
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # Train Encoder + Generator
            # generate noise
            z = G.sample_noise(n)  # Generate {z_1, ..., z_n} from the prior P_z
            z_tilde = E(real)  # Generate \tilde{z}_i from Q(Z|x_i) for i = 1:n
            x_recon = G(z_tilde)

            # According to the paper, this is the Wasserstein distance
            wasserstein_dist_WAE = criterion(real, x_recon)

            # Compute MMD
            mmd_loss = imq_kernel(z, z_tilde, p_distr=G.latent_distr)

            E_loss = wasserstein_dist_WAE + penalty_wae * mmd_loss

            # Back propagation
            E.zero_grad()
            G.zero_grad()
            E_loss.backward()
            E_optimizer.step()
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
                noise = G.sample_noise(n)
                fake = G(noise)
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
            {"Avg. Real": critic_real_avg_loss, "Avg. Fake": critic_fake_avg_loss},
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


def _train_encoder_with_noise(
    latent_dim,  # =100
    num_filters,  # =[1024, 512, 256, 128]
    channels_img=3,
    learning_rate=2e-4,
    batch_size=128,
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
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
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
    n=128,
    image_size=64,
    num_epochs=100,
    reg_mmd=10,
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
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
            ),
        ]
    )

    # Dataset and Dataloader
    dataset = get_dataset(file_path=file_path, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=n,
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
            z = G.sample_noise(n)  # Generate {z_1, ..., z_n} from the prior P_z
            z_tilde = E(real)  # Generate \tilde{z}_i from Q(Z|x_i) for i = 1:n
            x_recon = G(z_tilde)

            # According to the paper, this is the Wasserstein distance
            wasserstein_dist = criterion(real, x_recon)

            # Compute MMD
            mmd_loss = imq_kernel(z, z_tilde, p_distr=G.latent_distr)

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
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
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
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
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
            x_features, alexnet.features(alexnet_norm(interpolate(denorm(outputs))))
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
    CRITERION = "l1"
    CHANNELS_IMG = 1
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    REPORT_EVERY = 10
    # FILE_PATH = "/home/fmunoz/codeProjects/pythonProjects/wgan-gp/dataset/quick_draw/face_recognized.npy"
    FILE_PATH = "quickdraw"
    TRANSFORM = T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomZoomOut(side_range=(1, 1.25)),
            T.Resize(IMAGE_SIZE),
            T.RandomRotation(
                degrees=(-10, 10), interpolation=T.InterpolationMode.BILINEAR
            ),
            T.ToImageTensor(),
            T.ConvertDtype(torch.float32),
            T.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
        ]
    )
    MILESTONES = [15, 30, 45, 60, 75, 90]

    LATENT_DIM = 64
    NN_KWARGS = {
        "encoder": dict(Block=ResidualBlock),
        "generator": dict(Block=ResidualBlock),
        "critic": dict(Block=ResidualBlock),
    }
    NAME_DIR = f"_resnet_face_zDim{LATENT_DIM}_{NOISE_NAME}_bs_{BATCH_SIZE}_recognized_augmented_WAE_WGAN_loss_{CRITERION}_{size_x}p{size_y}"
    SAVE_DIR = Path("networks") / NAME_DIR
    SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    train_wgan_and_wae(
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
        file_path=FILE_PATH,
        save_dir=SAVE_DIR,
        summary_writer_dir=SUMMARY_WRITER_DIR,
        transform=TRANSFORM,
        report_every=REPORT_EVERY,
        milestones=MILESTONES,
        criterion=CRITERION,
    )

    LATENT_DIM = 128
    NN_KWARGS = {
        "encoder": dict(Block=ResidualBlock),
        "generator": dict(Block=ResidualBlock),
        "critic": dict(Block=ResidualBlock),
    }
    NAME_DIR = f"_resnet_face_zDim{LATENT_DIM}_{NOISE_NAME}_bs_{BATCH_SIZE}_recognized_augmented_WAE_WGAN_loss_{CRITERION}_{size_x}p{size_y}"
    SAVE_DIR = Path("networks") / NAME_DIR
    SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    train_wgan_and_wae(
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
        file_path=FILE_PATH,
        save_dir=SAVE_DIR,
        summary_writer_dir=SUMMARY_WRITER_DIR,
        transform=TRANSFORM,
        report_every=REPORT_EVERY,
        milestones=MILESTONES,
        criterion=CRITERION,
    )

    input(msg)
    LATENT_DIM = 64
    NN_KWARGS = {
        "encoder": dict(Block=ResidualBlockV2),
        "generator": dict(Block=ResidualBlockV2),
        "critic": dict(Block=ResidualBlockV2),
    }
    NAME_DIR = f"_resnetV2_face_zDim{LATENT_DIM}_{NOISE_NAME}_bs_{BATCH_SIZE}_recognized_augmented_WAE_WGAN_loss_{CRITERION}_{size_x}p{size_y}"
    SAVE_DIR = Path("networks") / NAME_DIR
    SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    train_wgan_and_wae(
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
        file_path=FILE_PATH,
        save_dir=SAVE_DIR,
        summary_writer_dir=SUMMARY_WRITER_DIR,
        transform=TRANSFORM,
        report_every=REPORT_EVERY,
        milestones=MILESTONES,
        criterion=CRITERION,
    )

    LATENT_DIM = 128
    NN_KWARGS = {
        "encoder": dict(Block=ResidualBlockV2),
        "generator": dict(Block=ResidualBlockV2),
        "critic": dict(Block=ResidualBlockV2),
    }
    NAME_DIR = f"_resnetV2_face_zDim{LATENT_DIM}_{NOISE_NAME}_bs_{BATCH_SIZE}_recognized_augmented_WAE_WGAN_loss_{CRITERION}_{size_x}p{size_y}"
    SAVE_DIR = Path("networks") / NAME_DIR
    SUMMARY_WRITER_DIR = Path("logs") / NAME_DIR
    train_wgan_and_wae(
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
        file_path=FILE_PATH,
        save_dir=SAVE_DIR,
        summary_writer_dir=SUMMARY_WRITER_DIR,
        transform=TRANSFORM,
        report_every=REPORT_EVERY,
        milestones=MILESTONES,
        criterion=CRITERION,
    )

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
