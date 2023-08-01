import os
from pathlib import Path
from typing import Literal

import dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from model import Discriminator, Encoder, Generator
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
from utils import alexnet_norm, denorm, gradient_penalty


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
    file_path: str | Literal["mnist", "celeba"] = "shoes_images/shoes.hdf5",
    save_dir="networks/",
    summary_writer_dir="logs",
    verbose=True,
):
    # Define device
    if device is None:  # Default
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Transforms
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
            ),
        ]
    )

    # Dataset and Dataloader
    if file_path == "mnist":
        dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    elif file_path == "celeba":
        dataset = datasets.ImageFolder(root="celeb_dataset", transform=transform)
    else:
        dataset = dataloader.H5Loader(file_path)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Models
    G = Generator(latent_dim, channels_img, num_filters).to(device)
    D = Discriminator(channels_img, num_filters[::-1]).to(device)

    # Optimizers
    G_optimizer = optim.Adam(
        G.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )
    D_optimizer = optim.Adam(
        D.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )

    # for tensorboard plotting
    fixed_noise = torch.randn(32, latent_dim, 1, 1).to(device)
    summary_writer_dir = Path(summary_writer_dir)
    writer_real = SummaryWriter(summary_writer_dir / "real")
    writer_fake = SummaryWriter(summary_writer_dir / "fake")
    writer_loss = SummaryWriter(summary_writer_dir / "loss")
    step = 0

    # Schedulers
    G_scheduler = optim.lr_scheduler.MultiStepLR(
        G_optimizer, milestones=milestones, gamma=0.75
    )
    D_scheduler = optim.lr_scheduler.MultiStepLR(
        D_optimizer, milestones=milestones, gamma=0.75
    )

    D.train()
    G.train()

    for epoch in range(num_epochs):
        D_epoch_losses = []
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
                noise = torch.randn(cur_batch_size, latent_dim, 1, 1).to(device)
                fake = G(noise)
                critic_real = D(real).reshape(-1)
                critic_fake = D(fake).reshape(-1)
                gp = gradient_penalty(D, real, fake, device=device)
                D_loss = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + lambda_gp * gp
                )
                D.zero_grad()
                D_loss.backward(retain_graph=True)
                D_optimizer.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = D(fake).reshape(-1)
            G_loss = -torch.mean(gen_fake)
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # loss values
            D_epoch_losses.append(D_loss.data.item())
            G_epoch_losses.append(G_loss.data.item())

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0 and batch_idx > 0:
                with torch.no_grad():
                    fake = G(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    loss_D_name, loss_G_name = "Loss Discriminator", "Loss Generator"
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    writer_loss.add_scalar(loss_D_name, D_loss, global_step=step)
                    writer_loss.add_scalar(loss_G_name, G_loss, global_step=step)

                step += 1

        D_avg_loss = torch.mean(torch.FloatTensor(D_epoch_losses)).item()
        G_avg_loss = torch.mean(torch.FloatTensor(G_epoch_losses)).item()
        writer_loss.add_scalar("Average loss Discriminator", D_avg_loss, epoch)
        writer_loss.add_scalar("Average loss Generator", G_avg_loss, epoch)

        # Save models
        save_dir = Path(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(G.state_dict(), save_dir / "generator")
        torch.save(D.state_dict(), save_dir / "discriminator")

        # Decrease learning-rate
        G_scheduler.step()
        D_scheduler.step()


def train_encoder_with_noise(
    latent_dim,  # =100
    num_filters,  # =[1024, 512, 256, 128]
    channels_img=3,
    learning_rate=2e-4,
    batch_size=128,
    image_size=64,
    num_epochs=100,
    betas=(0.5, 0.999),
    device=None,
    file_path="shoes_images/shoes.hdf5",
    save_dir="networks/",
    summary_writer_dir="logs",
    verbose=True,
):
    # Define device
    if device is None:  # Default
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Transforms
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
            ),
        ]
    )

    # Dataset and Dataloader
    if file_path == "mnist":
        dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    elif file_path == "celeba":
        dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
    else:
        dataset = dataloader.H5Loader(file_path)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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
            z = torch.randn(real.shape[0], latent_dim, 1, 1).to(device)
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

            if batch_idx % 100 == 0 and batch_idx > 0:
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
    file_path="shoes_images/shoes.hdf5",
    save_dir="networks/",
    summary_writer_dir="logs",
    verbose=True,
):
    # Define device
    if device is None:  # Default
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Transforms
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
            ),
        ]
    )

    # Dataset and Dataloader
    if file_path == "mnist":
        dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    elif file_path == "celeba":
        dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
    else:
        dataset = dataloader.H5Loader(file_path)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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

            if batch_idx % 100 == 0 and batch_idx > 0:
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
    file_path="shoes_images/shoes.hdf5",
    save_dir="networks/",
    train_log_dir="dcgan_log_dir",
):
    # Define device
    if device is None:  # Default
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Transforms
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
            ),
        ]
    )

    # Dataset and Dataloader
    if file_path == "mnist":
        dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
    elif file_path == "celeba":
        dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
    else:
        dataset = dataloader.H5Loader(file_path)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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
    LATENT_DIM = 100
    NUM_FILTERS = [256, 128, 64, 32]
    CHANNELS_IMG = 1
    BATCH_SIZE = 64
    FILE_PATH = "mnist"
    SAVE_DIR = "networks/MNIST"
    SUMMARY_WRITER_DIR = "logs/WGANGP_MNIST"

    train_gan(
        latent_dim=LATENT_DIM,
        num_filters=NUM_FILTERS,
        channels_img=CHANNELS_IMG,
        learning_rate=3e-4,
        batch_size=BATCH_SIZE,
        file_path=FILE_PATH,
        save_dir=SAVE_DIR,
        summary_writer_dir=SUMMARY_WRITER_DIR,
    )

    train_encoder_with_noise(
        latent_dim=LATENT_DIM,
        num_filters=NUM_FILTERS,
        channels_img=CHANNELS_IMG,
        learning_rate=3e-4,
        batch_size=BATCH_SIZE,
        file_path=FILE_PATH,
        save_dir=SAVE_DIR,
        summary_writer_dir=SUMMARY_WRITER_DIR,
    )

    finetune_encoder_with_samples(
        latent_dim=LATENT_DIM,
        num_filters=NUM_FILTERS,
        channels_img=CHANNELS_IMG,
        learning_rate=3e-4,
        batch_size=BATCH_SIZE,
        file_path=FILE_PATH,
        save_dir=SAVE_DIR,
        summary_writer_dir=SUMMARY_WRITER_DIR,
    )

    # test_encoder(
    #     latent_dim=LATENT_DIM,
    #     num_filters=NUM_FILTERS,
    #     channels_img=CHANNELS_IMG,
    #     file_path=FILE_PATH,
    #     save_dir=SAVE_DIR,
    # )
