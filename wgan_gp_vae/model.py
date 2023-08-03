from typing import Any, Iterable, List

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchvision

from wgan_gp_vae.utils import gradient_penalty


# Generator model
class Generator(nn.Module):
    def __init__(
        self, latent_dim, channels_img=3, num_filters: Iterable[int] = (128, 64, 32, 16)
    ) -> None:
        super().__init__()

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

    def forward(self, z):
        x = self.hidden_layer(z)
        out = self.output_layer(x)
        return out


# Critic model
class Critic(nn.Module):
    def __init__(self, channels_img=3, num_filters: Iterable[int] = (8, 16, 32, 64)):
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
        self, latent_dim, channels_img=3, num_filters: Iterable[int] = (16, 32, 64, 128)
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


class WGAN_GP(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        channels_img: int = 3,
        num_filters: Iterable[int] = (256, 128, 64, 32),
        reg_gp: float = 10,
        num_epochs: int = 100,
        critic_iterations: int = 5,
        batch_size: int = 128,
        learning_rate: float = 3e-4,
        betas=(0.0, 0.9),
        weight_decay=1e-5,
        milestones=(25, 50, 75),
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = Generator(
            latent_dim=self.hparams.latent_dim,
            channels_img=self.hparams.channels_img,
            num_filters=self.hparams.num_filters,
        )
        self.critic = Critic(
            channels_img=self.hparams.channels_img,
            num_filters=self.hparams.num_filters[::-1],
        )
        self.critic.train()
        self.generator.train()

        self.validation_z = 2 * torch.rand(32, self.hparams.latent_dim, 1, 1) - 1

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim, 1, 1)

        self.step = 0
        self.c_epoch_losses = []
        self.g_epoch_losses = []

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = self.hparams.betas
        weight_decay = self.hparams.weight_decay
        g_opt = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        c_opt = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        return [g_opt, c_opt], []

    def adversarial_loss(self, real, fake):
        critic_real = self.critic(real).reshape(-1)
        critic_fake = self.critic(fake).reshape(-1)

        BATCH_SIZE, C, H, W = real.shape
        alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).type_as(real)
        interpolated_images: torch.Tensor = real * alpha + fake * (1 - alpha)

        # Calculate critic scores
        mixed_scores = self.critic(interpolated_images)

        print(interpolated_images.requires_grad)
        print(mixed_scores.requires_grad)

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
        gp = torch.mean((gradient_norm - 1) ** 2)
        # gp = gradient_penalty(self.critic, real, fake, real.device)

        loss = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + self.hparams.reg_gp * gp
        )
        return loss

    def training_step(self, batch, batch_idx):
        real, _ = batch

        optimizer_g, optimizer_c = self.optimizers()

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        self.toggle_optimizer(optimizer_c)

        for _ in range(self.hparams.critic_iterations):
            # Sample noise
            z = 2 * torch.rand(real.shape[0], self.hparams.latent_dim, 1, 1) - 1
            z = z.type_as(real)

            fake = self.generator(z)

            c_loss = self.adversarial_loss(real, fake)

            self.log("c_loss", c_loss, prog_bar=True)
            self.manual_backward(c_loss, retain_graph=True)
            optimizer_c.step()
            optimizer_c.zero_grad()
            self.untoggle_optimizer(optimizer_c)

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        self.toggle_optimizer(optimizer_g)

        gen_fake = self.critic(fake).reshape(-1)
        g_loss = -torch.mean(gen_fake)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # loss values
        self.c_epoch_losses.append(c_loss.data.item())
        self.g_epoch_losses.append(g_loss.data.item())

        if batch_idx % 100 == 0 and batch_idx > 0:
            with torch.no_grad():
                self.generated_imgs = self(z)

                # log sampled images
                sample_imgs = self.generated_imgs[:32]
                img_grid_fake = torchvision.utils.make_grid(sample_imgs, normalize=True)
                self.logger.experiment.add_image(
                    "Fake", img_grid_fake, global_step=self.step
                )
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                self.logger.experiment.add_image(
                    "Real", img_grid_real, global_step=self.step
                )

            self.step += 1
        return

    def on_validation_epoch_end(self) -> None:
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # Log sampled imagees
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs, normalize=True)
        self.logger.experiment.add_image(
            "generated_images", grid, global_step=self.current_epoch
        )
        return super().on_validation_epoch_end()


def _test():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, in_channels, H, W = 8, 1, 64, 64
    latent_dim = 100
    x = torch.randn((N, in_channels, H, W)).to(dev)
    z = torch.randn((N, latent_dim, 1, 1)).to(dev)

    # Generator test
    gen = Generator(latent_dim, in_channels, [128, 64, 32, 16]).to(dev)
    assert gen(z).shape == (
        N,
        in_channels,
        H,
        W,
    ), f"Generator test failed. {gen(z).shape = }"

    # Discriminator test
    disc = Critic(in_channels, [8, 16, 32, 64]).to(dev)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminatior test failed"

    # Encoder
    enc = Encoder(latent_dim, in_channels, [16, 32, 64, 128]).to(dev)
    assert enc(x).shape == (N, latent_dim, 1, 1), "Encoder test failed"

    print("Tests passed")

    wgan_gp = WGAN_GP()
    print(f'{wgan_gp.hparams["reg_gp"] = }')


if __name__ == "__main__":
    _test()
