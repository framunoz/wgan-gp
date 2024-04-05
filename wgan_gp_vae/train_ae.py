from torchvision import disable_beta_transforms_warning

disable_beta_transforms_warning()

from pathlib import Path

import matplotlib.pyplot as plt
import quick_torch as qt
import torch
import torch.optim as optim
import torchvision.transforms.v2 as T
import tqdm
from model_resnet import *
from torch.utils.data import DataLoader

plt.rcParams["figure.figsize"] = 15, 10


INIT = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
REPORTS_EVERY = 1
LR = 3e-4
BETAS = (0.5, 0.999)
WD = 1e-5
LATENT_DIM = 8
BATCH_SIZE = 512

SAVE_DIR = Path("saved_models")
SAVE_DIR.mkdir(exist_ok=True)
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

transforms = T.Compose(
    [
        T.Resize(32),
        T.ToImageTensor(),
        T.ConvertDtype(),
        T.Normalize([0.5 for _ in range(1)], [0.5 for _ in range(1)]),
    ]
)


dataset = qt.QuickDraw(
    "dataset",
    [qt.Category.FACE, qt.Category.SMILEY_FACE],
    train=None,
    transform=transforms,
    download=True,
)


data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


G = Generator(
    LATENT_DIM,
    1,
    num_filters=[64] * 5,
    resample_list=["up"] * 3 + [None],
).to(DEV)
E = Encoder(
    LATENT_DIM,
    1,
    num_filters=[64] * 4,
    resample_list=["down"] * 3 + [None],
).to(DEV)

N, in_ch, H, W = 8, 1, 32, 32
x = torch.randn((N, in_ch, H, W)).to(DEV)
z = torch.randn((N, LATENT_DIM, 1, 1)).to(DEV)

assert E(x).shape == (N, LATENT_DIM, 1, 1), f"Generator test failed {E(x).shape = }"
assert G(z).shape == (N, in_ch, H, W), "Generator test failed"

criterion = nn.MSELoss()


G_optimizer = optim.Adam(G.parameters(), lr=LR, betas=BETAS, weight_decay=WD)

E_optimizer = optim.Adam(E.parameters(), lr=LR, betas=BETAS, weight_decay=WD)


losses = []
val_losses_avg = []

outputs = {}
x_real_val, _ = next(iter(data_loader))
x_real_val = x_real_val[:10].to(DEV)

for epoch in range(1, EPOCHS + 1):
    sum_loss = 0

    # Train loop
    tepoch = tqdm.tqdm(
        data_loader,
        unit="batch",
        desc=f"Epoch [{epoch}/{EPOCHS}]",
    )
    iterable = enumerate(tepoch)
    for batch_idx, (x_real, _) in iterable:
        # Reconstruction
        x_real = x_real.to(DEV)
        x_recon = G(E(x_real))

        loss = criterion(x_real, x_recon)
        sum_loss += loss.item()
        tepoch.set_postfix(
            loss=f"{loss.item():.6f}", avg_loss=f"{sum_loss / (batch_idx + 1):.6f}"
        )

        E.zero_grad()
        G.zero_grad()
        loss.backward()
        E_optimizer.step()
        G_optimizer.step()

        if batch_idx % REPORTS_EVERY == 0:
            losses.append(loss.item())

    tepoch.close()

    # Validation
    with torch.no_grad():
        x_recon_val = G(E(x_real_val))
        x_real_val_ = x_real_val.detach().cpu().numpy()
        x_recon_val_ = x_recon_val.detach().cpu().numpy()

        count = 1

        for idx in range(10):
            plt.subplot(2, 10, count)
            plt.title("Original\nImage")
            plt.imshow(x_real_val_[idx].reshape(32, 32), cmap="gray")
            plt.axis("off")
            count += 1

        for idx in range(10):
            plt.subplot(2, 10, count)
            plt.title("Reconstructed\nImage")
            plt.imshow(x_recon_val_[idx].reshape(32, 32), cmap="gray")
            plt.axis("off")
            count += 1

        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"epoch_{epoch}.png")

    # Save model
    torch.save(E.state_dict(), SAVE_DIR / "encoder")
    torch.save(G.state_dict(), SAVE_DIR / "generator")

plt.plot(range(INIT, len(losses)), losses[INIT:])
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs. Iterations")
plt.savefig(PLOT_DIR / "losses.png")

# Here write a code for an outlier detection using the autoencoder
