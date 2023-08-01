import torch
import torch.nn as nn


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


def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint["gen"])
    disc.load_state_dict(checkpoint["disc"])


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
