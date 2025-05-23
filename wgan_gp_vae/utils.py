import abc
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

_models_option = "encoder", "generator"
TypeModels = Literal["encoder", "generator"]


@torch.jit.script
def interpolate_images(real: torch.Tensor, fake: torch.Tensor, alpha: torch.Tensor):
    return real * alpha + fake * (1 - alpha)


@torch.jit.script
def compute_gradient_penalty_norm(gradient: torch.Tensor, penalty_gr: float):
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean(F.leaky_relu(gradient_norm - 1, penalty_gr) ** 2)
    return gradient_penalty, gradient_norm


def gradient_penalty_improved(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device="cpu",
    penalty_gr: float = 10.0,
):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1), device=device).repeat(1, C, H, W)
    interpolated_images = interpolate_images(real, fake, alpha)

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
    gradient_penalty, gradient_norm = compute_gradient_penalty_norm(
        gradient,  # torch.as_tensor(penalty_gr, device=device)
        penalty_gr,
    )
    return gradient_penalty, gradient_norm


def gradient_penalty(critic, real, fake, device="cpu", penalty_gr=10):
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
    gradient_penalty = torch.mean(F.leaky_relu(gradient_norm - 1, penalty_gr) ** 2)
    return gradient_penalty, gradient_norm


def load_checkpoint(
    model: nn.Module, root_path: str | Path, type_model: TypeModels, device=None
):
    if type_model not in _models_option:
        raise ValueError(f"Choose one of the options: {_models_option}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        source_path = Path(root_path) / type_model
        model.load_state_dict(torch.load(source_path, map_location=device))
    except:
        try:
            source_path = Path(root_path) / f"{type_model}.pt"
            model.load_state_dict(torch.load(source_path, map_location=device))
        except:
            raise FileNotFoundError(
                f"File not found in {source_path} or {source_path}.pt"
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


def compute_distances(
    X: torch.Tensor,
    Y: torch.Tensor,
):
    """
    Computes the distances of the tensors X and Y, where their dimensions
    are (batch_size, z_dim).

    Returns three distances matrices, where the first one is asociated with X,
    the second one with Y and the third one is the distance between X and Y.
    Each distance matrix has dimensions (batch_size, batch_size).
    """
    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    return dists_x, dists_y, dists_c


def imq_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    scale: float = 1.0,
    p_distr: str = "norm",
):
    """
    Computes the MMD using the inverse multiquadratics kernel.
    Code inspired from https://github.com/schelotto/Wasserstein-AutoEncoders/blob/master/wae_mmd.py#L120

    X, Y (tensors): Tensors with size (batch_size, z_dim, 1, 1)
    """
    X, Y = X.squeeze(2, 3), Y.squeeze(2, 3)
    device = X.device
    sigma2_p = scale**2
    nf, z_dim = X.shape

    dists_x, dists_y, dists_c = compute_distances(X, Y)

    # Compute MMD using imq kernel
    match p_distr:
        case "norm":
            Cbase = 2.0 * z_dim * sigma2_p
        case "unif":
            Cbase = z_dim

    to_return = 0.0
    for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        C = Cbase * scale  # According to the paper

        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)
        res1 *= 1 - torch.eye(nf).to(device)
        res1 = res1.sum() / (nf * nf - nf)

        res2 = C / (C + dists_c)
        res2 = 2.0 * res2.sum() / (nf * nf)

        to_return += res1 - res2

    return to_return


def rbf_kernel(
    X: torch.Tensor,
    Y: torch.Tensor,
    scale: float = 1.0,
):
    """
    Computes the MMD using the RBF.
    Code inspired from https://github.com/schelotto/Wasserstein-AutoEncoders/blob/master/wae_mmd.py#L120

    X, Y (tensors): Tensors with size (batch_size, z_dim, 1, 1)
    """
    X, Y = X.squeeze(2, 3), Y.squeeze(2, 3)
    sigma2_p = scale**2
    device = X.device
    n, z_dim = X.shape

    dists_x, dists_y, dists_c = compute_distances(X, Y)

    # Compute MMD using rbf kernel
    res1 = torch.exp(-dists_x / sigma2_p)
    res1 += torch.exp(-dists_y / sigma2_p)
    res1 *= 1 - torch.eye(n).to(device)
    res1 = res1.sum() / (n * n - n)

    res2 = torch.exp(-dists_c / sigma2_p)
    res2 = 2.0 * res2.sum() / (n * n)

    to_return = res1 - res2

    return to_return


@torch.jit.script
def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


@torch.jit.script
def _sqrtm_newton_schulz(
    matrix: torch.Tensor, num_iters: int = 100
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    expected_num_dims = 2
    if matrix.dim() != expected_num_dims:
        raise ValueError(
            f"Input dimension equals {matrix.dim()}, expected {expected_num_dims}"
        )

    if num_iters <= 0:
        raise ValueError(
            f"Number of iteration equals {num_iters}, expected greater than 0"
        )

    dim = matrix.size(0)
    device, dtype = matrix.device, matrix.dtype
    norm_of_matrix = torch.norm(matrix)
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(n=dim, device=device, dtype=dtype)
    Z = torch.eye(n=dim, device=device, dtype=dtype)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1, device=device, dtype=dtype)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.zeros_like(error), atol=1e-5):
            break

    return s_matrix, error


def bures_wass_dist(m, C):
    n = m.shape[0]
    C12, _ = _sqrtm_newton_schulz(C, num_iters=5)
    # B(s, t)^2 = trace(Cs + Ct - 2 * sqrt(sqrt(Cs) * Ct * sqrt(Cs)))
    # If Ct = Id then B(s, t)^2 = trace(Cs) + n - 2 * trace(sqrt(Cs))
    B2 = torch.trace(C) + n - 2 * torch.trace(C12)
    dist = torch.norm(m) ** 2 + B2
    # return torch.sqrt(torch.maximum(dist, torch.zeros_like(dist))) / n
    return dist / n**2


class ProjectorOnManifold:
    def __init__(
        self,
        encoder,
        generator,
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
        self._transform_in = transform_in or transforms.Compose(
            [
                # From pdf to grayscale
                transforms.Lambda(lambda x: x / torch.max(x)),
                # transforms.Lambda(lambda x: x),
                transforms.ToPILImage(),
                transforms.Resize(image_size_net[1:]),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(channels_img)],
                    [0.5 for _ in range(channels_img)],
                ),
            ]
        )
        self._transform_out = transform_out or transforms.Compose(
            [
                # Ensure the range is in [0, 1]
                transforms.Lambda(lambda x: x - torch.min(x)),
                transforms.Lambda(lambda x: x / torch.max(x)),
                transforms.ToPILImage(),
                transforms.Resize(image_size[1:]),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x / torch.sum(x)),
            ]
        )

    @torch.no_grad()
    def forward(self, x):
        x = self._transform_in(x).to(x.device)
        x = torch.unsqueeze(x, 0)
        x = self.encoder(x)
        x = self.generator(x)
        x = torch.squeeze(x)
        x = self._transform_out(x)
        return x

    def __call__(self, x):
        return self.forward(x)
