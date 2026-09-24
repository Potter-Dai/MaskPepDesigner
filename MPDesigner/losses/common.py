import torch
from unicore.utils import one_hot

from unifold.losses.utils import softmax_cross_entropy

def _gaussian_smooth_kernel(
    num_bins: int,
    cutoff: int,
    sigma: float,
    dtype: torch.dtype = None,
    device: torch.device = None,
):
    """
    return N x N tensor.
    i-th row stores the normalized gaussian-smoothed probabilities given index i
    """
    def gauss(x):
        return torch.exp(-x**2 / (2*sigma))

    pos = torch.arange(num_bins)
    relpos = pos[None] - pos[:, None]
    fx = torch.where(
        relpos.abs() <= cutoff,
        gauss(relpos),
        0
    )
    fx = fx / fx.sum(dim=-1, keepdim=True)
    fx = fx.to(dtype=dtype, device=device)
    return fx


def gaussian_smooth(
    bin_index: torch.Tensor,    #
    num_bins: int,
    smooth_factor: float,   # 0: do not smooth; 1: smoothed only.
    bin_cutoff: int = 2,        # in the unit of bin_width.
    sigma: float = 1.0,     # in the unit of bin_width.
) -> torch.Tensor:
    dtype, device = bin_index.dtype, bin_index.device
    kernel = _gaussian_smooth_kernel(num_bins, bin_cutoff, sigma, dtype, device)
    bin_1h = one_hot(bin_index, num_bins)
    smoothed_labels = (
        bin_1h * (1. - smooth_factor) +
        kernel[bin_index] * smooth_factor
    )
    return smoothed_labels


def uniform_smooth(
    bin_index: torch.Tensor,
    num_bins: int,
    smooth_factor: float,
) -> torch.Tensor:
    dtype, device = bin_index.dtype, bin_index.device
    bin_1h = one_hot(bin_index, num_bins)
    smoothed_labels = (
        bin_1h * (1. - smooth_factor) +
        torch.ones_like(bin_1h) / num_bins * smooth_factor
    )
    return smoothed_labels


def related_smooth(
    bin_index: torch.Tensor,
    num_bins: int,
    smooth_factor: float,
    bin_correlations: torch.Tensor,
    eps: float = 1e-6,
):
    assert bin_correlations == (num_bins, num_bins), bin_correlations.shape
    assert torch.all(bin_correlations >= 0.), bin_correlations

    kernel = bin_correlations / bin_correlations.sum(dim=-1, keepdim=True).clamp_min(eps)
    bin_1h = one_hot(bin_index, num_bins)
    smoothed_labels = (
        bin_1h * (1. - smooth_factor) +
        kernel[bin_index] * smooth_factor
    )
    return smoothed_labels


SMOOTH_FUNCS = {
    "gaussian": gaussian_smooth,
    "uniform": uniform_smooth,
    "related": related_smooth,
}


def smoothed_softmax_cross_entropy(
    logits: torch.Tensor,
    label_bins: torch.Tensor,
    num_bins: int,
    smooth_factor: float,
    method: str = "uniform",
    **smooth_kwargs
) -> torch.Tensor:
    smooth_fn = SMOOTH_FUNCS[method]
    smoothed_labels = smooth_fn(
        label_bins,
        num_bins,
        smooth_factor,
        **smooth_kwargs
    )
    return softmax_cross_entropy(logits, smoothed_labels)


def per_layer_loss_weight(method: str, num_layers: int, **factory_kwargs):
    ret = torch.arange(1, num_layers+1, **factory_kwargs)
    if method == "uniform":
        return torch.ones((num_layers,), **factory_kwargs) / num_layers
    elif method == "linear":
        return ret / ret.sum()
    elif method == "sqrt":
        ret = ret.sqrt()
        return ret / ret.sum()
    elif method == "square":
        ret = ret.square()
        return ret / ret.sum()
    elif method == "sqrt_exp":
        ret = ret.sqrt().exp()
        return ret / ret.sum()
    elif method == "exp":
        ret = (1.1 ** ret)
        return ret / ret.sum()
    elif method == "last":
        ret = torch.zeros(num_layers, **factory_kwargs)
        ret[-1] = 1.
    return ret


def layer_averaged_loss(per_layer_loss: torch.Tensor, method: str, stack_dim: int = 0):
    loss = per_layer_loss.transpose(stack_dim, -1)
    weight = per_layer_loss_weight(
        method, loss.shape[-1],
        dtype=loss.dtype,
        device=loss.device
    )
    loss = (loss * weight).sum(dim=-1)
    return loss


def display_per_layer_loss(per_layer_loss: torch.Tensor, key: str):
    # stack dim must be 0
    num_layers = per_layer_loss.shape[0]
    to_show = {
        "init": 0,
        "quarter": (num_layers - 1) // 4,
        "half": (num_layers - 1) // 2,
        "last": num_layers - 1
    }
    ret = {
        f"z_{key}_{k}": per_layer_loss[i].data for k, i in to_show.items()
    }
    return ret
