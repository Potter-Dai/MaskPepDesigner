import torch

from typing import *
from unicore.utils import masked_mean, one_hot

import unifold.data.residue_constants as rc
from unifold.losses.geometry import (
    compute_aligned_error,
    compute_lddt,
)

from .common import (
    softmax_cross_entropy,
    smoothed_softmax_cross_entropy,
    layer_averaged_loss,
    display_per_layer_loss,
)


def distogram_loss(
    logits,
    true_distogram,
    frame_mask,
    num_bins: int,
    smooth_factor: float = 0.,
    smooth_bin_cutoff: int = 2,
    smooth_sigma: float = 1.,
    eps: float = 1e-10,
    loss_dict: dict = None,
    **kwargs,
):
    errors = smoothed_softmax_cross_entropy(
        logits, true_distogram,
        num_bins=num_bins,
        smooth_factor=smooth_factor,
        method="gaussian",
        bin_cutoff=smooth_bin_cutoff,
        sigma=smooth_sigma,
    )

    pair_mask = frame_mask[..., None] * frame_mask[..., None, :]

    loss = masked_mean(pair_mask, errors, dim=(-1, -2), eps=eps)

    loss_dict["distogram"] = loss.data

    return loss


def supervised_chi_loss(
    pred_angles_sin_cos: torch.Tensor,
    pred_unnormed_angles_sin_cos: torch.Tensor,
    true_angles_sin_cos: torch.Tensor,
    aatype: torch.Tensor,
    bfactor: torch.Tensor,
    torsion_angle_mask: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps: float,
    loss_dict: dict = {},
    **kwargs,
) -> torch.Tensor:
    # TODO: refactor this.
    pred_angles_sin_cos = pred_angles_sin_cos.float()
    pred_unnormed_angles_sin_cos = pred_unnormed_angles_sin_cos.float()
    true_angles_sin_cos = true_angles_sin_cos.float()
    seq_mask = seq_mask.float()
    # chi_mask = chi_mask.float()

    pred_bb, pred_chi = torch.split(pred_angles_sin_cos, (3, 4), dim=-2)    # [*, L, 3/4, 2]
    true_bb, true_chi = torch.split(true_angles_sin_cos, (3, 4), dim=-2)
    bb_mask, chi_mask = torch.split(torsion_angle_mask, (3, 4), dim=-1)     # [*, L, 3/4]

    restype_1h = one_hot(aatype, rc.restype_num + 1)
    chi_pi_periodic = pred_angles_sin_cos.new_tensor(
        rc.chi_pi_periodic, requires_grad=False,
    )
    chi_pi_periodic = restype_1h @ chi_pi_periodic

    true_chi_shifted = true_chi * (
        1.0 - 2.0 * chi_pi_periodic[..., None]
    )

    chi_err = (true_chi - pred_chi).square().sum(dim=-1)
    chi_err_shifted = (true_chi_shifted - pred_chi).square().sum(dim=-1)
    chi_err = torch.minimum(chi_err, chi_err_shifted)   # [*, L, 4]
    chi_loss = masked_mean(chi_mask * bfactor[..., None] , chi_err, dim=(-1, -2), eps=eps)   # [*]

    bb_err = (true_bb - pred_bb).square().sum(dim=-1)
    bb_loss = masked_mean(bb_mask * bfactor[..., None], bb_err, dim=(-1, -2), eps=eps)       # [*]

    angle_norm = pred_unnormed_angles_sin_cos.norm(dim=-1)  # [*, L, 7]
    norm_error = torch.abs(angle_norm - 1.0)
    angle_norm_loss = masked_mean(seq_mask[..., None], norm_error, dim=(-1, -2), eps=eps)

    loss = (
        chi_weight * chi_loss + 
        (1. - chi_weight) * bb_loss + 
        angle_norm_weight * angle_norm_loss
    )

    loss_dict.update({
        "chi_loss": chi_loss.data,
        "backb_torsion_loss": bb_loss.data,
        "angle_norm_loss": angle_norm_loss.data
    })

    return loss


def plddt_loss(
    logits: torch.Tensor,
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    frame_mask: torch.Tensor,
    resolution: torch.Tensor = None,
    num_bins: int = 50,
    cutoff: float = 15.0,
    smooth_factor: float = 0.1,
    smooth_bin_cutoff: int = 2,
    smooth_sigma: float = 1.,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
    loss_dict: dict = {},
    **kwargs,   # escape weight
) -> torch.Tensor:
    lddt = compute_lddt(
        pred_ca, true_ca, frame_mask[..., None], cutoff=cutoff, eps=eps
    ).detach()  # expand as 1-atom mask to satisfy compute_lddt

    bin_index = torch.floor(lddt * num_bins).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))

    errors = smoothed_softmax_cross_entropy(
        logits,
        bin_index,
        num_bins,
        smooth_factor=smooth_factor,
        method="gaussian",
        bin_cutoff=smooth_bin_cutoff,
        sigma=smooth_sigma
    )

    loss = masked_mean(frame_mask, errors, dim=-1, eps=eps)

    if resolution is not None:
        res = resolution.squeeze(-1)
        loss *= ((res >= min_resolution) & (res <= max_resolution)).type(loss.dtype)

    ca_lddt_mean = masked_mean(frame_mask, lddt, dim=-1, eps=eps)

    loss_dict["ca_lddt_score"] = ca_lddt_mean.data
    loss_dict["plddt_loss"] = loss.data

    return loss


def pae_loss(
    logits: torch.Tensor,
    pred_frames: torch.Tensor,
    true_frames: torch.Tensor,
    frame_mask: torch.Tensor,
    resolution: torch.Tensor = None,
    num_bins: int = 64,
    max_bin: float = 31.,
    smooth_factor: float = 0.1,
    smooth_bin_cutoff: int = 2,
    smooth_sigma: float = 1.,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-8,
    loss_dict: dict = {},
    **kwargs,
):
    aligned_error_vals, aligned_error_bins, mask = compute_aligned_error(
        pred_frames,
        true_frames,
        frame_mask,
        max_bin,
        num_bins,
    )

    errors = smoothed_softmax_cross_entropy(
        logits, aligned_error_bins,
        num_bins=num_bins,
        smooth_factor=smooth_factor,
        method="gaussian",
        bin_cutoff=smooth_bin_cutoff,
        sigma=smooth_sigma,
    )

    loss = masked_mean(mask, errors, dim=(-1, -2), eps=eps)
    mean_aligned_err = masked_mean(mask, aligned_error_vals, dim=(-1, -2), eps=eps)

    if resolution is not None:
        res = resolution.squeeze(-1)
        loss *= ((res >= min_resolution) & (res <= max_resolution)).type(loss.dtype)

    loss_dict["pae_loss"] = loss.data
    loss_dict["aligned_error"] = mean_aligned_err.data

    return loss


def accuracy_on_logits(
    pred_logits: torch.Tensor,          # [*s, L, K]
    true_logits: torch.Tensor,          # [*, L, K]
    seq_mask: torch.Tensor,                 # [*, L]
    stack_dim: int = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    if stack_dim is not None:
        pred_logits = pred_logits.transpose(stack_dim, 0)    # [s, *, L, K]
        true_logits = true_logits.unsqueeze(0)               # [1, *, L, K]
        seq_mask = seq_mask.unsqueeze(0)       # [1, *, L]
    is_correct = (pred_logits.argmax(dim=-1) == true_logits.argmax(dim=-1))
    acc = masked_mean(seq_mask, is_correct.float(), dim=-1, eps=eps)
    if stack_dim is not None:
        acc = acc.transpose(stack_dim, 0)
    return acc


def sequence_xent(
    logits: torch.Tensor,       # [*s, L, K]
    true_logits: torch.Tensor,  # [*, L, K]
    seq_mask: torch.Tensor,     # [*, L]
    stack_dim: int = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    labels = true_logits.softmax(dim=-1)
    if stack_dim is not None:
        logits = logits.transpose(stack_dim, 0)    # [s, *, L, K]
        labels = labels.unsqueeze(0)               # [1, *, L, K]
        seq_mask = seq_mask.unsqueeze(0)       # [1, *, L]
    xent = softmax_cross_entropy(logits, labels)    # [s, *, L]
    loss = masked_mean(seq_mask, xent, dim=-1, eps=eps)     # [s, *]
    return loss


def sequence_denoise_loss(
    logits: torch.Tensor,               # [s, *, L, K]
    true_logits: torch.Tensor,          # [*, L, K]
    diffused_logits: torch.Tensor,      # [*, L, K]
    seq_mask: torch.Tensor,             # [*, L]
    layer_weighting: str = "linear",
    eps: float = 1e-6,
    loss_dict: dict = {},
    **kwargs
):
    seq_mask = seq_mask.float()
    per_layer_xent = sequence_xent(
        logits, true_logits, seq_mask,
        stack_dim=0, eps=eps,
    )
    per_layer_acc = accuracy_on_logits(
        logits, true_logits, seq_mask,
        stack_dim=0, eps=eps
    )

    loss_dict.update({
        **display_per_layer_loss(per_layer_acc, key="sequence_acc"),
        **display_per_layer_loss(per_layer_xent, key="sequence_cse")
    })

    loss = layer_averaged_loss(per_layer_xent, method=layer_weighting, stack_dim=0)

    init_xent = sequence_xent(
        diffused_logits, true_logits, seq_mask, eps=eps,
    )
    init_acc = accuracy_on_logits(
        diffused_logits, true_logits, seq_mask, eps=eps
    )

    loss_dict.update({
        "improve_seq_recon_cse": (per_layer_xent[-1] - init_xent).data,
        "improve_seq_recon_acc": (per_layer_acc[-1] - init_acc).data,
        "sequence_loss": loss.data
    })

    return loss


def accuracy(
    pred_aatype: torch.Tensor,          # [*, L, K]
    true_aatype: torch.Tensor,          # [*, L]
    seq_mask: torch.Tensor,             # [*, L]
    eps: float = 1e-10,
) -> torch.Tensor:
    is_correct = (pred_aatype == true_aatype)
    acc = masked_mean(seq_mask, is_correct.float(), dim=-1, eps=eps)
    return acc


def perplexity(
    pred_logits: torch.Tensor,
    true_aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    true_aatype_1h = one_hot(true_aatype, pred_logits.shape[-1])
    xent = softmax_cross_entropy(pred_logits, true_aatype_1h)
    mean_xent = masked_mean(seq_mask, xent, dim=-1, eps=eps)
    return 2. ** mean_xent


def sequence_denoise_loss_v3(
    logits: torch.Tensor,               # [s, *, L, K]
    true_aatype: torch.Tensor,          # [*, L, K]
    diffused_aatype: torch.Tensor,      # [*, L, K]
    seq_mask: torch.Tensor,             # [*, L]
    smooth_factor: float,
    layer_weighting: str = "linear",
    eps: float = 1e-6,
    loss_dict: dict = {},
    **kwargs
):
    num_classes = logits.shape[-1]
    seq_mask = seq_mask.float()
    per_layer_xent = smoothed_softmax_cross_entropy(
        logits,
        true_aatype[None],
        num_bins=num_classes,
        smooth_factor=smooth_factor,
    )
    per_layer_xent = masked_mean(seq_mask, per_layer_xent, dim=-1, eps=eps)
    per_layer_acc = accuracy(
        logits.argmax(dim=-1), true_aatype, seq_mask, eps=eps
    )
    pred_perplex = perplexity(logits[-1], true_aatype, seq_mask, eps)

    loss_dict.update({
        **display_per_layer_loss(per_layer_acc, key="sequence_acc"),
        **display_per_layer_loss(per_layer_xent, key="sequence_cse")
    })

    loss = layer_averaged_loss(per_layer_xent, method=layer_weighting, stack_dim=0)

    init_acc = accuracy(
        diffused_aatype, true_aatype, seq_mask, eps
    )

    loss_dict.update({
        "improve_seq_recon_acc": (per_layer_acc[-1] - init_acc).data,
        "sequence_loss": loss.data,
        "perplexity": pred_perplex.data
    })

    return loss


def sequence_denoise_loss_gaussian(
    pred_aalogits: torch.Tensor,        # [*, L, K]
    true_aalogits: torch.Tensor,        # [*, L, K]
    diffused_aalogits: torch.Tensor,      # [*, L, K]
    seq_mask: torch.Tensor,             # [*, L]
    seq_gen_mask: torch.Tensor,         # [*, L]
    motif_weight: float,
    eps: float = 1e-8,
    loss_dict: dict = {},
    **kwargs
):
    seq_mask = motif_weight * seq_mask + (1. - motif_weight) * seq_gen_mask
    pred_aalogits = pred_aalogits.float()
    loss = (pred_aalogits - true_aalogits).norm(dim=-1)
    loss = masked_mean(seq_mask, loss, dim=-1, eps=eps)

    acc = (torch.argmax(pred_aalogits, dim=-1) == torch.argmax(true_aalogits, dim=-1)).float()
    acc = masked_mean(seq_mask, acc, dim=-1, eps=eps)

    init_acc = (torch.argmax(diffused_aalogits, dim=-1) == torch.argmax(true_aalogits, dim=-1)).float()
    init_acc = masked_mean(seq_mask, init_acc, dim=-1, eps=eps)

    init_loss = (diffused_aalogits - true_aalogits).norm(dim=-1)
    init_loss = masked_mean(seq_mask, init_loss, dim=-1, eps=eps)

    loss_dict.update({
        "improve_seq_recon_acc": (acc - init_acc).data,
        "improve_seq_loss": (loss - init_loss).data,
        "sequence_loss": loss.data,
        "sequence_acc": acc.data
    })

    return loss
