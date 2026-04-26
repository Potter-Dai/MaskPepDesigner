import torch
import logging
logger = logging.getLogger(__name__)

from unifold.losses.geometry import compute_renamed_ground_truth, compute_metric
from unifold.losses.violation import find_structural_violations, violation_loss

from diffold.data.db import decompress_uid

from .auxillary import (
    distogram_loss,
    supervised_chi_loss,
    plddt_loss,
    pae_loss,
    sequence_denoise_loss_v3,
    masked_mean
)
from .structure import (
    structure_loss,
    backbone_clash_loss,
    compute_ca_metrics
)

def diffold_loss(out, batch, cfg):
    """Compute the loss for the given sample.

    Returns a tuple with three elements:
    1) the loss
    2) the sample size, which is used as the denominator for the gradient
    3) logging outputs to display while training
    """
    with torch.no_grad():
        uid = decompress_uid(batch["uid"][0].cpu().numpy())
        bsz = batch["aatype"].shape[0]
        sample_size = 1
        seq_len = batch["seq_mask"].float().sum(dim=-1)
        sample_weight = seq_len.sqrt()
        bfactor = masked_mean(
            (batch["bfactor"] > 0).float(),
            batch["bfactor"],
            dim=-1,
        )
        log = {
            "seq_len": seq_len,
            "bfactor": bfactor,
            "sample_weight": sample_weight,
        }

    if "violation" not in out and cfg.violation.weight:
        out["violation"] = find_structural_violations(
            batch, out["sm"]["atom14_positions"], **cfg.violation)

    if "renamed_atom14_gt_positions" not in batch:
        batch.update(
            compute_renamed_ground_truth(batch, out["sm"]["atom14_positions"]))

    loss_dict = log
    loss_fns = {
        "structure": lambda: structure_loss(
            out, batch, cfg.structure, loss_dict,
        ),
        "backbone_clash": lambda: backbone_clash_loss(
            pred_ca=out["pred_frames_per_layer"][..., :3, 3],
            seq_mask=batch["seq_mask"],
            **cfg.backbone_clash,
            loss_dict=loss_dict,
        ),
        "sequence": lambda: sequence_denoise_loss_v3(
            logits=out["aux"]["stacked_seq_logits"],
            true_aatype=batch["aatype"],
            diffused_aatype=batch["diffused_aatype"],
            seq_mask=batch["seq_mask"],
            **cfg.sequence,
            loss_dict=loss_dict,
        ),
        "distogram": lambda: distogram_loss(
            logits=out["aux"]["distogram_logits"],
            true_distogram=batch["distogram"],
            frame_mask=batch["bfactor"],
            **cfg.distogram,
            loss_dict=loss_dict,
        ),
        "supervised_chi": lambda: supervised_chi_loss(
            pred_angles_sin_cos=out["sm"]["angles"],
            pred_unnormed_angles_sin_cos=out["sm"]["unnormalized_angles"],
            true_angles_sin_cos=batch["torsion_angles_sin_cos"],
            aatype=batch["aatype"],
            bfactor=batch["bfactor"],
            torsion_angle_mask=batch["torsion_angles_mask"],
            seq_mask=batch["seq_mask"],
            **cfg.supervised_chi,
            loss_dict=loss_dict,
        ),
        "predicted_lddt": lambda: plddt_loss(
            logits=out["aux"]["plddt_logits"],
            pred_ca=out["pred_frames"][..., :3, 3],
            true_ca=batch["frames"][..., :3, 3],
            frame_mask=batch["bfactor"],
            resolution=batch.get("resolution", None),
            **cfg.predicted_lddt,
            loss_dict=loss_dict,
        ),
        "predicted_aligned_error": lambda: pae_loss(
            logits=out["aux"]["pae_logits"],
            pred_frames=out["pred_frames"],
            true_frames=batch["frames"],
            frame_mask=batch["bfactor"],
            resolution=batch.get("resolution", None),
            **cfg.predicted_aligned_error,
            loss_dict=loss_dict,
        ),
        "violation": lambda: violation_loss(
            out["violation"],
            **cfg.violation,
            loss_dict=loss_dict,
        ),
    }

    cum_loss = 0
    for loss_name, loss_fn in loss_fns.items():
        weight = cfg[loss_name].weight
        if weight > 0.:
            loss = loss_fn()
            # always use float type for loss
            assert loss.dtype == torch.float, loss.dtype
            assert len(loss.shape) == 1 and loss.shape[0] == bsz, (loss_name, loss.shape, bsz)

            if any(torch.isnan(loss)) or any(torch.isinf(loss)):
                logger.warning(f"{loss_name} loss for {uid} is NaN. Skipping...")
                loss = loss.new_tensor(0.0, requires_grad=True)

            cum_loss = cum_loss + weight * loss

    with torch.no_grad():
        log.update({
            "plddt": out["aux"]["plddt"],
            "ptm": out["aux"]["ptm"],
        })
        log.update(compute_ca_metrics(
            pred_ca=out["pred_frames_per_layer"][..., :3, 3],
            true_ca=batch["frames"][..., :3, 3],
            frame_mask=batch["bfactor"],
            eps=1e-6,
        ))

    for key, val in loss_dict.items():
        assert isinstance(val, torch.Tensor), (key, type(val))
        loss_dict[key] = float((loss_dict[key]).float().mean()) # to float

    normed_loss = cum_loss.mean()
    loss = (cum_loss * sample_weight).mean()

    log.update({
        "loss": loss.data,
        "normed_loss": normed_loss.data,
        "bsz": bsz,
        "sample_size": sample_size,
    })

    return loss, sample_size, log
