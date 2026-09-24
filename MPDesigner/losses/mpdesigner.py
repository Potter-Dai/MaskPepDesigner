import torch
import logging
logger = logging.getLogger(__name__)

from MPDesigner.data.db import decompress_uid

from .auxillary import (
    distogram_loss,
    plddt_loss,
    pae_loss,
    sequence_denoise_loss_gaussian,
    masked_mean
)
from .structure import (
    backbone_clash_loss,
    adjacent_ca_violation_loss,
    compute_ca_metrics,
    structure_loss_v6,
    structure_loss_v65,
)

def mpdesigner_loss(out, batch, cfg):
    """Compute the loss for the given sample.

    Returns a tuple with three elements:
    1) the loss
    2) the sample size, which is used as the denominator for the gradient
    3) logging outputs to display while training
    """
    # plddt cutoff
    if "frame_mask" in batch and "bfactor" in batch:
        loss_frame_mask = (batch["bfactor"] > cfg.plddt_cutoff).float() * batch["frame_mask"]
    if "frame_gen_mask" in batch and "bfactor" in batch:
        loss_frame_gen_mask = (batch["bfactor"] > cfg.plddt_cutoff).float() * batch["frame_gen_mask"] * batch["frame_mask"]
    if "frames" in out:
        pred_ca = out["frames"][..., :3, 3]
    with torch.no_grad():
        bsz = batch["seq_mask"].shape[0]
        sample_size = 1
        seq_len = batch["seq_mask"].sum(dim=-1)
        sample_weight = seq_len.sqrt()
        has_residue_pct = masked_mean(batch["seq_mask"], loss_frame_mask, dim=-1)
        log = {
            "seq_len": seq_len,
            "bfactor": has_residue_pct,
            "sample_weight": sample_weight,
        }

    # if "violation" not in out and cfg.violation.weight:
    #     out["violation"] = find_structural_violations(
    #         batch, out["sm"]["atom14_positions"], **cfg.violation)

    # if "renamed_atom14_gt_positions" not in batch:
    #     batch.update(
    #         compute_renamed_ground_truth(batch, out["sm"]["atom14_positions"]))

    loss_dict = log
    loss_fns = {
        # "structure": lambda: structure_loss_v6(
        #     out["frames"],
        #     batch["frames"],
        #     batch["diffused_frames"],
        #     loss_frame_gen_mask if cfg.structure.gen_region_only else loss_frame_mask,
        #     **cfg.structure,
        #     loss_dict=loss_dict,
        # ),
        "structure": lambda: structure_loss_v65(
            out["frames"],
            batch["frames"],
            batch["diffused_frames"],
            loss_frame_mask,
            loss_frame_gen_mask,
            batch["diffusion_t"],
            **cfg.structure,
            loss_dict=loss_dict,
        ),
        "backbone_clash": lambda: backbone_clash_loss(
            pred_ca=pred_ca,
            diffusion_t=batch["diffusion_t"],
            seq_mask=batch["seq_mask"],
            **cfg.backbone_clash,
            loss_dict=loss_dict,
        ),
        "adjacent_ca": lambda: adjacent_ca_violation_loss(
            pred_ca=pred_ca,
            relative_positions=batch["relative_positions"],
            diffusion_t=batch["diffusion_t"],
            **cfg.adjacent_ca,
            loss_dict=loss_dict,
        ),
        "sequence": lambda: sequence_denoise_loss_gaussian(
            pred_aalogits=out["aalogits"],
            true_aalogits=batch["aalogits"],
            diffused_aalogits=batch["diffused_aalogits"],
            seq_mask=batch["seq_mask"],
            seq_gen_mask=batch["seq_gen_mask"],
            **cfg.sequence,
            loss_dict=loss_dict
        ),
        "distogram": lambda: distogram_loss(
            logits=out["distogram_logits"],
            true_distogram=batch["distogram"],
            frame_mask=loss_frame_mask,
            **cfg.distogram,
            loss_dict=loss_dict,
        ),
        # "supervised_chi": lambda: supervised_chi_loss(
        #     pred_angles_sin_cos=out["sm"]["angles"],
        #     pred_unnormed_angles_sin_cos=out["sm"]["unnormalized_angles"],
        #     true_angles_sin_cos=batch["torsion_angles_sin_cos"],
        #     aatype=batch["aatype"],
        #     bfactor=batch["bfactor"],
        #     torsion_angle_mask=batch["torsion_angles_mask"],
        #     seq_mask=batch["seq_mask"],
        #     **cfg.supervised_chi,
        #     loss_dict=loss_dict,
        # ),
        "predicted_lddt": lambda: plddt_loss(
            logits=out["plddt_logits"],
            pred_ca=pred_ca[-1, ...],   # use last-layer pred_ca
            true_ca=batch["frames"][..., :3, 3],
            frame_mask=loss_frame_mask,
            resolution=None,
            **cfg.predicted_lddt,
            loss_dict=loss_dict,
        ),
        "predicted_aligned_error": lambda: pae_loss(
            logits=out["pae_logits"],
            pred_frames=out["frames"][-1],
            true_frames=batch["frames"],
            frame_mask=loss_frame_mask,
            resolution=None,
            **cfg.predicted_aligned_error,
            loss_dict=loss_dict,
        ),
        # "violation": lambda: violation_loss(
        #     out["violation"],
        #     **cfg.violation,
        #     loss_dict=loss_dict,
        # ),
    }

    cum_loss = 0
    for loss_name, loss_fn in loss_fns.items():
        weight = cfg[loss_name].weight
        time_scaling = cfg[loss_name].get("t_scale", 1.)
        if weight > 0.:
            loss = loss_fn()
            # always use float type for loss
            assert loss.dtype == torch.float, loss.dtype
            assert len(loss.shape) == 1 and loss.shape[0] == bsz, (loss_name, loss.shape, bsz)

            if any(torch.isnan(loss)) or any(torch.isinf(loss)):
                uid = decompress_uid(batch["uid"][0].cpu().numpy())
                logger.warning(f"{loss_name} loss for {uid} is NaN. Skipping...")
                loss = loss.new_tensor(0.0, requires_grad=True)

            # diffusion_t=batch["diffusion_t"].float()
            diffusion_t=batch["diffusion_t"]
            time_scaling_weight = time_scaling ** diffusion_t
            scaled_loss = time_scaling_weight * loss
            if time_scaling != 1:
                loss_dict["scaled_" + loss_name] = scaled_loss.data
            cum_loss = cum_loss + weight * scaled_loss

    normed_loss = cum_loss.mean()
    loss = (cum_loss * sample_weight).mean()

    with torch.no_grad():
        for k in ("plddt", "ptm"):
            if k in out:
                log[k] = out[k]
        if "frames" in out:
            log.update(compute_ca_metrics(
                pred_ca=pred_ca,
                true_ca=batch["frames"][..., :3, 3],
                frame_gen_mask=loss_frame_gen_mask,
                frame_mask=loss_frame_mask,
                layer_weighting=cfg.structure.layer_weighting,
                eps=1e-12,
            ))

    for key, val in loss_dict.items():
        assert isinstance(val, torch.Tensor), (key, type(val))
        loss_dict[key] = float((loss_dict[key]).float().mean()) # to float

    log.update({
        "loss": loss.data,
        "normed_loss": normed_loss.data,
        "bsz": bsz,
        "sample_size": sample_size,
    })

    return loss, sample_size, log

# Compatibility names for existing downstream code.
dfv6_loss = mpdesigner_loss
