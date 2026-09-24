import torch

from unicore.utils import masked_mean

from unifold.modules.frame import Frame, Quaternion, Rotation

from .common import layer_averaged_loss, display_per_layer_loss

from unifold.losses.fape import compute_fape

def _make_local_coords(
    frames: torch.Tensor,
    reference_points: torch.Tensor = None,  # [*, nref, 3]
) -> torch.Tensor:
    """
    unit: to normalize frame translations.
    reference_points: internal reference points for each batch, in the scale
        of given unit. if set none, use I instead.
    """
    # [*, 3/1=nref, 3]
    rot_T, trans = frames[..., :3, :].transpose(-1, -2).split((3, 1), dim=-2)
    if reference_points is not None:
        pts = reference_points[..., None, :, :]     # [*, 1, nref, 3]
        ret = pts @ rot_T + trans   # [*, L, nref, 3]
    else:
        ret = rot_T + trans         # [*, L, nref, 3]
    return ret      # [*, L, nref, 3]


def compute_frame_loss(
    pred_frames: Frame,     # [*s, L, :]
    true_frames: Frame,     # [*, L, :]
    bfactor: torch.Tensor,  # [*, L]
    reference_pts: torch.Tensor = None,   # [*, nref, 3]
    loss_unit: float = 10.,
    clamp_distance: float = 10.,
    stack_dim: int = None,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:          # [*s]
    pred_coords = _make_local_coords(
        pred_frames, reference_pts
    ) / loss_unit           # [*s, L, nref, 3]
    true_coords = _make_local_coords(
        true_frames, reference_pts
    ) / loss_unit           # [*, L, nref, 3]
    if stack_dim is not None:
        # match stack dim to 0
        pred_coords = pred_coords.transpose(0, stack_dim)   # [s, *, L, nref, 3]
        true_coords = true_coords.unsqueeze(0)              # [1, *, L, nref, 3]
        bfactor = bfactor.unsqueeze(0)                      # [1, *, L]
    err = (
        pred_coords - true_coords
    ).norm(dim=-1).clamp_max(clamp_distance / loss_unit)    # [s, *, L], in-frame rmsd
    err = err.mean(dim=-1)
    loss = masked_mean(bfactor, err, dim=-1, eps=eps)       # [s, *]
    # match stack dim back
    if stack_dim is not None:
        loss = loss.transpose(0, stack_dim)                     # [*s]
    return loss


def structure_loss(
    out, batch, config, loss_dict: dict = {}
) -> torch.Tensor:
    bfactor = batch["bfactor"]
    sc_frame_mask = batch["rigidgroups_gt_exists"]          # [*, L, 8, 4, 4]
    sc_frame_mask *= bfactor[..., None]
    sc_position_mask = batch["renamed_atom14_gt_exists"]    # [*, L, 14, 3]
    sc_position_mask *= bfactor[..., None]

    bb_err = compute_frame_loss(
        out["pred_frames_per_layer"],
        batch["frames"],
        bfactor,
        stack_dim=0,
        reference_pts=None,
        **config.backbone
    )   # [s, *]
    bb_loss = layer_averaged_loss(
        bb_err,
        config.backbone.layer_weighting,
        stack_dim=0
    )   # [*]
    loss_dict.update({
        "last_bb_loss": bb_err[-1].data,
    })

    sc_loss = 0.
    if config.sidechain.weight:
        sc_rot, sc_pos = sidechain_loss(
            out["sm"]["sidechain_frames"],
            batch["rigidgroups_gt_frames"],
            batch["rigidgroups_alt_gt_frames"],
            sc_frame_mask,
            out["sm"]["atom14_positions"],
            batch["renamed_atom14_gt_positions"],
            sc_position_mask,
            batch["alt_naming_is_better"],
            **config.sidechain
        )
        loss_dict.update({
            "sc_rot": sc_rot.data,
            "sc_pos": sc_pos.data,
        })
        sc_loss = (
            sc_rot * config.sidechain.rot_weight +
            sc_pos * (1. - config.sidechain.rot_weight)
        )

    if config.calculate_improve:
        init_loss = compute_frame_loss(
            batch["diffused_frames"],
            batch["frames"],
            bfactor,
            **config.backbone,
        )
        loss_dict.update({
            "decoy_loss": init_loss.data,
            "improve_bb": (bb_err[-1] - init_loss).data
        })

    loss = (
        config.backbone.weight * bb_loss +
        config.sidechain.weight * sc_loss
    )
    loss_dict["structure"] = loss.data

    return loss


def sidechain_loss(
    sidechain_frames: torch.Tensor,             # [L, Nf, 4, 4]
    rigidgroups_gt_frames: torch.Tensor,        # [L, Nf, 4, 4]
    rigidgroups_alt_gt_frames: torch.Tensor,    # [L, Nf, 4, 4]
    rigidgroups_gt_exists: torch.Tensor,        # [L, Nf]
    sidechain_atom_pos: torch.Tensor,           # [L, 14, 3]
    renamed_atom14_gt_positions: torch.Tensor,  # [L, 14, 3]
    renamed_atom14_gt_exists: torch.Tensor,     # [L, 14]
    alt_naming_is_better: torch.Tensor,         # [L]
    loss_unit: float,
    clamp_distance: float,
    eps: float,
    **kwargs,
) -> torch.Tensor:
    renamed_gt_frames = torch.where(
        alt_naming_is_better[..., None, None, None] > 0,
        rigidgroups_alt_gt_frames,
        rigidgroups_gt_frames
    )

    batch_shape = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_shape, -1, 4, 4)
    renamed_gt_frames = renamed_gt_frames.view(*batch_shape, -1, 4, 4)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_shape, -1)
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_shape, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(*batch_shape, -1, 3)
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_shape, -1)

    rot_err = compute_rotation_error(
        sidechain_frames[..., :3, :3],
        renamed_gt_frames[..., :3, :3],
        rigidgroups_gt_exists,
        eps
    )

    pos_err = compute_atom_pos_error(
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        renamed_atom14_gt_exists,
        clamp_distance,
        loss_unit,
        eps
    )

    return rot_err, pos_err

def l2_err(a, b, mask, dim, eps, clamp=None):
    err_sq = ((a-b)**2).sum(dim=dim)
    err = (err_sq + eps).sqrt()
    if clamp is not None:
        err = err.clamp_max(clamp)
    return masked_mean(mask, err, dim=-1, eps=eps)

def compute_rotation_error(
    pred_rotations: torch.Tensor,   # [*, L, 3, 3]
    true_rotations: torch.Tensor,   # [*, L, 3, 3]
    mask: torch.Tensor,    # [*, L]
    eps=1e-8,
) -> torch.Tensor:
    return l2_err(pred_rotations, true_rotations, mask, dim=(-1, -2), eps=eps)


def compute_atom_pos_error(
    pred_atom_pos: torch.Tensor,    # [*, L, 3]
    true_atom_pos: torch.Tensor,    # [*, L, 3]
    mask: torch.Tensor,        # [*, L]
    length_scale: float,
    clamp_distance: float,
    eps=1e-8,
):
    err = l2_err(pred_atom_pos, true_atom_pos, mask, dim=-1, eps=eps, clamp=clamp_distance)
    return err / length_scale


def compute_ca_metrics(
    pred_ca: torch.Tensor,    # [s, *, L, 3]
    true_ca: torch.Tensor,              # [*, L, 3]
    frame_gen_mask: torch.Tensor,       # [*, L]
    frame_mask: torch.Tensor,              # [*, L]
    layer_weighting: str,
    eps: float = 1e-12,
):
    ca_sqdiff = (
        pred_ca - true_ca[None]
    ).square().sum(dim=-1)  # [s, *, L]

    rmsd = (masked_mean(
        frame_mask, ca_sqdiff, dim=-1, eps=eps
    ) + eps).sqrt()

    gen_rmsd = (masked_mean(
        frame_gen_mask, ca_sqdiff, dim=-1, eps=eps
    ) + eps).sqrt()

    mean_rmsd = layer_averaged_loss(rmsd, method=layer_weighting, stack_dim=0)
    mean_gen_rmsd = layer_averaged_loss(gen_rmsd, method=layer_weighting, stack_dim=0)

    # def tm_kernel(bfactor: torch.Tensor):
    #     nres = (bfactor > 0).float().sum(dim=-1, keepdim=True)
    #     clipped_n = nres.clamp_min(19)
    #     sq_d0 = (1.24 * (clipped_n - 15) ** (1.0 / 3.0) - 1.8) ** 2
    #     return lambda sq: 1.0 / (1.0 + sq / sq_d0)

    # tm_terms = tm_kernel(frame_mask)(ca_sqdiff)
    # tm_score = masked_mean(frame_mask, tm_terms, dim=-1, eps=eps)

    # return {
    #     **display_per_layer_loss(rmsd, "ca_rmsd"),
    #     **display_per_layer_loss(tm_score, "tm_score")
    # }

    return {
        "rmsd_last": rmsd[-1].data,
        "rmsd": mean_rmsd.data,
        "gen_rmsd_last": gen_rmsd[-1].data,
        "gen_rmsd": mean_gen_rmsd.data,
    }


def backbone_clash_loss(
    pred_ca: torch.Tensor,          # [s, *, L, 3]
    diffusion_t: torch.Tensor,      # [*,]
    seq_mask: torch.Tensor,         # [*, L]
    ca_ca_threshold: float,
    layer_weighting: str,
    t_scale: float,
    eps: float,
    loss_dict: dict = {},
    **kwargs,
):
    pred_dist = (
        pred_ca[..., None, :] -
        pred_ca[..., None, :, :]
    ).norm(dim=-1)
    # [s, *, L, L], normalized to [0, 1]
    ca_clash = (ca_ca_threshold - pred_dist).clamp_min(0.) / ca_ca_threshold

    # make masks
    nres = seq_mask.shape[-1]
    triu_mask = (
        seq_mask[..., None] *
        seq_mask[..., None, :] *
        torch.triu(
            torch.ones((nres, nres), dtype=torch.float, device=seq_mask.device),
            diagonal=1  # skip diagonal
        ).to(seq_mask.dtype)
    )   # [*, L, L]

    ca_clash *= triu_mask[None]

    # calculate loss only on pairs with clashes.
    has_clash = (ca_clash.detach() > 0.).float()
    num_clash = has_clash.sum(dim=(-1, -2)) # [s, *]

    # per_layer_loss = masked_mean(mask, ca_clash, dim=(-1, -2), eps=eps)   # [s, *]
    clash_loss = ca_clash.sum(dim=(-1, -2)) / (seq_mask.sum(dim=-1) + eps)  # use an empirical normalization of linear scale.

    loss = layer_averaged_loss(clash_loss, method=layer_weighting, stack_dim=0)

    # scale loss to be more reasonable
    loss = loss * 10

    loss_dict["clash"] = clash_loss[-1].data
    loss_dict["clash_loss"] = loss.data
    loss_dict["num_clash_last"] = num_clash[-1].data

    # if diffusion_t is not None and t_scale > 0.:
    #     loss *= (1. - diffusion_t) ** t_scale

    return loss


def adjacent_ca_violation_loss(
    pred_ca: torch.Tensor,    # [s, *, L, 3]
    relative_positions: torch.Tensor,             # [*, L]
    diffusion_t: torch.Tensor,      # [*,]
    min_distance: float = 3.2,
    max_distance: float = 4.0,
    min_weight: float = 3.0,
    clamp_distance: float = 4.0,
    layer_weighting: str = "linear",
    t_scale: float = 1.5,
    eps: float = 1e-6,
    loss_dict: dict = {},
    **kwargs,
):
    """
    an sliced-linear approximated LJ penalty over the between-residue distances.
    LJ(d)   = 0, d \in [a, b]
            = r(a - d), d \in [0, a)        (r > 1 is the `min_weight`)
            = (d - b), d \in (b, b + c]
            = c, d \in (b + c, inf)
    the values are then scaled to [0, 1].
    """
    pred_dist = (
        pred_ca[..., None, :] -
        pred_ca[..., None, :, :]
    ).norm(dim=-1)

    # [s, *, L, L], normalized to [0, 1]
    violations = (
        (min_distance - pred_dist).clamp_min(0.) * min_weight +
        (pred_dist - max_distance).clamp_min(0.)
    ).clamp_max(clamp_distance) / clamp_distance

    # make masks
    is_between = (relative_positions == 1).float()

    # calculate loss only on pairs with clashes.
    adj_ca = masked_mean(is_between, violations, dim=(-1, -2), eps=eps)   # [s, *]

    loss = layer_averaged_loss(adj_ca, method=layer_weighting, stack_dim=0)

    # scale loss to be more reasonable
    loss = loss * 10

    loss_dict["adjacent_ca"] = adj_ca[-1].data
    loss_dict["adjacent_ca_loss"] = loss.data

    # if diffusion_t is not None and t_scale > 0.:
    #     loss *= (1. - diffusion_t) ** t_scale

    return loss


def fape_v6(
    pred_frames: torch.Tensor,
    true_frames: torch.Tensor,
    diffused_frames: torch.Tensor,
    frame_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    loss_unit: float,
    clamp_distance: float,
    layer_weighting: str,
    loss_dict: dict = {},
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    pred_frames = Frame.from_tensor_4x4(pred_frames)
    true_frames = Frame.from_tensor_4x4(true_frames[None])
    diffused_frames = Frame.from_tensor_4x4(diffused_frames[None])
    pred_positions = pred_frames.get_trans()
    true_positions = true_frames.get_trans()
    diffused_positions = diffused_frames.get_trans()
    fape = compute_fape(
        pred_frames, true_frames, frame_mask,
        pred_positions, true_positions, frame_mask,
        pair_mask, loss_unit, clamp_distance, eps
    )   # [b, *]
    init_fape = compute_fape(
        diffused_frames, true_frames, frame_mask,
        diffused_positions, true_positions, frame_mask,
        pair_mask, loss_unit, clamp_distance, eps
    )   # [b, *]
    loss = layer_averaged_loss(fape, layer_weighting, stack_dim=0)
    loss_dict.update({
        "fape": fape[-1].data,
        "fape_loss": loss.data,
        "improve_fape": (fape[-1] - init_fape[0]).data,
    })
    return loss


def euclidean_v6(
    pred_frames: torch.Tensor,
    true_frames: torch.Tensor,
    diffused_frames: torch.Tensor,
    frame_mask: torch.Tensor,
    loss_unit: float,
    clamp_distance: float,
    rotation_weight: float,
    layer_weighting: str,
    loss_dict: dict = {},
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    # pred_frames = Frame.from_tensor_4x4(pred_frames)
    # true_frames = Frame.from_tensor_4x4(true_frames[None])
    p_rot, p_pos = pred_frames[..., :3, :].split((3, 1), dim=-1)    # *, L, 3/1, 3
    t_rot, t_pos = true_frames[..., :3, :].split((3, 1), dim=-1)
    rot_err = compute_rotation_error(p_rot, t_rot[None], frame_mask, eps)
    pos_err = compute_atom_pos_error(
        p_pos[..., 0, :], t_pos[None, ..., 0, :], frame_mask,
        length_scale=loss_unit, clamp_distance=clamp_distance, eps=eps
    )
    loss = layer_averaged_loss(
        pos_err + rotation_weight * rot_err, layer_weighting, stack_dim=0
    )

    if diffused_frames is not None:
        i_rot, i_pos = diffused_frames[..., :3, :].split((3, 1), dim=-1)
        rot_init = compute_rotation_error(i_rot, t_rot, frame_mask, eps)
        pos_init = compute_atom_pos_error(
            i_pos[..., 0, :], t_pos[None, ..., 0, :], frame_mask,
            length_scale=loss_unit, clamp_distance=clamp_distance, eps=eps
        )
        loss_dict.update({
            "improve_pos": (pos_err[-1] - pos_init).data,
            "improve_rot": (rot_err[-1] - rot_init).data,
        })

    loss_dict.update({
        "bb_pos": pos_err[-1].data,
        "bb_rot": rot_err[-1].data,
        "eucl_loss": loss.data
    })
    return loss


def structure_loss_v6(
    pred_frames: torch.Tensor,
    true_frames: torch.Tensor,
    diffused_frames: torch.Tensor,
    frame_mask: torch.Tensor,
    use_euclidean: bool,
    loss_unit: float,
    clamp_distance: float,
    rotation_weight: float,
    layer_weighting: str,
    loss_dict: dict = {},
    eps: float = 1e-4,
    **kwargs,
):
    if use_euclidean:
        loss = euclidean_v6(
            pred_frames, true_frames, diffused_frames, frame_mask,
            loss_unit=loss_unit,
            clamp_distance=clamp_distance,
            rotation_weight=rotation_weight,
            layer_weighting=layer_weighting,
            loss_dict=loss_dict,
            eps=eps,
        )
    else:
        loss = fape_v6(
            pred_frames, true_frames, frame_mask,
            pair_mask=None,
            loss_unit=loss_unit,
            clamp_distance=clamp_distance,
            layer_weighting=layer_weighting,
            loss_dict=loss_dict,
            eps=eps,
        )
    return loss


def structure_loss_v65(
    pred_frames: torch.Tensor,
    true_frames: torch.Tensor,
    diffused_frames: torch.Tensor,
    frame_mask: torch.Tensor,
    frame_gen_mask: torch.Tensor,
    diffusion_t: torch.Tensor,
    motif_weight: float,
    loss_type: str,
    t_scale: float,
    loss_unit: float,
    clamp_distance: float,
    rotation_weight: float,
    layer_weighting: str,
    loss_dict: dict = {},
    eps: float = 1e-4,
    **kwargs,
):
    if loss_type in ("euclidean", "hybrid"):
        weighted_frame_mask = motif_weight * frame_mask + (1. - motif_weight) * frame_gen_mask
        loss_eucl = euclidean_v6(
            pred_frames, true_frames, diffused_frames, weighted_frame_mask,
            loss_unit=loss_unit,
            clamp_distance=clamp_distance,
            rotation_weight=rotation_weight,
            layer_weighting=layer_weighting,
            loss_dict=loss_dict,
            eps=eps,
        )
    if loss_type in ("fape", "hybrid"):
        pair_mask = frame_mask[..., None] * frame_mask[..., None, :]
        is_motif = (frame_mask - frame_gen_mask).clamp_min(0.)  # is a valid frame, while not generated.
        motif_pair_mask = is_motif[..., None] * is_motif[..., None, :]
        weighted_pair_mask = torch.where(
            motif_pair_mask > 0,
            motif_weight,
            pair_mask,
        )
        loss_fape = fape_v6(
            pred_frames, true_frames, diffused_frames, frame_mask,
            pair_mask=weighted_pair_mask,
            loss_unit=loss_unit,
            clamp_distance=clamp_distance,
            layer_weighting=layer_weighting,
            loss_dict=loss_dict,
            eps=eps,
        )

    if loss_type == "euclidean":
        loss = loss_eucl
    elif loss_type == "fape":
        loss = loss_fape
    elif loss_type == "hybrid":
        # eucl_weight = (1. - diffusion_t) ** t_scale
        eucl_weight = 0.1
        loss = eucl_weight * loss_eucl + loss_fape
        loss_dict["structure_loss"] = loss.data
    else:
        raise ValueError(loss_type)
    return loss
