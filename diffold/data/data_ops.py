import numpy as np
import torch
import math
from typing import *

from ml_collections import ConfigDict

from unicore.utils import masked_mean
from unicore.data.data_utils import numpy_seed
from unifold.data.data_ops import curry1, MSA_FEATURE_NAMES, one_hot, apply_crop_idx
from unifold.modules.frame import Frame, Rotation, rot_to_quat

from unifold.losses.geometry import compute_distogram
from unifold.data import residue_constants as rc
ca_idx = rc.atom_order["CA"]

from diffold.config import N_RES
from diffold.diffusion.transition_v6 import uniform_random_rotations
from .utils import recur_print, TensorDict

from itertools import chain

import logging
logger = logging.getLogger(__name__)

ShapeSchema = Dict[str, List[Any]]


def add_gaussian_noise(coords: torch.Tensor, mean: float = 0.0, std: float = 0.2) -> torch.Tensor:
    """Adds Gaussian noise to the input coordinates."""
    noise = torch.randn_like(coords) * std + mean
    return coords + noise


def process_coordinates(
    all_atom_positions: torch.Tensor,
    add_noise: bool = True,  
    noise_mean: float = 0.0,
    noise_std: float = 0.2
) -> torch.Tensor:
    """Processes protein coordinates with optional Gaussian noise."""
    
    if add_noise:
        all_atom_positions = add_gaussian_noise(all_atom_positions, mean=noise_mean, std=noise_std)
    return all_atom_positions


def preprocess_add_noise(
    protein: TensorDict, 
    add_noise: bool = True
):
    """Preprocess protein data including coordinate noise addition."""

    all_atom_positions = protein["all_atom_positions"]
    protein["all_atom_positions"] = process_coordinates(
        all_atom_positions,
        add_noise=add_noise
    )

    return protein

def class_id_to_normalized_logits(
    aatype: torch.Tensor,
    num_classes: int,
):
    aatype_pos = one_hot(aatype, num_classes)
    pos_coef = math.sqrt(num_classes - 1)
    neg_coef = -1. / pos_coef
    return aatype_pos * pos_coef + (1. - aatype_pos) * neg_coef


@curry1
def make_initial_sequence_logits(
    protein: TensorDict,
    num_classes: int,
):
    protein["seq_logits"] = class_id_to_normalized_logits(protein["aatype"], num_classes)
    return protein


def augment_unknown_frames(
    frames,                 # [L, 4, 4]
    relative_positions,     # [L, L]
    seq_mask,               # [L]
    frame_mask,             # [L]
    sigma: float = 8.,
    randomness: float = 2.,
    eps: float = 1e-12,
):
    trans = frames[..., :, :3, 3]   # [*, L, 3]
    relpos_weight = torch.exp(-.5 * (relative_positions / sigma) ** 2)    # [*, L, L]
    relpos_weight *= frame_mask[..., None, :]
    sum_weight = relpos_weight.sum(dim=-1)    # [*, L]
    std = randomness * ((20. / sum_weight.clamp(min=1., max=20.))) ** 0.3

    if frames.device != torch.device("cpu"):
        approx_trans = torch.einsum("...ij,...jr->...ir", relpos_weight, trans)
        approx_trans /= sum_weight[..., None] + eps

        approx_trans += torch.from_numpy(np.random.randn(*approx_trans.shape)) * std[..., None]

        random_rot = uniform_random_rotations(frames.shape[:-2], frames.dtype, frames.device)
        approx_frames = Frame(Rotation(random_rot), approx_trans).to_tensor_4x4()

        ret_frames = torch.where(
            ((frame_mask[..., None, None] > 0.) | (seq_mask == 0.))[..., None, None],
            frames, approx_frames
        )

    else:   # presume no batch shape.
        ret_frames = frames.clone()
        for i, fm, sm in zip(range(len(frame_mask)), frame_mask, seq_mask):
            if fm > 0. or sm == 0.:
                continue
            approx_trans = (relpos_weight[i, :, None] * trans).sum(dim=-2)
            approx_trans /= (sum_weight[i] + eps)
            approx_trans += torch.from_numpy(np.random.randn(3)) * std[i]
            ret_frames[i, :3, 3] = approx_trans
            ret_frames[i, :3, :3] = uniform_random_rotations((), frames.dtype, frames.device)

    return ret_frames


@curry1
def make_initial_frames(
    protein: TensorDict,
    augmentation_seed: int = None,
    use_random_rotations: bool = False,
    use_augment_unknown_frames: bool = True,
    relpos_sigma: float = 8.,
    translation_randomness: float = 2.,
    eps: float = 1e-12,
):
    atom_pos, atom_mask = protein["all_atom_positions"], protein["all_atom_mask"]
    # 从 protein 字典中获取 centering_mask_idx 并传递
    centering_idx = protein.get("centering_mask_idx", None)
    atom_pos = remove_center(atom_pos, atom_mask, eps, centering_mask_idx=centering_idx)
    
    if use_random_rotations:
        with numpy_seed(augmentation_seed, 0, key="random_rotation"):
            atom_pos = apply_random_rotation(atom_pos, atom_mask)
    
    frames, frame_mask = make_backbone_frames(
        atom_pos, atom_mask, eps
    )

    if use_augment_unknown_frames:
        with numpy_seed(augmentation_seed, 0, key="augment_frames"):
            frames = augment_unknown_frames(
                frames,
                protein["relative_positions"],
                protein["seq_mask"],
                frame_mask,
                sigma=relpos_sigma,
                randomness=translation_randomness,
                eps=eps,
            )
    
    protein.update({
        "all_atom_positions": atom_pos,
        "frames": frames,
        "frame_mask": frame_mask
    })

    return protein


# data_ops.py

def remove_center(
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    eps: float = 1e-8,
    centering_mask_idx: Optional[torch.Tensor] = None, # <--- 添加新参数
):
    """
    Moves the center of a selected subset of residues (or all residues) to the origin.
    The translation is applied to all atoms.
    """
    if centering_mask_idx is not None and len(centering_mask_idx) > 0:
        # 使用提供的索引来选择CA原子以计算中心
        pos = all_atom_positions[centering_mask_idx, ca_idx, :]
        msk = all_atom_mask[centering_mask_idx, ca_idx, None]
    else:
        # 回退到原始行为：使用所有CA原子
        pos = all_atom_positions[..., ca_idx, :]
        msk = all_atom_mask[..., ca_idx, None]

    # 计算中心点 (centre) 的逻辑不变
    centre = (pos * msk).sum(dim=-2, keepdim=True) / (msk.sum(dim=-2, keepdim=True)).clamp_min(eps)
    
    # 将平移应用于所有的原子坐标 (all_atom_positions)
    all_atom_positions -= centre[..., :, None, :]
    all_atom_positions *= all_atom_mask[..., None]

    return all_atom_positions


def apply_random_rotation(
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
):
    random_rot = uniform_random_rotations(
        (), all_atom_positions.dtype, all_atom_positions.device,
    )
    all_atom_positions @= (random_rot.T)
    all_atom_positions *= (all_atom_mask[..., None])
    return all_atom_positions


def make_backbone_frames(
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    eps: float = 1e-8,
):
    num_batch_dim = len(all_atom_positions.shape[:-2])

    gt_frames = Frame.from_3_points(
        p_neg_x_axis=all_atom_positions[..., 2, :],
        origin=all_atom_positions[..., 1, :],
        p_xy_plane=all_atom_positions[..., 0, :],
        eps=eps,
    )

    rots = torch.diag(
        torch.tensor(
            (-1., 1., -1.),
            dtype=all_atom_positions.dtype,
            device=all_atom_positions.device
        )
    )
    rots = torch.tile(rots, (1,) * (num_batch_dim + 2))
    gt_frames = gt_frames.compose(Frame(Rotation(rots), None))

    gt_exists = torch.min(all_atom_mask[..., :3], dim=-1, keepdim=False)[0]

    return gt_frames.to_tensor_4x4(), gt_exists


def flip(prob):
    return int(np.random.rand() < prob)


def locally_continuous_mask(mask: torch.Tensor, keepprob: float, K: int):
    assert len(mask.shape) == 1
    beta_1 = 1. - 1. / K
    if keepprob < beta_1:
        beta_2 = keepprob / (1. - keepprob) / K
    else:
        beta_1 = beta_2 = keepprob

    p = keepprob
    ret = []
    for mc in mask.tolist():
        m = flip(p) * mc
        p = beta_1 if m else beta_2
        p = keepprob if mc == 0. else p    # reset p for hard 0s.
        ret.append(m)

    return torch.tensor(ret, dtype=mask.dtype)   # E(continuous 1s)=K


def generate_ligand_mask(protein, mask_type: str):
    """
    Generate a mask based on the specified mask_type.
    
    Args:
        protein: Dictionary containing protein data (e.g., 'ligmask', 'chain_id').
        mask_type: str, either "random" (for hotspots on peptide) or "chain_id" (for hotspots on target protein).
    
    Returns:
        gen_mask: torch.Tensor, mask with 0s and 1s based on the specified logic.
    """
    gen_mask = torch.ones(protein["chain_id"].shape, dtype=torch.float32)

    if mask_type == "random":
        gen_mask[protein["chain_id"] == 2.] = 0.
        # Randomly unmask 1~3 hotspot residues (ligmask == 0.) as structure anchors.
        ligmask = protein["ligmask"]
        indices = torch.where(ligmask == 0.)[0]
        if len(indices) > 0:
            num_to_select = torch.randint(1, min(3, len(indices)) + 1, (1,)).item()
            selected_indices = indices[torch.randperm(len(indices))[:num_to_select]]
            gen_mask[selected_indices] = 1.
        else:
            logging.warning("No residues with ligmask == 0 found. bad data for multimer.")
    elif mask_type == "chain_id":
        # Mask all residues with chain_id == 2
        indices = torch.where(protein["chain_id"] == 2.)[0]
        if len(indices) > 0:
            gen_mask[indices] = 0.
        else:
            logging.warning("No residues with chain_id = 2 found. bad data for multimer.")
    else:
        raise ValueError(f"Invalid mask_type: {mask_type}. Expected 'random' or 'chain_id'.")

    return gen_mask

def linear_kernel(p, low: float = 0.2, high: float = 0.8):
    return max(p - low, 0.) * (high / (1. - low))

def square_kernel(p, low: float = 0.2, high: float = 0.8):
    return (max(p - low, 0.) ** 2) * (high / (1. - low) ** 2)

def improved_ddpm_kernel(t, s, T=1):
    fn = lambda x: math.cos(x * math.pi / 2) ** 2
    normed_t = (t + s) / (T + s)
    normed_zero = s / (T + s)
    return fn(normed_t) / fn(normed_zero)


OBSERVED_PROB_FN = {
    "linear": linear_kernel,
    "square": square_kernel,
}


def generate_training_gammas_v3(
    seed: int,
    shape: Tuple[int],
    dtype: torch.dtype,
    codesign_prob: float,
    s: float,
    K: int,
    observed_prob_kernel: str,
    denovo_prob: float,
    max_observed_prob: float,
    **kwargs,
) -> TensorDict:
    raise NotImplementedError("depreciated.")
    seeds = (seed, 0)       # explicitly add an addl seed 0 for unicore issue.
    with numpy_seed(*seeds, key="diff_mask_type"):
        use_codesign = flip(codesign_prob)

    with numpy_seed(*seeds, key="diff_mask_prob"):
        observed_prob = np.random.rand() # [0, 1)
    
    with numpy_seed(*seeds, key="diff_t"):
        t = np.random.rand()

    gamma = improved_ddpm_kernel(t, s, T=1)
    observed_prob = OBSERVED_PROB_FN[observed_prob_kernel](
        observed_prob, low=denovo_prob, high=max_observed_prob
    )  # [0, max_obs_prob)

    seqlen = shape[-1]
    if observed_prob < 0.1:  # <==> prob = 0, i.e de novo generation.
        is_observed_seq = torch.zeros(shape)
        is_observed_frame = torch.zeros(shape)
    elif use_codesign:    # shared seq & frame mask
        is_observed_seq = locally_continuous_mask(
            seqlen,
            observed_prob,   # prob is the ratio of 1s. which is to be kept undiffed.
            K=K,
            seeds=seeds,
            key="codesign_diff_mask"
        )
        is_observed_frame = is_observed_seq
    else:               # individual seq & frame mask
        is_observed_seq = locally_continuous_mask(
            seqlen,
            observed_prob,
            K=K,
            seeds=seeds,
            key="seq_diff_mask"
        )
        is_observed_frame = locally_continuous_mask(
            seqlen,
            observed_prob,
            K=K,
            seeds=seeds,
            key="frame_diff_mask"
        )
    seq_gammas = gamma + is_observed_seq.type(dtype) * (1. - gamma)
    frame_gammas = gamma + is_observed_frame.type(dtype) * (1. - gamma)
    return seq_gammas, frame_gammas


GAMMA_FN = {
    "v3": generate_training_gammas_v3,
    # "v4": generate_training_gammas,
}


@curry1
def generate_gammas(
    protein: TensorDict,
    seed: int,
    task: str,
    gamma_schema: ConfigDict,
):
    """
    Generate the gammas which controls the noise scales of the sequence and
        structure. Output gammas are equivalent to the \\bar{\\alpha}s in 
        ddpms. Tntuitively, gammas are signal ratios: gamma -> 1 indicates no 
        noises; gamma -> 0 indicates all noises no signal.
    """
    raise NotImplementedError("depreciated.")

    seq_mask = protein["seq_mask"]
    frame_mask = protein["frame_mask"]

    if task == "train":
        gamma_fn = GAMMA_FN[gamma_schema.version]
        seq_gammas, frame_gammas = gamma_fn(seed, seq_mask.shape, seq_mask.dtype, **gamma_schema)
    elif task == "eval_fold":
        seq_gammas = torch.ones_like(seq_mask)
        frame_gammas = torch.zeros_like(seq_mask)
    elif task == "eval_invfold":
        seq_gammas = torch.zeros_like(seq_mask)
        frame_gammas = torch.ones_like(seq_mask)
    elif task == "eval_denovo":
        seq_gammas = torch.zeros_like(seq_mask)
        frame_gammas = torch.zeros_like(seq_mask)
    elif task == "eval_inpaint":
        seq_gammas = 1. - locally_continuous_mask(seq_mask.shape[-1], 0.2, K=6, seeds=(seed, 0), key="seq_lcm")
        frame_gammas = seq_gammas
    elif task == "eval_scaffold":
        seq_gammas = 1. - locally_continuous_mask(seq_mask.shape[-1], 0.8, K=6, seeds=(seed, 0), key="seq_lcm")
        frame_gammas = seq_gammas
    else:
        raise ValueError(f"unknown task {task}")

    # for residues w/o known seq / frames, ensure they are sampled from prior.
    seq_gammas *= seq_mask
    frame_gammas *= frame_mask

    protein.update({
        "seq_gammas": seq_gammas,
        "frame_gammas": frame_gammas
    })

    return protein


@curry1
def diffuse(protein, transition, seed: int):
    raise NotImplementedError("depreciated.")
    diffused_vars = transition.add_noise(protein, seed)
    protein.update(diffused_vars)
    return protein


@curry1
def make_noisy_quats(protein, enable=False):
    if not enable:
        return protein
    assert "diffused_frames" in protein
    frames = protein["diffused_frames"]
    rot, trans = torch.split(frames[..., :3, :], (3, 1), dim=-1)
    quat = rot_to_quat(rot.cpu()).to(rot.device)
    protein["diffused_quats"] = torch.cat([quat, trans.squeeze(-1)], dim=-1)
    return protein


@curry1
def make_target_feat(protein, num_classes):
    """Create and concatenate MSA features."""
    raise NotImplementedError("depreciated.")
    aatype_1h = one_hot(protein["diffused_aatype"], num_classes)
    target_feat = [
        aatype_1h,
        protein["seq_gammas"][..., None],       # [*, L, 1]
        protein["frame_gammas"][..., None],     # [*, L, 1]
    ]
    protein["target_feat"] = torch.cat(target_feat, dim=-1)
    return protein

# crop & pad utils

def get_contiguous_crop_index(
    chain_id: torch.Tensor, crop_size: int, seed: int
) -> torch.Tensor:
    # 0. do not crop if unnecessary.
    if chain_id.shape[0] <= crop_size:
        return torch.arange(chain_id.shape[0])
    
    # 1. sample lengths of each chains.
    with numpy_seed(seed, 0, key="get_chain_sizes"):
        sampled_chain_id = np.random.choice(
            chain_id.numpy(), size=crop_size, replace=False
        )
    sampled_chains, sampled_chain_sizes = \
        torch.from_numpy(sampled_chain_id).unique(return_counts=True)
    
    # 2. generate lengths and offsets of each chains
    unique_chains, chain_lens = chain_id.unique(return_counts=True)
    chain_reverse_offsets = torch.cumsum(chain_lens, -1)    # reverse offsets
    chain_info = {
        int(ch): (seqlen, rev_offset)
        for ch, seqlen, rev_offset 
        in zip(unique_chains, chain_lens, chain_reverse_offsets)
    }
    
    # 3. create sampled index
    ret = []
    for ch, sz in zip(sampled_chains, sampled_chain_sizes):
        seqlen, rev_offset = chain_info[int(ch)]
        with numpy_seed(seed, 0, int(ch), key="sample_start"):
            reverse_start = np.random.randint(0, seqlen - sz + 1)
        idx = torch.arange(
            rev_offset - sz - reverse_start,
            rev_offset - reverse_start,
            1, dtype=torch.int
        )
        ret.append(idx)
    ret = torch.cat(ret)
    assert ret.shape[0] == crop_size
    return ret


### 修改为蛋白-多肽在两条链上的crop
def compute_min_distances(ca_coords, indices1, indices2):
    """Compute minimum distances between two sets of CA coordinates."""
    if len(indices2) == 0:
        raise ValueError("No residues in the second set.")
    ca1 = ca_coords[indices1]  # [N1, 3]
    ca2 = ca_coords[indices2]  # [N2, 3]
    diff = ca1.unsqueeze(1) - ca2.unsqueeze(0)  # [N1, N2, 3]
    distances = torch.sqrt((diff ** 2).sum(dim=-1))  # [N1, N2]
    min_distances, _ = distances.min(dim=1)  # [N1]
    return min_distances

def get_spatial_crop_index(
    ca_coords: torch.Tensor,  # [L, 3]
    ca_mask: torch.Tensor,  # [L,]
    chain_id: torch.Tensor,  # [L,]
    crop_size: int,
    seed: int,
    hpmask: Optional[torch.Tensor] = None,  # 可选的hotspot掩码 [L,]
    inf: float = 1e5,
) -> Tuple[torch.Tensor, torch.Tensor]:  # <--- 修改返回类型
    # 先计算 hotspot_indices 和 indices_for_centering，以确保所有分支一致返回
    protein_indices = torch.where(chain_id == 1.)[0]
    peptide_indices = torch.where(chain_id == 2.)[0]

    # Determine hotspot residues based on hpmask
    if hpmask is not None:
        # Case 1: hpmask exists
        hpmask_indices = torch.where(hpmask == 1.)[0]
        if len(hpmask_indices) == 0:
            raise ValueError("No residues with hpmask == 1 found.")
        # min_distances = compute_min_distances(ca_coords, protein_indices, hpmask_indices)
        # pocket_indices = protein_indices[min_distances <= 10.0]
        hotspot_indices = hpmask_indices
    else:
        # Case 2: hpmask is None
        peptide_ca_indices = torch.where(chain_id == 2.)[0]
        if len(peptide_ca_indices) == 0:
            raise ValueError("No peptide residues with ca_mask == 1 found.")
        min_distances_to_peptide = compute_min_distances(ca_coords, protein_indices, peptide_ca_indices)
        contact_indices = protein_indices[min_distances_to_peptide <= 8.0]
        if len(contact_indices) == 0:
            raise ValueError("No contact residues found within 8 Å of peptide.")
        # min_distances_to_contact = compute_min_distances(ca_coords, protein_indices, contact_indices)
        # motif_indices = protein_indices[min_distances_to_contact <= 10.0]
        hotspot_indices = contact_indices

    # `indices_for_centering` 只包含热点残基，将用它来定义中心点
    indices_for_centering = hotspot_indices

    # 0. do not crop if unnecessary. 
    if chain_id.shape[0] <= crop_size:
        return torch.arange(chain_id.shape[0]), indices_for_centering

    # 1. if num. residues with structures is within crop size, return them all and sample some no-structure residues.
    if (ca_mask.sum() <= crop_size): # 这是说：假如有的残基没有CA原子
        with_structure_index = torch.nonzero(ca_mask)[:, 0]
        to_pad = crop_size - with_structure_index.shape[0]
        if to_pad:
            no_structure_index = torch.nonzero(1. - ca_mask)[:, 0].numpy()
            with numpy_seed(seed, 0, key="no_structure_residues"):
                sampled_index = np.random.choice(
                    no_structure_index, size=to_pad, replace=False
                )
            ret = torch.cat([with_structure_index, torch.from_numpy(sampled_index)])
        else:
            ret = with_structure_index

    # 2. find hpmask residues and compute the geometric center of them, and return nearest residues.
    else:
        # Combine peptide and hotspot indices
        ref_indices = torch.cat([peptide_indices, hotspot_indices])
        # Get CA coordinates of residues with chain_id > 1
        ref_ca_coords = ca_coords[ref_indices]

        # Compute pairwise distances from all residues to ligmask residues
        ca_diff = ca_coords.unsqueeze(0) - ref_ca_coords.unsqueeze(1)
        ca_sq_distances = ca_diff.square().sum(dim=-1)
        min_distances, _ = ca_sq_distances.min(dim=0)

        # Penalize residues not in ca_mask
        min_distances += (1 - ca_mask) * inf

        # Add slight random noise to break ties
        with numpy_seed(seed, 0, key="spatial_crop"):
            noise = np.random.randn(min_distances.shape[0]) * 1e-4
        min_distances += torch.from_numpy(noise).to(min_distances.device)

        # Select top `crop_size` residues based on distance
        ret = min_distances.argsort()[:crop_size]

    ret = ret.sort().values
    assert ret.shape[0] <= crop_size, f"crop_idx size {ret.shape[0]} exceeds crop_size {crop_size}"
    return ret, indices_for_centering


@curry1
def crop_to_size(
    protein: TensorDict,
    crop_size: int,
    shape_schema: ShapeSchema,
    seed: int,
    spatial_crop_prob: float,
    interface_threshold: float,
    inf: float,
):
    chain_id = protein["chain_id"]
    # input check
    assert len(chain_id.shape) == 1, chain_id.shape
    assert torch.all(chain_id > 0), \
        f"placeholders are not expected before crop, got chain_id {chain_id}."
    assert torch.all(chain_id[1:] >= chain_id[:-1]), \
        f"chain_id must be sorted, got {chain_id}."
    with numpy_seed(seed, 0, key="crop_style"):
        use_spatial_crop = np.random.rand() < spatial_crop_prob
        crop_seed = np.random.randint(1 << 31)
    
    if use_spatial_crop:
        hpmask = protein.get("hpmask", None)

        crop_idx, centering_indices_orig = get_spatial_crop_index(
            protein["all_atom_positions"][..., ca_idx, :],
            protein["all_atom_mask"][..., ca_idx],
            protein["chain_id"],
            crop_size=crop_size,
            seed=crop_seed,
            hpmask=hpmask, 
            inf=inf,
        )
        # --- 核心逻辑：将原始的中心化索引映射到裁切后的新索引 ---
        # 创建一个从原始索引到新索引的查找表
        # e.g., if crop_idx is [10, 11, 25], orig_to_new_map[10] will be 0, orig_to_new_map[11] will be 1, etc.
        sorter = torch.argsort(crop_idx)
        orig_to_new_map = torch.zeros(crop_idx.max() + 1, dtype=torch.long, device=crop_idx.device) - 1
        orig_to_new_map[crop_idx[sorter]] = torch.arange(len(crop_idx), device=crop_idx.device)[sorter]

        # 从中心化索引中只保留那些实际存在于裁切块中的索引
        mask_in_crop = centering_indices_orig < len(orig_to_new_map)
        valid_centering_indices = centering_indices_orig[mask_in_crop]
        
        # 获取这些索引在新坐标系下的值
        new_centering_idx = orig_to_new_map[valid_centering_indices]
        new_centering_idx = new_centering_idx[new_centering_idx != -1] # 过滤掉不在 crop_idx 中的索引

        protein["centering_mask_idx"] = new_centering_idx.long() # 存入字典，供后续使用
        
    else:
        crop_idx = get_contiguous_crop_index(
            protein["chain_id"],
            crop_size=crop_size,
            seed=crop_seed
        )
        # 对于连续裁切，我们可以将所有被裁切的残基作为中心化依据
        # 或者不提供 centering_mask_idx，让 remove_center 使用默认行为
        protein["centering_mask_idx"] = torch.arange(len(crop_idx))


    protein = apply_crop_idx(
        protein, shape_schema, crop_idx
    )

    return protein


def get_pad_size(cur_size, multiplier=4):
        return max(multiplier, ((cur_size + multiplier - 1) // multiplier) * multiplier)


@curry1
def pad_to_size(protein: TensorDict, pad_size: int, shape_schema: ShapeSchema):
    """pad num_res to (multiple) pad_size."""
    
    num_res = protein["aatype"].shape[0]
    pad_size = get_pad_size(num_res, pad_size)
    
    padded_protein = {}
    for k, v in protein.items():
        if k not in shape_schema:  # skip items with unknown shape schema
            continue
        schema = shape_schema[k]
        padding = [
            (0, pad_size - sz) if (d == N_RES or d is None) else (0, 0)
            for d, sz in zip(schema, v.shape)
        ]
        padding.reverse()
        padding = list(chain(*padding))

        v = torch.nn.functional.pad(v, padding, value=0.)
        padded_protein[k] = v
    return padded_protein


def compute_relative_positions(
    res_id: torch.Tensor,
    chain_id: torch.Tensor,
    cutoff: int,
    cyclic: Optional[torch.Tensor] = None
):
    different_chain_symbol = -(cutoff + 1)
    relpos = res_id[..., None] - res_id[..., None, :]

    if cyclic is not None and cyclic.any():
        cyclic_indices = torch.nonzero(cyclic, as_tuple=False).squeeze(-1)  
        cyclic_res_id = res_id[cyclic]
        cyclic_L = cyclic_res_id.shape[0]
        relpos_cyclic = cyclic_res_id[..., None] - cyclic_res_id[..., None, :]

        i = torch.arange(cyclic_L, device=res_id.device)
        ij = torch.stack([i, i + cyclic_L], -1) 
        cyclic_offsets = torch.abs(ij[:, None, :, None] - ij[None, :, None, :]).amin(dim=(2, 3))

        a = cyclic_offsets < torch.abs(relpos_cyclic)
        cyclic_offsets[a] = -cyclic_offsets[a]
        relpos_cyclic = cyclic_offsets * torch.sign(relpos_cyclic)

        relpos[cyclic_indices[:, None], cyclic_indices[None, :]] = relpos_cyclic

    relpos = relpos.clamp(-cutoff, cutoff)

    different_chain = (chain_id[..., None] != chain_id[..., None, :])
    relpos[different_chain] = different_chain_symbol

    return relpos

@curry1
def make_relative_positions(protein, cutoff: int, is_cyclic: bool):
    """
    Checks if the ligmask is continuous where values are 0.
    """
    if is_cyclic:
        cyc_mask = protein["ligmask"]
        # Check if `cyc_mask` values are continuous where they are `0`
        zero_indices = torch.where(cyc_mask == 0)[0]  # Get indices where `cyc_mask` is 0
        if len(zero_indices) > 1:
            is_continuous = torch.all((zero_indices[1:] - zero_indices[:-1]) == 1)
            if not is_continuous:
                raise ValueError("The `cyc_mask` values of 0 are not continuous. Please ensure continuity.")
        
        cyclic_mask_part = (cyc_mask == 0)    
        protein["relative_positions"] = compute_relative_positions(
            protein["residue_index"],
            protein["chain_id"],
            cutoff,
            cyclic=cyclic_mask_part
        )
    else:
        protein["relative_positions"] = compute_relative_positions(
            protein["residue_index"],
            protein["chain_id"],
            cutoff
        )
    return protein


def compute_distogram(
    ca_coords: torch.Tensor,
    min_bin=2.3125,
    max_bin=21.6875,
    num_bins=64,
):
    # bins are (0, min, min+d, ..., max, >max)
    # as we clamped distance min, all should be in distogram.
    sq_boundaries = torch.linspace(min_bin, max_bin, num_bins - 1) ** 2
    sq_boundaries = sq_boundaries.to(device=ca_coords.device, dtype=ca_coords.dtype)

    dists = torch.sum(
        (ca_coords[..., None, :] - ca_coords[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    bin_idx = torch.sum(dists > sq_boundaries, dim=-1)

    return bin_idx.long()

@curry1
def make_distograms(protein, distogram_args):
    distogram = compute_distogram(
        protein["all_atom_positions"][..., ca_idx, :],
        **distogram_args
    )

    diffused_distogram = compute_distogram(
        protein["diffused_frames"][..., :3, 3],
        **distogram_args
    )

    protein.update({
        "distogram": distogram,
        "diffused_distogram": diffused_distogram,
    })

    return protein


@curry1
def make_generate_masks(
    protein: TensorDict,
    seed: int,
    task: str,
    motif_schema: ConfigDict,
):
    seq_mask = protein["seq_mask"]
    frame_mask = protein["frame_mask"]

    if task in ("fold", "eval_fold_complex"):
        # Complex-level fold: 给真实序列，mask 整个复合物的结构
        is_observed_seq = torch.ones_like(seq_mask)
        is_observed_frame = torch.zeros_like(seq_mask)
    elif task in ("invfold", "eval_invfold_complex"):
        # Complex-level invfold: 给真实结构，mask 整个复合物的序列
        is_observed_seq = torch.zeros_like(seq_mask)
        is_observed_frame = torch.ones_like(seq_mask)
    elif task in ("eval_fold_peptide",):
        # Peptide-level fold: 给真实序列，只 mask 多肽结构（保留 hotspot 锚点）
        is_observed_seq = torch.ones_like(seq_mask)
        is_observed_frame = generate_ligand_mask(protein, mask_type=motif_schema.mask_type)
    elif task in ("eval_invfold_peptide",):
        # Peptide-level invfold: 给真实结构，mask 整条多肽序列
        is_observed_frame = torch.ones_like(seq_mask)
        is_observed_seq = generate_ligand_mask(protein, mask_type="chain_id")
    elif task in ("denovo", "eval_denovo"):
        is_observed_seq = torch.zeros_like(seq_mask)
        is_observed_frame = torch.zeros_like(seq_mask)
    elif task in ("customized", "predict", "eval"):
        assert "seq_gen_mask" in protein
        assert "frame_gen_mask" in protein
        is_observed_seq = 1. - protein["seq_gen_mask"]
        is_observed_frame = 1. - protein["frame_gen_mask"]
    elif task in ("train", "eval_train", "eval_last", "eval_half", "eval_init"):
        is_observed_seq, is_observed_frame = sample_train_motifs(
            protein, seed, seq_mask, frame_mask, **motif_schema,
        )
    else:
        raise ValueError(f"unknown task {task}.")

    # for residues w/o known seq / frames, ensure they are sampled.
    is_observed_seq = is_observed_seq * seq_mask
    is_observed_frame = is_observed_frame * frame_mask

    protein.update({
        "seq_gen_mask": 1. - is_observed_seq,
        "frame_gen_mask": 1. - is_observed_frame,
    })

    return protein


OBS_PROB_KERNEL = {
    "linear_00-80": lambda p: 0.8 * p,      # [0.0, 0.8]
    "linear_00-50": lambda p: 0.5 * p,      # [0.0, 0.5]
}


def sample_train_motifs(
    protein: TensorDict,
    seed: int,
    seq_mask: torch.Tensor,
    frame_mask: torch.Tensor,
    denovo_complex_prob: float,
    codesign_complex_prob: float,
    codesign_peptide_weight: float,
    fold_peptide_weight: float,
    invfold_peptide_weight: float,
    use_true_sequence: bool,
    use_true_structure: bool,
    observed_prob_kernel: str,
    mask_type: str,
    motif_k: int,
) -> TensorDict:
    seeds = (seed, 0)       # explicitly add an addl seed 0 for unicore issue.
    prob_kernel = OBS_PROB_KERNEL[observed_prob_kernel]
    with numpy_seed(*seeds, key="train_gen_mask"):
        is_denovo_complex = flip(denovo_complex_prob)
        is_codesign_complex = flip(codesign_complex_prob)
        seq_observed_prob = prob_kernel(np.random.rand())
        frame_observed_prob = prob_kernel(np.random.rand())
        # categorical sampling for peptide-level tasks
        peptide_weights = np.array([codesign_peptide_weight, fold_peptide_weight, invfold_peptide_weight])
        peptide_task = np.random.choice(3, p=peptide_weights / peptide_weights.sum())

    if is_denovo_complex:
        # De novo: mask 整个复合物的序列和结构
        is_observed_seq = is_observed_frame = torch.zeros_like(seq_mask)
    elif is_codesign_complex:
        # Co-design complex: 连续片段 mask（全复合物级别）
        with numpy_seed(*seeds, key="codesign_lcm"):
            is_observed_seq = is_observed_frame = locally_continuous_mask(
                frame_mask, seq_observed_prob, motif_k
            )
    elif peptide_task == 1:  # fold_peptide
        # Fold: 给真实序列，只 mask 多肽结构（保留 hotspot 结构锚点）
        is_observed_seq = torch.ones_like(seq_mask)
        is_observed_frame = generate_ligand_mask(protein, mask_type=mask_type)
    elif peptide_task == 2:  # invfold_peptide
        # Invfold: 给真实结构，mask 整条多肽序列
        is_observed_frame = torch.ones_like(seq_mask)
        is_observed_seq = generate_ligand_mask(protein, mask_type="chain_id")
    else:  # peptide_task == 0, codesign_peptide (default)
        # Co-design peptide: 同时 mask 多肽的序列和结构
        peptide_mask = generate_ligand_mask(protein, mask_type=mask_type)
        is_observed_seq = peptide_mask
        is_observed_frame = peptide_mask.clone()

    if use_true_sequence:
        is_observed_seq = seq_mask
    if use_true_structure:
        is_observed_frame = frame_mask

    return is_observed_seq, is_observed_frame
