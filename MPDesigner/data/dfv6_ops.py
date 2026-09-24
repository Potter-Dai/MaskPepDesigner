import torch
import numpy as np

from unifold.data.data_ops import curry1
from unicore.utils import one_hot

from unicore.data.data_utils import numpy_seed

from MPDesigner.diffusion.transition_v6 import Diffuser


def cast_dtypes(protein):
    for k, v in protein.items():
        if k.endswith("_mask"):
            protein[k] = v.type(torch.float32)
        elif v.dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            protein[k] = v.type(torch.float32)
        elif v.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            protein[k] = v.type(torch.int64)
        else:
            raise ValueError(f"unknown dtype {v.dtype}")

    return protein


@curry1
def make_aalogits(
    protein,
    num_classes: int,
    scale: float,
):
    aa1h = one_hot(protein["aatype"], num_classes, dtype=torch.float)
    aalogits = aa1h * scale * 2. - scale     # {-s, s}
    protein["aalogits"] = aalogits
    return protein


def rbf_kernel(
    r: torch.Tensor,    # [*]
    num_bins: int,
    r_min: float = 0,
    r_max: float = 1,
) -> torch.Tensor:      # [*, num_bins]
    r = r.clamp(max=r_max, min=r_min)   # clipping to range.
    bins = torch.linspace(
        r_min, r_max, num_bins,
        dtype=r.dtype, device=r.device
    )
    sigma = (r_max - r_min) / (num_bins - 1)
    rbf = torch.exp(-0.5 * ((r[..., None] - bins) / sigma) ** 2)
    return rbf


def featurization(
    aalogits,
    relpos,
    distogram,
    seq_gen_mask,
    frame_gen_mask,
    diffusion_t,
    num_relpos_cls,
    num_distogram_bins,
    num_time_rbf_bins,
    num_aalogits_rbf_bins,
    aalogits_rbf_max,
):
    # aatype_1h = one_hot(aatype, num_aatype_cls)
    relpos_1h = one_hot(relpos % num_relpos_cls, num_relpos_cls)
    distogram_1h = one_hot(distogram, num_distogram_bins)
    time_rbf = rbf_kernel(diffusion_t, num_time_rbf_bins, r_min=0., r_max=1.)

    aalogits_rbf = rbf_kernel(
        aalogits, num_aalogits_rbf_bins,
        r_min=-aalogits_rbf_max, r_max=aalogits_rbf_max,
    )
    aalogits_rbf = aalogits_rbf.reshape(*aalogits.shape[:-1], -1)

    feat_1d = torch.cat([
        aalogits,           # 21
        aalogits_rbf,       # 21 * num_rbf
        seq_gen_mask[..., None],       # 1
        frame_gen_mask[..., None],     # 1
        # time_feat,
    ], dim=-1)

    feat_2d = torch.cat([
        relpos_1h,          # num_relpos_cls
        distogram_1h,       # num_distogram_cls
        (seq_gen_mask[..., None] * seq_gen_mask[..., None, :])[..., None],       # 1
        (frame_gen_mask[..., None] * frame_gen_mask[..., None, :])[..., None],   # 1
    ], dim=-1)

    time_rbf = time_rbf.tile(*seq_gen_mask.shape, 1)
    time_feat = torch.cat((time_rbf, seq_gen_mask[..., None], frame_gen_mask[..., None]), dim=-1)

    return feat_1d, feat_2d, time_feat


@curry1
def make_diffusion_t(protein, seed, task):
    # for stage-wise evaluation
    if "diffusion_t" in protein:
        assert len(protein["diffusion_t"].shape) == 0, protein["diffusion_t"]
        assert protein["diffusion_t"] >= 0 and protein["diffusion_t"] <= 1
        assert task in ["predict", "eval"]
        return protein

    if "init" in task:
        t = 0.01
    elif "half" in task:
        t = 0.50
    elif "last" in task:
        t = 0.99
    else:
        with numpy_seed(seed, 0, key="diffusion_t"):
            t = np.random.rand()
    protein["diffusion_t"] = torch.tensor(t)
    return protein


@curry1
def diffuse(
    protein,
    diffuser: Diffuser,
    seed: int
):
    with numpy_seed(seed, 0, key="diffuse"):
        s_t, f_t = diffuser.addnoise(
            protein["aalogits"],
            protein["frames"],
            protein["seq_gen_mask"],
            protein["frame_gen_mask"],
            protein["diffusion_t"],
        )
    protein["diffused_aalogits"] = s_t
    protein["diffused_frames"] = f_t
    return protein


@curry1
def make_features(
    protein,
    num_relpos_cls,
    num_distogram_bins,
    num_time_rbf_bins,
    num_aalogits_rbf_bins,
    aalogits_rbf_max,
):
    feat1d, feat2d, time_feat = featurization(
        protein["diffused_aalogits"],
        protein["relative_positions"],
        protein["diffused_distogram"],
        protein["seq_gen_mask"],
        protein["frame_gen_mask"],
        protein["diffusion_t"],
        num_relpos_cls=num_relpos_cls,
        num_distogram_bins=num_distogram_bins,
        num_time_rbf_bins=num_time_rbf_bins,
        num_aalogits_rbf_bins=num_aalogits_rbf_bins,
        aalogits_rbf_max=aalogits_rbf_max,
    )
    protein["feat_1d"] = feat1d
    protein["feat_2d"] = feat2d
    protein["time_feat"] = time_feat
    return protein
