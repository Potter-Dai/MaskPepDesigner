from unifold.data import data_ops as uf_ops
from . import data_ops as df_ops
from . import dfv6_ops as v6_ops
# from MPDesigner.diffusion.transition import DiffusionTransition
from MPDesigner.diffusion.transition_v6 import Diffuser

from unicore.data.data_utils import numpy_seed
from .utils import NumpyDict, TensorDict, timeout
from typing import *
from ml_collections import ConfigDict

import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)
PROCESS_TIMEOUT = 180

def make_process_operators(
    data_cfg: ConfigDict,
    mode_cfg: ConfigDict,
    task: str,
    diffuser: Diffuser,
    crop_seed: int,
    motif_seed: int,
    time_seed: int,
    diffusion_seed: int,
    augmentation_seed: int,
):
    """process pipeline operators."""
    operators = [
        uf_ops.select_feat(data_cfg.raw_features),
        v6_ops.cast_dtypes,
        uf_ops.make_seq_mask,
        df_ops.preprocess_add_noise,  # add by DBT
    ]

    # no sidechain!
    # if mode_cfg.supervised: # this op relies on before-cropped info.
    #     operators.append(uf_ops.atom37_to_torsion_angles(""))

    if mode_cfg.crop:
        operators.extend([
            df_ops.crop_to_size(
                crop_size=data_cfg.crop.crop_size,
                shape_schema=data_cfg.batch_features,
                seed=crop_seed,
                spatial_crop_prob=data_cfg.crop.spatial_crop_prob,
                interface_threshold=data_cfg.crop.interface_threshold,
                inf=data_cfg.crop.inf,
            ),
            df_ops.pad_to_size(
                pad_size=data_cfg.crop.pad_size,
                shape_schema=data_cfg.batch_features,
            )
        ])

    operators.extend([
        df_ops.make_relative_positions(cutoff=data_cfg.relative_position_cutoff, is_cyclic=data_cfg.is_cyclic),
        # clean before diffusion
        v6_ops.make_aalogits(
            num_classes=data_cfg.num_aatype_cls, scale=data_cfg.aalogits_scale,
        ),
        df_ops.make_initial_frames(augmentation_seed=augmentation_seed, **data_cfg.frame_augmentation),
        # diffuse
        df_ops.make_generate_masks(motif_seed, task, data_cfg.motif_schema),
        v6_ops.make_diffusion_t(time_seed, task=task),
        # df_ops.diffuse(diffuser, diffusion_seed),
        v6_ops.diffuse(diffuser, diffusion_seed),
        df_ops.make_noisy_quats(enable=data_cfg.make_initial_quaterions),
        df_ops.make_distograms(data_cfg.distogram),
        v6_ops.make_features(
            num_relpos_cls=(2*data_cfg.relative_position_cutoff+2),
            num_distogram_bins=data_cfg.distogram.num_bins,
            num_time_rbf_bins=data_cfg.num_time_rbf_bins,
            num_aalogits_rbf_bins=data_cfg.num_aalogits_rbf_bins,
            aalogits_rbf_max=data_cfg.aalogits_rbf_max
        )
    ])

    # if mode_cfg.supervised:
    #     operators.extend([
    #         uf_ops.make_atom14_masks,
    #         uf_ops.make_atom14_positions,
    #         uf_ops.atom37_to_frames,
    #     ])

    operators.append(uf_ops.select_feat(data_cfg.batch_features))

    return operators


def make_random_seeds(
    seed: int,
    **fixed_seeds: Dict[str, int]
) -> Dict[str, int]:
    """
    parse and generate random seeds.
    """
    valid_keys = (
        "crop_seed",
        "motif_seed",
        "time_seed",
        "diffusion_seed",
        "augmentation_seed",
    )

    for k, v in fixed_seeds.items():
        assert k in valid_keys, f"unknown random seed key {k}."
        assert isinstance(v, int), f"seeds must be integers, got {type(k)}={k}."

    def gen_seed(k):
        s = fixed_seeds.get(k, None)
        if s is None:
            with numpy_seed(seed, 0, key=f"gen_{k}"):
                s = np.random.randint(1 << 31)
        return s

    return {k: gen_seed(k) for k in valid_keys}


@timeout(PROCESS_TIMEOUT, "process_features")
def process_features(
    features: NumpyDict,
    data_cfg: ConfigDict,
    mode_cfg: ConfigDict,
    task: str,
    diffusion_transition: Optional[Diffuser] = None,
    seed: Optional[int] = None,
    **fixed_seeds,
):
    """process inputs with given config, diffusion transition and seeds."""
    tensors: TensorDict = {
        k: torch.tensor(v)
        for k, v in features.items()
    }

    if diffusion_transition is None:    # lazy init, not recommended.
        logger.warn(f"No available diffusion transition object, "
                     f"so create one runtime. This can be slow.")
        diffusion_transition = Diffuser(data_cfg.diffusion)

    random_seeds = make_random_seeds(
        seed, **fixed_seeds
    )

    operators = make_process_operators(
        data_cfg, mode_cfg, task, diffusion_transition, **random_seeds,
    )

    if seed is not None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            tensors = compose(operators)(tensors)
    else:
        tensors = compose(operators)(tensors)

    return tensors

def update_features(tensors, data_cfg):
    operators = [
        df_ops.make_noisy_quats(enable=data_cfg.make_initial_quaterions),
        df_ops.make_distograms(data_cfg.distogram),
        v6_ops.make_features(
            num_relpos_cls=(2*data_cfg.relative_position_cutoff+2),
            num_distogram_bins=data_cfg.distogram.num_bins,
            num_time_rbf_bins=data_cfg.num_time_rbf_bins,
            num_aalogits_rbf_bins=data_cfg.num_aalogits_rbf_bins,
            aalogits_rbf_max=data_cfg.aalogits_rbf_max
        )
    ]
    tensors = compose(operators)(tensors)
    return tensors

@uf_ops.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x
