import ml_collections as mlc
import copy
from typing import Any


N_RES = "number of residues"
N_ATOM = "number of atoms := 37"


d_pair = mlc.FieldReference(128, field_type=int)
d_single = mlc.FieldReference(384, field_type=int)

d_input = mlc.FieldReference(23, field_type=int)

num_zormer = mlc.FieldReference(48, field_type=int)

eps = mlc.FieldReference(1e-8, field_type=float)
inf = mlc.FieldReference(3e4, field_type=float)

relpos_k = mlc.FieldReference(32, field_type=int)
plddt_bins = mlc.FieldReference(50, field_type=int)
pae_bins = mlc.FieldReference(64, field_type=int)
distogram_bins = mlc.FieldReference(64, field_type=int)

layer_weighting = mlc.FieldReference("last", field_type=str)
crop_size = mlc.FieldReference(256, field_type=int)
pad_size = mlc.FieldReference(4, field_type=int)

num_aatype = mlc.FieldReference(21, field_type=int)

RAW_FEATURES = {
    "uid": [None],
    "aatype": [N_RES],
    "residue_index": [N_RES],
    "all_atom_positions": [N_RES, N_ATOM, 3],
    "all_atom_mask": [N_RES, N_ATOM],
    "chain_id": [N_RES],
    "bfactor": [N_RES],
    "resolution": [1],
}

BATCH_FEATURES = {
    **RAW_FEATURES,
    # diffusion params
    "seq_gammas": [N_RES],
    "frame_gammas": [N_RES],
    "seq_gammas_to": [N_RES],
    "frame_gammas_to": [N_RES],
    # model inputs
    # "diffused_seq_logits": [N_RES, 21],
    "diffused_aatype": [N_RES],
    "diffused_frames": [N_RES, 4, 4],
    "diffused_quats": [N_RES, 7],
    "target_feat": [N_RES, 23],
    "seq_mask": [N_RES],
    # supervisions.
    # labels
    "seq_logits": [N_RES, 21],
    "frames": [N_RES, 4, 4],
    "frame_mask": [N_RES],
    # atom14/37 constants
    "residx_atom14_to_atom37": [N_RES, 14],
    "residx_atom37_to_atom14": [N_RES, 37],
    "atom14_atom_exists": [N_RES, 14],
    "atom37_atom_exists": [N_RES, 37],
    "atom14_gt_exists": [N_RES, 14],
    "atom14_gt_positions": [N_RES, 14, 3],
    "atom14_alt_gt_positions": [N_RES, 14, 3],
    "atom14_alt_gt_exists": [N_RES, 14],
    "atom14_atom_is_ambiguous": [N_RES, 14],
    # rigidgroup constants
    "rigidgroups_gt_frames": [N_RES, 8, 4, 4],
    "rigidgroups_gt_exists": [N_RES, 8],
    "rigidgroups_group_exists": [N_RES, 8],
    "rigidgroups_group_is_ambiguous": [N_RES, 8],
    "rigidgroups_alt_gt_frames": [N_RES, 8, 4, 4],
    # torsion angles
    "torsion_angles_sin_cos": [N_RES, 7, 2],
    "alt_torsion_angles_sin_cos": [N_RES, 7, 2],
    "torsion_angles_mask": [N_RES, 7],
    # others
    "relative_positions": [N_RES, N_RES],
    "distogram": [N_RES, N_RES],
    "diffused_distogram": [N_RES, N_RES],
}

def base_config():
    return mlc.ConfigDict({
        # mode config
        "mode": {
            "train": {
                "crop": True,
                "supervised": True,
                "chunk_size": None,
                "block_size": None,
                "train_max_date": "2022-04-30",
                "database_weights": {
                    "monomer":  1,
                    "multimer": 2,
                    "afdb":     1,
                    "esmdb":    6,
                },
            },
            "runtime_eval": {
                "crop": True,
                "supervised": True,
                "chunk_size": None,
                "block_size": None,
            },
            "eval": {
                "crop": False,
                "supervised": True,
                "chunk_size": "auto",
                "block_size": "auto",
            },
            "predict": {
                "crop": False,
                "supervised": False,
                "chunk_size": "auto",
                "block_size": "auto",
            },
        },
        "data": {
            "raw_features": RAW_FEATURES,
            "batch_features": BATCH_FEATURES,
            "use_multimer": True,
            "max_chains": 18,
            "use_random_rotation": True,
            "save_esmdb_env": False,
            "relative_position_cutoff": relpos_k,
            "crop": {
                "crop_size": crop_size,
                "pad_size": pad_size,
                "spatial_crop_prob": 1.,
                "interface_threshold": 5.5,
                "inf": 1e5,
            },
            "diffusion": {
                "denoise_threshold": 1e-4,
                "scale_factor": "auto",
                "angle_distribution": {
                    "num_omega_bins": 2560, # > 100 is ok
                    "l_cutoff": 50,         # > 20 is ok, > 50 is accurate.
                    "std_threshold": 0.1,   # > 0.05 is ok
                    "eps": 1e-12,           # do not change.
                },
                "sequence": {
                    "num_classes": num_aatype,
                    "prior_type": "masking",    # set "customized" to use customized prior
                    "mask_index": -1,           # specify the index of mask, available only if "masking" is prior type
                    "customized_prior": None,   # a list of `num_aatype` floats that sums to 1.
                },
                "gamma_schema": {
                    "version": "v3",
                    "s": 0.01,
                    "K": 8,
                    "codesign_prob": 0.3,
                    "denovo_prob": 0.2,
                    "max_observed_prob": 0.84,
                    "observed_prob_kernel": "linear",
                    "r_0": 0.05,
                    "r_1": 0.95,
                }
            },
            "distogram": {
                "num_bins": distogram_bins,
                "min_bin": 2.3125,
                "max_bin": 21.6875,
            }
        },
        "model": {
            "globals": {
                "d_single": d_single,
                "d_pair": d_pair,
                "scale_factor": 10.,
                "chunk_size": None,
                "block_size": None,
                "eps": eps,
                "inf": inf,
                "use_float_input_embedder": False,
                "nan_check": 0,     # 0: disable; 1: give warnings; 2: error out
                "extreme_value_tolerance": 1e3,
            },
            "input_embedder": {
                "d_single": d_single,
                "d_pair": d_pair,
                "d_input": d_input,
                "relative_position_cutoff": relpos_k,
                "num_distogram_bins": distogram_bins,
            },
            "zormer": {
                "num_blocks": num_zormer,
                "use_gradient_ckpt": False,
                "evoformer": {
                    "num_blocks": 1,        # num blocks per zormer
                    "use_gradient_ckpt": True,
                    "d_single": d_single,
                    "d_pair": d_pair,
                    "d_hid_seq_attn": 32,
                    "d_hid_opm": 32,
                    "d_hid_mul": 128,
                    "d_hid_pair_att": 32,
                    "num_seq_attn_blocks": 1,
                    "num_heads_seq": 8,
                    "num_heads_pair": 4,
                    "transition_n": 4,
                    "use_triangle_update": False,
                    "use_triangle_attn": False,
                    "seq_dropout": 0.15,
                    "pair_dropout": 0.25,
                },
                "lite_structure_module": {
                    "d_single": d_single,
                    "d_pair": d_pair,
                    "num_transition_layers": 2,
                    "enhance_pair_with_attn": True,
                    "transition_dropout": 0.1,
                    "ipa_dropout": 0.1,
                    "ipa": {
                        "d_hid": 16,
                        "num_heads": 24,        # 16 * 24 = 384 = d_single
                        "num_qk_points": 8,
                        "num_v_points": 16,
                        "separate_kv": True,    # multimer
                        "bias": False,          # multimer
                        "eps": 1e-12,
                    }
                },
            },
            "sidechain_module": {
                "d_single": d_single,
                "d_hid_angle": 128,
                "num_resnet_blocks": 2,
                "num_angles": 7,
            },
            "heads": {
                "sequence_denoise": {
                    "d_single": d_single,
                    "num_classes": 21,
                    "share_across_stack": True,
                    "num_stacks": num_zormer,
                },
                "predicted_lddt": {
                    "num_bins": 50,
                    "d_in": d_single,
                    "d_hid": 128,
                },
                "predicted_aligned_error": {
                    "enabled": True,
                    "d_pair": d_pair,
                    "num_bins": distogram_bins,
                    "iptm_weight": 0.8,
                    "disable_enhance_head": False,
                },
                "distogram": {
                    "d_pair": d_pair,
                    "num_bins": distogram_bins,
                    "disable_enhance_head": False,
                },
                "experimentally_resolved": {
                    "enabled": True,
                    "d_single": d_single,
                    "d_out": 37,
                    "disable_enhance_head": False,
                },
            },
        },
        "loss": {
            "structure": {
                "weight": 1.0,
                "backbone": {
                    "weight": 0.8,
                    "loss_unit": 10.0,
                    "clamp_distance": 20.0,
                    "eps": 1e-6,
                    "layer_weighting": layer_weighting,
                },
                "sidechain": {
                    "weight": 0.2,
                    "loss_unit": 10.0,
                    "clamp_distance": 20.0,
                    "rot_weight": 0.25,  # 1. rot_weight = pos_weight
                    "eps": 1e-6,
                },
                "calculate_improve": True,
            },
            "backbone_clash": {
                "weight": 1.0,
                "ca_ca_threshold": 3.2,
                "layer_weighting": layer_weighting,
                "eps": 1e-6,
            },
            "sequence": {
                "weight": 2.0,
                "smooth_factor": 0.1,
                "layer_weighting": layer_weighting,
                "eps": 1e-6,
            },
            "distogram": {
                "weight": 0.3,
                "num_bins": distogram_bins,
                "smooth_factor": 0.1,
                "smooth_bin_cutoff": 2,
                "smooth_sigma": 1.,
                "eps": 1e-6,
            },
            "supervised_chi": {
                "weight": 1.0,
                "chi_weight": 0.5,
                "angle_norm_weight": 0.01,
                "eps": 1e-6,
            },
            "predicted_lddt": {
                "weight": 0.01,
                "num_bins": 50,
                "cutoff": 15.0,
                "smooth_factor": 0.1,
                "smooth_bin_cutoff": 2,
                "smooth_sigma": 1.,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "eps": 1e-10,
            },
            "predicted_aligned_error": {
                "weight": 0.01,
                "num_bins": 64,
                "max_bin": 31,
                "smooth_factor": 0.1,
                "smooth_bin_cutoff": 2,
                "smooth_sigma": 1.,
                "min_resolution": 0.1,
                "max_resolution": 3.0,
                "eps": 1e-10,
            },
            "violation": {
                "weight": 0.01,
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "bond_angle_loss_weight": 0.3,
                "eps": 1e-6,
            },
        },
    })


def recursive_set(c: mlc.ConfigDict, key: str, value: Any, ignore: str = None):
    with c.unlocked():
        for k, v in c.items():
            if ignore is not None and k == ignore:
                continue
            if isinstance(v, mlc.ConfigDict):
                recursive_set(v, key, value)
            elif k == key:
                c[k] = value


def model_config(name):
    c = base_config()

    if "large" in name:
        c = large(c)
    elif "small" in name:
        c = small(c)
    elif "tiny" in name:
        c = tiny(c)
    elif "v34" in name:
        c = match_v34(c)
    else:
        c = default(c)

    if "debug" in name:
        c = debug(c)

    if "ckz" in name:        # checkpoint on zormer
        c = grad_ckpt_zormer(c)
    elif "ckn" in name:
        c = grad_ckpt_none(c)
    else:
        c = grad_ckpt_evoformer(c)

    recursive_set(c, "inf", 3e4)
    recursive_set(c, "eps", 1e-5, "loss")
    return c

def large(c):
    c.data.crop.crop_size = 256
    c.model.zormer.num_blocks = 96
    c.model.globals.d_single = 768
    c.model.globals.d_pair = 384
    return c

def default(c):
    c.data.crop.crop_size = 128
    c.model.zormer.num_blocks = 64
    c.model.globals.d_single = 384
    c.model.globals.d_pair = 256
    return c

def match_v34(c):
    c.data.crop.crop_size = 128
    c.model.zormer.num_blocks = 64
    c.model.globals.d_single = 320
    c.model.globals.d_pair = 128
    return c

def small(c):
    c.data.crop.crop_size = 96
    c.model.zormer.num_blocks = 32
    c.model.globals.d_single = 256
    c.model.globals.d_pair = 128
    return c

def tiny(c):
    c.data.crop.crop_size = 64
    c.model.zormer.num_blocks = 8
    c.model.globals.d_single = 64
    c.model.globals.d_pair = 32
    return c

def debug(c):
    c.model.globals.nan_check = 2
    return c

def grad_ckpt_zormer(c):
    c.model.zormer.use_gradient_ckpt = True
    c.model.zormer.evoformer.use_gradient_ckpt = False
    return c

def grad_ckpt_evoformer(c):
    c.model.zormer.use_gradient_ckpt = False
    c.model.zormer.evoformer.use_gradient_ckpt = True
    return c

def grad_ckpt_none(c):
    c.model.zormer.use_gradient_ckpt = False
    c.model.zormer.evoformer.use_gradient_ckpt = False
    return c
