import ml_collections as mlc
import copy
from typing import Any


N_RES = "number of residues"
N_ATOM = "number of atoms := 37"

d_single = mlc.FieldReference(384, field_type=int)
d_pair = mlc.FieldReference(128, field_type=int)

d_time = mlc.FieldReference(256, field_type=int)
num_time_rbf_bins = mlc.FieldReference(256, field_type=int)

scale_factor = mlc.FieldReference(20., field_type=float)

num_encoder_blocks = mlc.FieldReference(48, field_type=int)
num_decoder_iters = mlc.FieldReference(8, field_type=int)

eps = mlc.FieldReference(1e-8, field_type=float)
inf = mlc.FieldReference(3e4, field_type=float)

relpos_k = mlc.FieldReference(32, field_type=int)
plddt_bins = mlc.FieldReference(50, field_type=int)
pae_bins = mlc.FieldReference(64, field_type=int)
distogram_bins = mlc.FieldReference(64, field_type=int)

layer_weighting = mlc.FieldReference("uniform", field_type=str)
crop_size = mlc.FieldReference(128, field_type=int)
pad_size = mlc.FieldReference(4, field_type=int)

num_aatype_cls = mlc.FieldReference(21, field_type=int)
aalogits_rbf_bins = mlc.FieldReference(7, field_type=int)

d_feat_1d = num_aatype_cls * (1 + aalogits_rbf_bins) + 2
d_feat_2d = 2 * (relpos_k + 1) + distogram_bins + 2

decode_from_input_frames = mlc.FieldReference(False, field_type=bool)

RAW_FEATURES = {
    "uid": [None],
    "aatype": [N_RES],
    "residue_index": [N_RES],
    "all_atom_positions": [N_RES, N_ATOM, 3],
    "all_atom_mask": [N_RES, N_ATOM],
    "chain_id": [N_RES],
    "bfactor": [N_RES],
    "resolution": [1],
    "seq_gen_mask": [N_RES],
    "frame_gen_mask": [N_RES],
    "diffusion_t": [],
    "ligmask": [N_RES],  # add by DBT
    "hpmask": [N_RES],  # add by DBT
}

BATCH_FEATURES = {
    **RAW_FEATURES,
    # model inputs
    "diffusion_t": [],
    # "diffused_seq_logits": [N_RES, 21],
    "diffused_aalogits": [N_RES, 21],
    "diffused_frames": [N_RES, 4, 4],
    "diffused_quats": [N_RES, 7],
    "feat_1d": [N_RES, "feat_1d"],
    "feat_2d": [N_RES, N_RES, "feat_2d"],
    "time_feat": [N_RES, "time_feat"],
    "seq_mask": [N_RES],
    # supervisions.
    # labels
    "aalogits": [N_RES, 21],
    "frames": [N_RES, 4, 4],
    "frame_mask": [N_RES],
    # # atom14/37 constants
    # "residx_atom14_to_atom37": [N_RES, 14],
    # "residx_atom37_to_atom14": [N_RES, 37],
    # "atom14_atom_exists": [N_RES, 14],
    # "atom37_atom_exists": [N_RES, 37],
    # "atom14_gt_exists": [N_RES, 14],
    # "atom14_gt_positions": [N_RES, 14, 3],
    # "atom14_alt_gt_positions": [N_RES, 14, 3],
    # "atom14_alt_gt_exists": [N_RES, 14],
    # "atom14_atom_is_ambiguous": [N_RES, 14],
    # # rigidgroup constants
    # "rigidgroups_gt_frames": [N_RES, 8, 4, 4],
    # "rigidgroups_gt_exists": [N_RES, 8],
    # "rigidgroups_group_exists": [N_RES, 8],
    # "rigidgroups_group_is_ambiguous": [N_RES, 8],
    # "rigidgroups_alt_gt_frames": [N_RES, 8, 4, 4],
    # # torsion angles
    # "torsion_angles_sin_cos": [N_RES, 7, 2],
    # "alt_torsion_angles_sin_cos": [N_RES, 7, 2],
    # "torsion_angles_mask": [N_RES, 7],
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
                    # "monomer":  0.,
                    "multimer": 1.,
                    # "afdb":     0.,
                    # "esmdb":    0.,
                },
            },
            "runtime_eval": {
                "crop": True,
                "supervised": True,
                "chunk_size": None,
                "block_size": None,
            },
            "eval": {
                "crop": True,
                "supervised": True,
                "chunk_size": "auto",
                "block_size": "auto",
            },
            "predict": {
                "crop": True,
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
            "save_esmdb_env": False,
            "evaluate_on_training_set": False,      # only for debug
            "relative_position_cutoff": relpos_k,
            "is_cyclic": False,
            "aalogits_scale": 1.,
            "num_aatype_cls": num_aatype_cls,
            "num_time_rbf_bins": num_time_rbf_bins,
            "num_aalogits_rbf_bins": aalogits_rbf_bins,
            "make_initial_quaterions": decode_from_input_frames,
            "aalogits_rbf_max": 3.,
            "frame_augmentation": {
                "use_random_rotations": True,       # this doesn't matter. do for safety.
                "use_augment_unknown_frames": True,
                "translation_randomness": 2.,
                "relpos_sigma": 8.,
            },
            "crop": {
                "crop_size": crop_size,
                "pad_size": pad_size,
                "spatial_crop_prob": 1.,
                "interface_threshold": 5.5,
                "inf": 1e5,
            },
            "distogram": {
                "num_bins": distogram_bins,
                "min_bin": 2.3125,
                "max_bin": 21.6875,
            },
            "diffusion": {
                "position": {
                    "scale_factor": scale_factor,
                    "cos_s": 0.01,
                    "exponential_coef": 1.0,
                    "eps": 1e-8,
                },
                "sequence": {
                    "num_aatype": num_aatype_cls,
                    "scale_factor": 1.0,
                    "g_0": 0.82,
                    "t_eps": 1e-6,
                    "cos_s": 0.1,
                    "eps": 1e-8,
                },
                "rotation": {
                    "exponential_coef": 1.0,
                    "max_sigma_sq": 9.0,
                    "left": True,               # left gradient or right gradient
                    "rw_approx_thres": 0.6,
                    "num_omega_bins": 1000,
                    "l_cutoff": 128,
                    "igso3_gaussian_thres": 0.2,
                    "igso3_grad_gaussian_thres": 0.2,
                    "eps": 1e-12,
                },
            },
            "motif_schema": {
                "denovo_complex_prob": 0.,
                "codesign_complex_prob": 0.,
                "codesign_peptide_weight": 7.,
                "fold_peptide_weight": 1.5,
                "invfold_peptide_weight": 1.5,
                "use_true_sequence": False,
                "use_true_structure": False,
                "observed_prob_kernel": "linear_00-80",
                "motif_k": 8,
                "mask_type": "random",
            },
        },
        "model": {
            "globals": {
                "d_single": d_single,
                "d_pair": d_pair,
                "chunk_size": None,
                "block_size": None,
                "eps": eps,
                "inf": inf,
                "use_float_input_embedder": True,
                "nan_check": 0,     # 0: disable; 1: give warnings; 2: error out
                "extreme_value_tolerance": 1e3,
            },
            "scale_factor": scale_factor,
            "enable_decoder": True,
            "input_embedder": {
                "d_single": d_single,
                "d_pair": d_pair,
                "d_feat_1d": d_feat_1d,
                "d_feat_2d": d_feat_2d,
                "use_gated_linear": False,
            },
            "time_embedder": {
                "d_in": num_time_rbf_bins + 2,
                "d_out": d_time,
            },
            "zormer": {
                # unifold ipa config.
                "d_ipa": 16,
                "num_heads": 12,    # h * d = 192
                "num_qk_points": 4, # h * n * 3 = 192
                "num_v_points": 8,  # h * n * 3 = 384
                "use_point_bias": True,
                "use_rel_pts_update": False,
                "eps": 1e-8,
                # unifold evoformer config.
                "d_outprod_hid": 16,    # c * c = 256; reduced to half from unifold.
                "transition_n": 4,
                "s_dropout": 0.15,
                "z_dropout": 0.25,
            },
            "zormer_encoder": {
                "num_blocks": num_encoder_blocks,
                "checkpoint_ratio": 1.,             # 32GV100 + default = 0.6;
                "use_float_structures": False,
                "d_single": d_single,
                "d_pair": d_pair,
                "d_time": d_time,
            },
            "zormer_decoder": {
                "num_iter": num_decoder_iters,
                "use_hard_motif": True,
                "decode_from_input_frames": decode_from_input_frames,
                "use_float_structures": True,
                "use_gated_linear": False,
                "d_single": d_single,
                "d_pair": d_pair,
                "d_time": d_time,
                "scale_factor": scale_factor,
                "ipa_dropout": 0.1,
            },
            "heads": {
                "sequence_denoise": {
                    "enabled": True,
                    "d_single": d_single,
                    "num_classes": 21,  # do not give me "unk" as a result.
                },
                "predicted_lddt": {
                    "enabled": True,
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
                    "enabled": True,
                    "d_pair": d_pair,
                    "num_bins": distogram_bins,
                    "disable_enhance_head": False,
                },
                # "experimentally_resolved": {
                #     "enabled": False,
                #     "d_single": d_single,
                #     "d_out": 37,
                #     "disable_enhance_head": False,
                # },
            },
        },
        "loss": {
            "plddt_cutoff": 0.55,   # approx. = Q1 (skip 25%)
            "structure": {
                "weight": 1.0,
                "use_euclidean": False,
                "motif_weight": 0.0,
                "loss_type": "hybrid",
                "t_scale": 1.0,
                "gen_region_only": decode_from_input_frames,
                "rotation_weight": 0.5,
                "loss_unit": 10.0,
                "clamp_distance": 20.0,
                "eps": 1e-6,
                "layer_weighting": layer_weighting,
            },
            "backbone_clash": {
                "weight": 0.1,
                "ca_ca_threshold": 3.2,
                "layer_weighting": layer_weighting,
                "eps": 1e-8,
                "t_scale": 1.0,
            },
            "adjacent_ca": {
                "weight": 0.1,
                "min_distance": 3.2,
                "min_weight": 3.0,
                "max_distance": 4.0,
                "clamp_distance": 4.0,
                "layer_weighting": layer_weighting,
                "eps": 1e-8,
                "t_scale": 1.0,
            },
            "sequence": {
                "weight": 1.0,
                "motif_weight": 0.1,
                "eps": 1e-6,
                "t_scale": 1.0,
            },
            "distogram": {
                "weight": 0.3,
                "num_bins": distogram_bins,
                "smooth_factor": 0.1,
                "smooth_bin_cutoff": 2,
                "smooth_sigma": 1.,
                "eps": 1e-6,
                "t_scale": 1.0,
            },
            # "supervised_chi": {
            #     "weight": 1.0,
            #     "chi_weight": 0.5,
            #     "angle_norm_weight": 0.01,
            #     "eps": 1e-6,
            # },
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
                "t_scale": 1.0,
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
                "t_scale": 1.0,
            },
            # "violation": {
            #     "weight": 0.01,
            #     "violation_tolerance_factor": 12.0,
            #     "clash_overlap_tolerance": 1.5,
            #     "bond_angle_loss_weight": 0.3,
            #     "eps": 1e-6,
            # },
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
    else:
        c = default(c)
    
    if "fif" in name:
        c = from_input_frames(c)

    if "seq_only" in name:
        c = sequence_only(c)
    elif "struct_only" in name:
        c = structure_only(c)

    if "no_viol" in name:
        c = no_viol_loss(c)

    if "uf_data" in name:
        c = uf_data(c)

    if "debug" in name:
        c = debug(c)
    
    if "scale_seqdiff" in name:
        c = scaled_seqdiff(c)
    
    if "EOT" in name:
        c = eval_on_train(c)

    if "cyclic" in name:
        c = is_cyclic(c)

    if "v62" in name:
        c = v62(c)
    elif "v63" in name:
        c = v63(c)
    elif "v64" in name:
        c = v63(c)      # v63 with all bugs fixed and various optimization.
    elif "v65" in name:
        c = v65(c)
    elif "v66" in name:
        c = v66(c)
    elif "denovo" in name:
        c = denovo(c)
    else:
        raise ValueError(f"must specify model version in model name. got {name}")

    recursive_set(c, "inf", 3e4)
    recursive_set(c, "eps", 1e-5, "loss")
    return c

def large(c):
    c.model.zormer_encoder.num_blocks = 64
    c.model.globals.d_single = 512
    c.model.globals.d_pair = 384
    c.model.zormer.num_heads = 16
    return c

def default(c):
    return c

def small(c):
    c.data.crop.crop_size = 128
    c.model.zormer_encoder.num_blocks = 32
    c.model.zormer_decoder.num_iter = 4
    c.model.globals.d_single = 256
    c.model.globals.d_pair = 128
    return c

def tiny(c):
    c.data.crop.crop_size = 128
    c.model.zormer_encoder.num_blocks = 8
    c.model.zormer_decoder.num_iter = 5
    c.model.globals.d_single = 256
    c.model.globals.d_pair = 128
    return c

def debug(c):
    c.model.globals.nan_check = 2
    return c

def no_viol_loss(c):
    c.loss.backbone_clash.weight = 0.
    c.loss.adjacent_ca.weight = 0.
    return c

def from_input_frames(c):
    c.data.make_initial_quaterions = True
    c.model.zormer_decoder.decode_from_input_frames = True
    return c

def uf_data(c):
    c.mode.train.database_weights.monomer = 0.
    c.mode.train.database_weights.esmdb = 0.
    return c

def no_auxillary(c):
    c.model.heads.predicted_lddt.enabled = False
    c.model.heads.predicted_aligned_error.enabled = False
    c.model.heads.distogram.enabled = False
    c.loss.predicted_lddt.weight = 0.
    c.loss.predicted_aligned_error.weight = 0.
    c.loss.distogram.weight = 0.
    return c

def sequence_only(c):
    c.data.motif_schema.use_true_structure = True
    c = no_auxillary(c)
    c.model.enable_decoder = False
    c.loss.structure.weight = 0.
    c = no_viol_loss(c)
    return c

def structure_only(c):
    c.data.motif_schema.use_true_sequence = True
    # c = no_auxillary(c)
    c.model.heads.sequence_denoise.enabled = False
    c.loss.sequence.weight = 0.
    c = no_viol_loss(c)
    return c

def scaled_seqdiff(c):
    c.data.diffusion.sequence.scale_factor = 0.33
    c.data.diffusion.sequence.g_0 = 0.25
    c.data.diffusion.sequence.cos_s = 1.
    c.data.aalogits_rbf_max = 1.
    return c

def is_cyclic(c):
    c.data.is_cyclic = True
    return c

def v62(c):
    c.model.zormer.use_rel_pts_update = True
    c = from_input_frames(c)
    c = no_viol_loss(c)
    c = scaled_seqdiff(c)
    return c

def v63(c):
    c = scaled_seqdiff(c)
    c = from_input_frames(c)
    c.model.zormer.use_rel_pts_update = True
    c.loss.structure.use_euclidean = True
    c.model.input_embedder.use_gated_linear = True
    c.model.zormer_decoder.use_gated_linear = True
    c.data.motif_schema.motif_k = 15
    c.data.motif_schema.observed_prob_kernel = "linear_00-50"
    c.model.globals.use_float_structures = False
    c.model.globals.use_float_input_embedder = False
    c.data.diffusion.position.exponential_coef = 1.8
    c.data.diffusion.rotation.exponential_coef = 4.0
    c.data.diffusion.rotation.max_sigma_sq = 9.0
    return c

def eval_on_train(c):
    c.data.evaluate_on_training_set = True
    return c

def v65(c):
    c = v63(c)
    c.model.zormer_decoder.use_hard_motif = False
    c.loss.structure.motif_weight = 0.1
    c.loss.structure.loss_type = "hybrid"
    c.loss.backbone_clash.weight = 0.5
    c.loss.adjacent_ca.weight = 0.5
    return c

def v66(c):
    c = v65(c)
    c.data.diffusion.position.beta_clip = 0.99
    c.data.diffusion.sequence.beta_clip = 0.99
    return c

def denovo(c):
    c = v66(c)
    recursive_set(c, "t_scale", 0.1)
    c.data.motif_schema.mask_type = "chain_id"
    return c
