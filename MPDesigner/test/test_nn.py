import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from unifold.modules.attentions import gen_tri_attn_mask, gen_attn_mask
from unifold.modules.frame import Frame, Rotation, Quaternion

from MPDesigner.config import model_config
from MPDesigner.data import data_ops as ops
from MPDesigner.diffusion.transition import uniform_random_rotations
from MPDesigner.nn.modules import (
    InputEmbedder,
    MPDesignerEvoformerStack,
    LiteStructureModule,
    SidechainModule,
    frames_to_tensors,
)

config = model_config("tiny")
d_single = config.model.globals.d_single
d_pair = config.model.globals.d_pair

base_args = {
    "enable_nan_check": True,
    "tolerance": 1e3,
    "strict": True,
}

bsz, nres = 2, 12

s_shape = (bsz, nres, d_single)
z_shape = (bsz, nres, nres, d_pair)
frame_shape = (bsz, nres)

def get_extreme_vals(*args):
    return tuple(a.abs().max().item() for a in args)

def get_stddevs(*args):
    return tuple(a.std().item() for a in args)

def get_shapes(*args):
    return tuple(tuple(a.shape) for a in args)

def display(key, *out):
    logger.info(f"{key} runnable.\n"
                f"\toutput shapes: {get_shapes(*out)}\n"
                f"\toutput extreme values: {get_extreme_vals(*out)}\n"
                f"\toutput stds: {get_stddevs(*out)}")

def make_masks(inf=3e4):
    seq_mask = torch.randint(0, 2, size=(bsz, nres))
    seq_attn_mask = gen_attn_mask(seq_mask, -inf)
    pair_mask = seq_mask[..., None] == seq_mask[..., None, :]
    pair_attn_mask = gen_attn_mask(pair_mask, -inf)
    tri_start, tri_end = gen_tri_attn_mask(pair_mask, inf)
    masks = {
        "seq_mask": seq_mask,
        "seq_attn_mask": seq_attn_mask,
        "pair_mask": pair_mask,
        "pair_attn_mask": pair_attn_mask,
        "tri_start_attn_mask": tri_start,
        "tri_end_attn_mask": tri_end,
    }
    return masks

def test_input_emb():
    input_emb = InputEmbedder(
        **config.model.input_embedder,
        **base_args,
    )
    target_feat = torch.randn(
        size=(bsz, nres, config.model.input_embedder.d_input)
    )
    distogram = torch.randint(
        low=0,
        high=config.data.distogram.num_bins,
        size=(bsz, nres, nres),
        dtype=torch.long,
    )
    relpos = torch.randint(
        low=-config.data.relative_position_cutoff-1,
        high=config.data.relative_position_cutoff+1,
        size=(bsz, nres, nres),
        dtype=torch.long,
    )
    s, z = input_emb(target_feat, relpos, distogram)
    assert s.shape == s_shape, s.shape
    assert z.shape == z_shape, z.shape
    display("input_emb", s, z)
    return True

def test_evoformer():
    evoformer = MPDesignerEvoformerStack(
        **config.model.zormer.evoformer,
        **base_args,
    )
    masks = make_masks()
    masks.pop("pair_attn_mask")
    s = torch.randn(
        (bsz, nres, d_single)
    )
    z = torch.randn(
        (bsz, nres, nres, d_pair)
    )
    s, z = evoformer(s, z, **masks)
    assert s.shape == s_shape, s.shape
    assert z.shape == z_shape, z.shape
    display("evoformer", s, z)
    return True

def test_lite_sm():
    lite_sm = LiteStructureModule(
        **config.model.zormer.lite_structure_module,
        **base_args,
    )
    s = torch.randn(
        (bsz, nres, d_single)
    )
    z = torch.randn(
        (bsz, nres, nres, d_pair)
    )
    f = Frame(
        Rotation(uniform_random_rotations((bsz, nres, 3, 3))),
        torch.randn((bsz, nres, 3))
    )
    quat = Quaternion(
        ops.rot_to_quat(f.get_rots().rot_mat),
        f.get_trans()
    )
    pair_attn_mask = make_masks()["pair_attn_mask"]
    s, z, f, quat = lite_sm(s, z, f, quat, pair_attn_mask)
    assert s.shape == s_shape, s.shape
    assert z.shape == z_shape, z.shape
    assert f.shape == quat.shape == frame_shape
    display("lite_sm", s, z, *frames_to_tensors(f, quat))
    return True

def test_sidechain():
    sc_module = SidechainModule(
        **config.model.sidechain_module,
        **base_args,
    )
    s = torch.randn(
        (bsz, nres, d_single)
    )
    f = Frame(
        Rotation(uniform_random_rotations((bsz, nres, 3, 3))),
        torch.randn((bsz, nres, 3))
    )
    aatype = torch.randint(0, 21, size=(bsz, nres))
    ret = sc_module(s, f, aatype)
    display("sc_module", *tuple(ret.values()))


test_input_emb()
test_evoformer()
test_lite_sm()
test_sidechain()
