from diffold.nn.diffold import Diffold
from diffold.data import data_ops as ops
from diffold.data.utils import recur_print
from diffold.diffusion.transition import uniform_random_rotations

from diffold.test.test_nn import display

import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from unifold.modules.frame import Frame, Rotation, Quaternion

from diffold.config import model_config

config = model_config("tiny")
model = Diffold(config.model)

bsz, nres = 2, 12

def make_batch():
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
    seq_mask = torch.randint(
        0, 2, size=(bsz, nres), dtype=torch.float
    )
    
    f = Frame(
        Rotation(uniform_random_rotations((bsz, nres, 3, 3))),
        torch.randn((bsz, nres, 3))
    )
    quat = torch.cat([
        ops.rot_to_quat(f.get_rots().rot_mat),
        f.get_trans()
    ], dim=-1)
    return {
        "target_feat": target_feat,
        "diffused_distogram": distogram,
        "relative_positions": relpos,
        "seq_mask": seq_mask,
        "diffused_frames": f.to_tensor_4x4(),
        "diffused_quats": quat,
        "chain_id": torch.randint(0, 3, size=(bsz, nres), dtype=torch.long)
    }

def test_diffold():
    batch = make_batch()
    outs = model(batch)
    recur_print(outs)

test_diffold()


