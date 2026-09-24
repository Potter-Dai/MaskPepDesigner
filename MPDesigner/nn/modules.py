import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional
from functools import partial

# from unicore.modules import LayerNorm
from torch.nn import LayerNorm
from unicore.utils import permute_final_dims, one_hot

from ml_collections import ConfigDict

from unifold.modules.common import (
    Linear,
    residual,
    SimpleModuleList,
    Resnet,
    ResnetBlock,
)
from unifold.modules.evoformer import EvoformerStack
from unifold.modules.structure_module import (
    StructureModule,
    InvariantPointAttention,
)
from unifold.modules.frame import Frame, Rotation, Quaternion
from unifold.modules.structure_module import (
    StructureModuleTransition,
    BackboneUpdate,
)

from .base import RobustModule
from .evoformer import MPDesignerEvoformerStack


class InputEmbedder(RobustModule):
    def __init__(
        self,
        d_single,
        d_pair,
        d_input,
        relative_position_cutoff,
        num_distogram_bins,
        **base_cfg
    ):
        super().__init__(**base_cfg)
        self.linear_f = Linear(d_input, d_single)
        self.linear_f_zi = Linear(d_input, d_pair)
        self.linear_f_zj = Linear(d_input, d_pair)

        self.relpos_k = relative_position_cutoff
        self.num_relpos_cls = 2 * self.relpos_k + 2
        self.linear_relpos = Linear(self.num_relpos_cls, d_pair)

        self.num_distogram_bins = num_distogram_bins
        self.linear_distogram = Linear(self.num_distogram_bins, d_pair)

    def forward(
        self,
        target_feat: torch.Tensor,  # [*, L, :]
        relpos: torch.Tensor,       # [*, L, L, :]
        distogram: torch.Tensor,    # [*, L, L, :]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = target_feat.dtype
        s = self.linear_f(target_feat)

        relpos = relpos % self.num_relpos_cls   # negatives to positives
        relpos_1h = one_hot(relpos, self.num_relpos_cls, dtype)
        z = self.linear_relpos(relpos_1h)

        distogram_1h = one_hot(distogram, self.num_distogram_bins, dtype)
        z += self.linear_distogram(distogram_1h)

        zi = self.linear_f_zi(target_feat)
        zj = self.linear_f_zj(target_feat)
        z += (zi[..., None, :] + zj[..., None, :, :])

        return s, z


class LiteStructureModule(RobustModule):
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        num_transition_layers: int,
        enhance_pair_with_attn: bool,
        transition_dropout: float,
        ipa_dropout: float,
        ipa: ConfigDict,
        **base_cfg,
    ):
        super().__init__(**base_cfg)

        self.layer_norm_s = LayerNorm(d_single)
        self.layer_norm_z = LayerNorm(d_pair)

        self.linear_in = Linear(d_single, d_single)

        self.ipa = InvariantPointAttention(d_single, d_pair, **ipa)
        self.ipa_dropout = nn.Dropout(ipa_dropout)
        self.layer_norm_ipa = LayerNorm(d_single)

        self.transition = StructureModuleTransition(
            d_single,
            num_transition_layers,
            transition_dropout,
        )

        self.bb_update = BackboneUpdate(d_single)

        self.enhance_pair_with_attn = enhance_pair_with_attn
        if enhance_pair_with_attn:
            self.layer_norm_attn = LayerNorm(2 * ipa.num_heads) # cat i->j and j->i
            self.linear_attn = Linear(2 * ipa.num_heads, d_pair, init="final")

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        f: Frame,
        quat: Optional[Quaternion],
        pair_attn_mask: torch.Tensor,
    ):
        self.check_nan(
            [
                s, z,
                f.get_rots().rot_mat,
                f.get_trans(),
                quat.get_quats(),
                quat.get_trans(),
            ],
            key="lite_sm_inputs"
        )

        s_0, z_0 = s, z

        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        s = self.linear_in(s)

        ipa_out = self.ipa(
            s, z, f,
            square_mask=pair_attn_mask.unsqueeze(-3),   # for head dim
            return_attn=self.enhance_pair_with_attn
        )

        if self.enhance_pair_with_attn:
            s_prev = s
            s, attn = ipa_out
            self.check_nan([s, attn], [s_prev, z], key="ipa")

            attn = torch.cat([
                permute_final_dims(attn, inds=(1, 2, 0)),
                permute_final_dims(attn, inds=(2, 1, 0))
            ], dim=-1)
            attn = self.layer_norm_attn(attn)
            attn = self.linear_attn(attn)
            z = residual(z_0, attn, self.training)
            self.check_nan(z, [z_0, attn], key="enhance_pair")

        else:
            s = ipa_out
            attn = None
            z = z_0
            self.check_nan(s, [s, z], key="ipa")

        s = self.ipa_dropout(s)
        s = self.layer_norm_ipa(s)
        s = self.transition(s)

        up_vec = self.bb_update(s)
        self.check_nan(up_vec, s, key="bb_update", tolerance=50.)

        quat = quat.compose_update_vec(up_vec, pre_rot_mat=f.get_rots())
        f = Frame(Rotation(mat=quat.get_rot_mats()), quat.get_trans())

        s = residual(s_0, s, self.training)

        return s, z, f, quat


class ZormerBlock(nn.Module):
    def __init__(
        self,
        evoformer,
        lite_structure_module,
        **base_args,
    ):
        super().__init__()
        self.evoformer = MPDesignerEvoformerStack(**evoformer, **base_args)
        self.lite_sm = LiteStructureModule(**lite_structure_module, **base_args)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        f: Frame,
        quat: Quaternion,
        chunk_size: Optional[int] = None,
        block_size: Optional[int] = None,
        **masks
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        pair_attn_mask = masks.pop("pair_attn_mask")    # no need in evoformer

        s, z = self.evoformer(
            s, z, **masks,
            chunk_size=chunk_size,
            block_size=block_size,
        )
        s, z, f, quat = self.lite_sm(
            s, z, f, quat,
            pair_attn_mask=pair_attn_mask
        )
        return s, z, f, quat

# 4x tensors <-> frame & quat
def frames_to_tensors(f: Frame, quat: Quaternion):
    return (
        f.get_rots().rot_mat,
        f.get_trans(),
        quat.get_quats(),
        quat.get_trans()
    )

def tensors_to_frames(f_rot, f_trans, quat_rot, quat_trans):
    return (
        Frame(Rotation(f_rot), f_trans),
        Quaternion(quat_rot, quat_trans)
    )

class ZormerStack(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        use_gradient_ckpt: bool,
        **zormer,
    ):
        super().__init__()
        self.blocks = SimpleModuleList([
            ZormerBlock(**zormer) for _ in range(num_blocks)
        ])
        self.use_ckpt = use_gradient_ckpt

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        f: Frame,
        quat: Quaternion,
        chunk_size: Optional[int] = None,
        block_size: Optional[int] = None,
        **masks
    ):
        # apply fn to inputs, w/ or w/o ckpt
        def apply(func, *args, **kwargs):
            if torch.is_grad_enabled() and self.use_ckpt:
                return checkpoint(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)

        # wrap a block so the wrapper's inputs & outputs have no nested tensors.
        def get_block_wrapper(block):
            def wrapper(s, z, *frame_tensors):
                f, quat = tensors_to_frames(*frame_tensors)
                s, z, f, quat = block(
                    s, z, f, quat,
                    **masks,
                    chunk_size=chunk_size,
                    block_size=block_size,
                )
                return (s, z, *frames_to_tensors(f, quat))
            return wrapper

        funcs = [
            get_block_wrapper(b) for b in self.blocks
        ]

        traj = []
        s_stacked = []
        inputs = (s, z, *frames_to_tensors(f, quat))
        for i, func in enumerate(funcs):
            (
                s, z,
                f_rot, f_trans,
                quat_rot, quat_trans
            ) = apply(func, *inputs)
            traj.append(Frame(Rotation(f_rot), f_trans).unsqueeze(0))   # [1, *, L]
            s_stacked.append(s)
            if i < len(funcs) - 1:  # not last layer
                f_rot = f_rot.detach()
                quat_rot = quat_rot.detach()
            inputs = (s, z, f_rot, f_trans, quat_rot, quat_trans)

        f, quat = tensors_to_frames(*(inputs[2:]))
        traj = Frame.cat(traj, dim=0)
        s_stacked = torch.stack(s_stacked, dim=0)

        return s, z, f, quat, traj, s_stacked


class SidechainAngleResnet(nn.Module):
    def __init__(self, d_in, d_hid, num_blocks, num_angles):
        super().__init__()

        self.linear_in = Linear(d_in, d_hid)
        self.act = nn.GELU()
        self.layers = SimpleModuleList([
            ResnetBlock(d=d_hid) for _ in range(num_blocks)
        ])

        self.linear_out = Linear(d_hid, num_angles * 2)

    def forward(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        s = self.linear_in(self.act(s))
        for layer in self.layers:
            s = layer(s)
        s = self.linear_out(self.act(s))

        s = s.view(s.shape[:-1] + (-1, 2))
        unnormalized_s = s

        norm_denom = s.float().norm(dim=-1, keepdim=True)
        s = s.float() / norm_denom

        return unnormalized_s, s.type(unnormalized_s.dtype)


class SidechainModule(StructureModule):
    def __init__(
        self,
        d_single,
        d_hid_angle,
        num_resnet_blocks,
        num_angles,
        **kwargs,
    ):
        super(StructureModule, self).__init__()
        self.angle_resnet = SidechainAngleResnet(
            d_in=d_single,
            d_hid=d_hid_angle,
            num_blocks=num_resnet_blocks,
            num_angles=num_angles
        )
        self.frame_constants = None

    def forward(
        self,
        s: torch.Tensor,
        f: Frame,
        aatype: torch.Tensor,
    ):
        unnormalized_angles, angles = self.angle_resnet(s)

        all_frames = self.torsion_angles_to_frames(f, angles, aatype)

        atom14_pos = self.frames_and_literature_positions_to_atom14_pos(
            all_frames,
            aatype
        )

        return {
            "angles": angles,
            "unnormalized_angles": unnormalized_angles,
            "sidechain_frames": all_frames.to_tensor_4x4(),
            "atom14_positions": atom14_pos,
        }


# Compatibility names for existing downstream code.
DiffoldEvoformerStack = MPDesignerEvoformerStack
