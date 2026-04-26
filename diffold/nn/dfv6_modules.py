import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Tuple, Optional

import math

from ml_collections import ConfigDict

from unicore.modules import LayerNorm
# from torch.nn import LayerNorm
from unicore.utils import permute_final_dims, dict_multimap
from functools import partial

from unifold.modules.common import (
    Linear,
    Transition,
    residual,
    SimpleModuleList,
    Resnet,
    ResnetBlock,
)

from unifold.modules.frame import Frame, Rotation, Quaternion
from unifold.modules.structure_module import (
    StructureModuleTransition,
    BackboneUpdate,
    InvariantPointAttention,
)

from .base import RobustModule

from unicore.modules import LayerNorm, softmax_dropout

from unicore.utils import checkpoint_sequential

from unifold.modules.structure_module import (
    BackboneUpdate,
    SidechainAngleResnet,
)


class InputEmbedder(RobustModule):
    def __init__(
        self,
        d_single,
        d_pair,
        d_feat_1d,
        d_feat_2d,
        use_gated_linear: bool,
        **base_cfg
    ):
        super().__init__(**base_cfg)
        linear = Linear if not use_gated_linear else GatedLinear
        self.linear_s = linear(d_feat_1d, d_single)
        self.linear_z = linear(d_feat_2d, d_pair)
        self.linear_s_zi = linear(d_feat_1d, d_pair)
        self.linear_s_zj = linear(d_feat_1d, d_pair)

    def forward(
        self,
        feat1d: torch.Tensor,      # [*, L, :]
        feat2d: torch.Tensor,      # [*, L, L, :]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        s = self.linear_s(feat1d)
        z = self.linear_z(feat2d)
        zi = self.linear_s_zi(feat1d)
        zj = self.linear_s_zj(feat1d)
        z += (zi[..., :, None, :] + zj[..., None, :, :])

        return s, z


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class OuterProductTransition(RobustModule):
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_hid: int,
        **base_args
    ):
        super().__init__(**base_args)

        self.d_single = d_single
        self.d_pair = d_pair
        self.d_hid = d_hid

        self.layer_norm = LayerNorm(d_single)
        self.linear_1 = Linear(d_single, d_hid)
        self.linear_2 = Linear(d_single, d_hid)
        self.linear_out = Linear(d_hid**2, d_pair, init="relu")
        self.act = nn.GELU()
        self.linear_z = Linear(self.d_pair, self.d_pair, init="final")
        self.layer_norm_out = LayerNorm(self.d_pair)

    def forward(
        self,
        s: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:

        s = self.layer_norm(s)
        a = self.linear_1(s)
        b = self.linear_2(s)
        op = torch.einsum("...ia,...jb->...ijab", a, b)
        op = op.contiguous().view(*op.shape[:-2], -1)   # ...ij(ab)

        z = self.linear_out(op)
        z = self.act(z)
        z = self.layer_norm_out(z)
        z = self.linear_z(z)
        z *= pair_mask[..., None]

        return z


class ZormerBlock(RobustModule):
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_time: int,
        d_ipa: int,
        d_outprod_hid: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        transition_n: int,
        s_dropout: float,
        z_dropout: float,
        use_rel_pts_update: bool,
        use_point_bias: bool = True,
        eps: float = 1e-8,
        **base_args
    ):
        super().__init__(**base_args)

        self.layer_norm_s = LayerNorm(d_single)
        self.layer_norm_z = LayerNorm(d_pair)

        self.linear_t = Linear(d_time, d_single, init="final")

        self.ipa = IPAv6(
            d_single=d_single,
            d_pair=d_pair,
            d_ipa=d_ipa,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            use_point_bias=use_point_bias,
            use_rel_pts_update=use_rel_pts_update,
            eps=eps,
            **base_args
        )

        self.s_dropout = nn.Dropout(s_dropout)
        self.z_dropout = nn.Dropout(z_dropout)

        self.s_transition = Transition(
            d_in=d_single,
            n=transition_n,
        )
        self.z_transition = Transition(
            d_in=d_pair,
            n=transition_n,
        )

        self.outprod = OuterProductTransition(
            d_single=d_single,
            d_pair=d_pair,
            d_hid=d_outprod_hid,
        )

    def forward(
        self,
        s: torch.Tensor,            # *is
        z: torch.Tensor,            # *ijz
        t_emb: torch.Tensor,            # *it
        frames: Frame,                   # *i
        pair_mask: torch.Tensor,    # *ij
        attn_mask: torch.Tensor,    # *ij
    ) -> torch.Tensor:
        
        sx = self.layer_norm_s(s) + self.linear_t(t_emb)
        zx = self.layer_norm_z(z)

        sx, zx = self.ipa(sx, zx, frames, attn_mask)
        self.check_nan((sx, zx), (s, z), key="ipa")

        sx = self.s_dropout(sx)
        zx = self.z_dropout(zx)

        s = residual(s, sx, self.training)
        z = residual(z, zx, self.training)

        s = residual(s, self.s_transition(s), self.training)

        z = residual(z, self.outprod(s, pair_mask), self.training)
        self.check_nan(z, s, key="outprod")
        z = residual(z, self.z_transition(z), self.training)

        return s, z, t_emb      # t_emb not altered.


class IPAv6(RobustModule):
    """
    IPA takes care of linear attentions only.
    do layer norms before ipa; do transition after ipa.
    """
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_ipa: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        use_rel_pts_update: bool,
        use_point_bias: bool = True,
        eps: float = 1e-8,      # for pts norm calculation.
        **base_args,
    ):
        super().__init__(**base_args)

        self.d_hid = d_ipa
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.eps = eps

        # self attention
        hc = self.d_hid * self.num_heads
        self.linear_q = Linear(d_single, hc, init="glorot")
        self.linear_k = Linear(d_single, hc, init="glorot")
        self.linear_v = Linear(d_single, hc, init="glorot")
        self.norm_attn = (1.0 / (3 * self.d_hid)) ** (-0.5)

        # biased attention
        self.linear_b = Linear(d_pair, self.num_heads, init="normal")
        self.norm_bias_attn = (1.0 / 3) ** (-0.5)

        # point attention
        hpqk = self.num_heads * self.num_qk_points * 3
        hpv = self.num_heads * self.num_v_points * 3
        self.linear_q_points = Linear(d_single, hpqk, bias=use_point_bias, init="glorot")
        self.linear_k_points = Linear(d_single, hpqk, bias=use_point_bias, init="glorot")
        self.linear_v_points = Linear(d_single, hpv, bias=use_point_bias, init="glorot")
        self.softplus = nn.Softplus()
        self.head_weights = nn.Parameter(torch.zeros((num_heads)))
        ipa_point_weights_init_(self.head_weights)
        self.norm_pt_att = -0.5 * (1.0 / (3 * (self.num_qk_points * 9.0 / 2))) **(-0.5)

        # triangle attention for z
        self.linear_z_v_left = Linear(d_pair, hc, init="glorot")
        self.linear_z_v_right = Linear(d_pair, hc, init="glorot")

        # gate & output linears
        self.linear_pair = Linear(d_pair, hc)
        s_out_dim = self.num_heads * (self.d_hid * 2 + self.num_v_points * 4)
        self.linear_g_s = Linear(d_single, s_out_dim, init="gating")
        self.sigmoid_s = nn.Sigmoid()
        self.linear_o_s = Linear(s_out_dim, d_single, init="final")
        self.use_rel_pts_update = use_rel_pts_update
        if use_rel_pts_update:
            z_out_dim = 2 * hc + hpv
        else:
            z_out_dim = 2 * hc
        self.linear_g_z = Linear(d_pair, z_out_dim, init="gating")
        self.sigmoid_z = nn.Sigmoid()
        self.linear_o_z = Linear(z_out_dim, d_pair, init="final")
        
    def forward(
        self,
        s: torch.Tensor,            # *is
        z: torch.Tensor,            # *ijz
        f: Frame,                   # *if, in float.
        attn_mask: torch.Tensor,    # *ij
    ) -> torch.Tensor:
        def h_linear(layer, x):     # *ia -> *ihb
            return layer(x).view(*x.shape[:-1], self.num_heads, -1)
        
        def pts_linear(layer, x, num_pts):  # *is -> *ihp3
            return layer(x).view(*x.shape[:-1], self.num_heads, num_pts, 3).float()

        def local_to_global(pts):   # *ihp3 -> *ihp3
            # s.dtype -> f.dtype
            return f[..., None, None].apply(pts.to(f.dtype))
        
        # def global_to_local(pts):   # *ihp3 -> *ihp3
        #     # f.dtype -> s.dtype
        #     return f[..., None, None].invert_apply(pts).to(s.dtype)

        # 1. calculate q, k & bias
        q = h_linear(self.linear_q, s)
        k = h_linear(self.linear_k, s)
        q_pts_global = local_to_global(pts_linear(self.linear_q_points, s, self.num_qk_points))
        k_pts_global = local_to_global(pts_linear(self.linear_k_points, s, self.num_qk_points))

        bias = permute_final_dims(self.linear_b(z), (2, 0, 1))  # *hij

        # 2. calculate attention matrix
        attn = torch.einsum("...ihd,...jhd->...hij", q, k) * self.norm_attn
        del q, k

        attn += attn_mask[..., None, :, :]
        attn = residual(attn, bias * self.norm_bias_attn, self.training)

        pt_att = q_pts_global.unsqueeze(-4) - k_pts_global.unsqueeze(-5)    # *ijhp3
        self.check_nan(pt_att, (q_pts_global, k_pts_global), key="pt_att")
        del q_pts_global, k_pts_global

        pt_att = (pt_att ** 2).sum(dim=(-1, -2)).to(z.dtype)   # *ijhp3 -> *ijh

        head_weights = self.softplus(self.head_weights)
        pt_att *= (head_weights * self.norm_pt_att)
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        attn = residual(attn, pt_att, self.training)
        attn = torch.softmax(attn, dim=-1)

        # 3. apply attention matrix to values

        # 3.1 self attention
        v = h_linear(self.linear_v, s)  # *ihv
        o = torch.einsum("...hij,...jhd->...ihd", attn, v)
        o = o.contiguous().view(*o.shape[:-2], -1)  # *io
        self.check_nan(o, (v, attn), key="self_attn")
        del v

        # 3.2 point attention
        v_pts_global = local_to_global(pts_linear(self.linear_v_points, s, self.num_v_points))
        o_pts_global = torch.einsum("...hij,...jhpr->...ihpr", attn.to(f.dtype), v_pts_global)
        o_pts_global = o_pts_global.contiguous().view(*o_pts_global.shape[:-3], -1, 3)  # *i(hp)3
        # o_pts = global_to_local(o_pts_global)
        o_pts = f[..., None].invert_apply(o_pts_global)
        self.check_nan((o_pts, o_pts_global), (v_pts_global, attn), key="o_pts")
        del v_pts_global

        if self.training:
            o_pts_norm = torch.sqrt(torch.sum(o_pts.float() ** 2, dim=-1) + self.eps)
        else:
            o_pts_norm = torch.sqrt(torch.sum(o_pts ** 2, dim=-1) + self.eps)
        self.check_nan(o_pts_norm, o_pts, key="o_pts_norm")
        # o_pts_norm = o_pts_norm.view(*o_pts_norm.shape[:-2], -1)
        o_pts = o_pts.view(*o_pts.shape[:-2], -1)   # *i(hp3)

        # 3.3 aggregate pair repr into s
        v_pair = h_linear(self.linear_pair, z)
        o_pair = torch.einsum("...hij,...ijhd->...ihd", attn, v_pair)
        o_pair = o_pair.contiguous().view(*o_pair.shape[:-2], -1)
        del v_pair

        # 3.4 gated output
        o_s = torch.cat((o, o_pts.to(s.dtype), o_pts_norm.to(s.dtype), o_pair), dim=-1)
        del o, o_pts, o_pts_norm, o_pair
        g_s = self.sigmoid_s(self.linear_g_s(s))
        s = self.linear_o_s(g_s * o_s)

        # 4. triangle attention for pair repr.
        v_z_tri = torch.concat((
            h_linear(self.linear_z_v_left, z),
            h_linear(self.linear_z_v_right, z.transpose(-2, -3))
        ), dim=-1)
        o_z = torch.einsum("...hik,...kjhd->...ijhd", attn, v_z_tri)
        o_z = o_z.contiguous().view(*o_z.shape[:-2], -1)    # *ijd
        del v_z_tri

        if self.use_rel_pts_update:
            o_rel_pts = f[..., None, None].invert_apply(o_pts_global[..., None, :, :, :])   # *i11f,*1jn3->*ijn3
            o_rel_pts = o_rel_pts.contiguous().view(*o_rel_pts.shape[:-2], -1)  # *ij(n3)
            o_z = torch.cat((o_z, o_rel_pts.to(s.dtype)), dim=-1)

        g_z = self.sigmoid_z(self.linear_g_z(z))
        z = self.linear_o_z(g_z * o_z)

        return s, z


class ZormerEncoderStack(RobustModule):
    def __init__(
        self,
        num_blocks: int,
        checkpoint_ratio: float,
        d_single: int,
        d_pair: int,
        d_time: int,
        d_ipa: int,
        d_outprod_hid: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        transition_n: int,
        s_dropout: float,
        z_dropout: float,
        use_rel_pts_update: bool,
        use_float_structures: bool,
        use_point_bias: bool = True,
        eps: float = 1e-8,
        **base_args
    ):
        super().__init__(**base_args)

        self.blocks = SimpleModuleList([
            ZormerBlock(
                d_single=d_single,
                d_pair=d_pair,
                d_time=d_time,
                d_ipa=d_ipa,
                d_outprod_hid=d_outprod_hid,
                num_heads=num_heads,
                num_qk_points=num_qk_points,
                num_v_points=num_v_points,
                transition_n=transition_n,
                s_dropout=s_dropout,
                z_dropout=z_dropout,
                use_rel_pts_update=use_rel_pts_update,
                use_point_bias=use_point_bias,
                eps=eps,
                **base_args,
            ) for _ in range(num_blocks)
        ])
        self.checkpoint_ratio = checkpoint_ratio
        self.use_float_structures = use_float_structures

    def forward(
        self,
        s: torch.Tensor,            # *is
        z: torch.Tensor,            # *ijz
        t_emb: torch.Tensor,     # *it
        frames: Frame,              # *i
        pair_mask: torch.Tensor,    # *ij
        attn_mask: torch.Tensor,    # *ij
    ) -> torch.Tensor:
        # change frames dtype.
        if self.use_float_structures:
            frames = frames.to(dtype=torch.float)
        else:
            frames = frames.to(dtype=s.dtype)

        blocks = [
            partial(
                b,
                frames=frames,
                pair_mask=pair_mask,
                attn_mask=attn_mask
            )
            for b in self.blocks
        ]

        for i, b in enumerate(blocks):
            if self.training and (i < self.checkpoint_ratio * len(self.blocks)):
                s, z, t_emb = torch.utils.checkpoint.checkpoint(b, s, z, t_emb)
            else:
                s, z, t_emb = b(s, z, t_emb)

        return s, z


class ZormerDecoderStack(RobustModule):
    def __init__(
        self,
        num_iter: int,
        d_single: int,
        d_pair: int,
        d_time: int,
        d_ipa: int,
        d_outprod_hid: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        transition_n: int,
        s_dropout: float,
        z_dropout: float,
        ipa_dropout: float,
        use_rel_pts_update: bool,
        use_point_bias: bool,
        scale_factor: int,
        use_float_structures: bool,
        decode_from_input_frames: bool,
        use_gated_linear: bool,
        use_hard_motif: bool = False,
        # d_angle: int,
        # num_angle_blocks: int,
        # num_angles: int = 7,
        eps: float = 1e-8,
        **base_args
    ):
        super().__init__(**base_args)

        self.num_iter = num_iter
        self.scale_factor = scale_factor
        self.use_float_structures = use_float_structures

        self.use_hard_motif = use_hard_motif

        self.decode_from_input_frames = decode_from_input_frames

        self.layer_norm_s = LayerNorm(d_single)
        self.layer_norm_z = LayerNorm(d_pair)

        self.zormer = ZormerBlock(
            d_single=d_single,
            d_pair=d_pair,
            d_time=d_time,
            d_ipa=d_ipa,
            d_outprod_hid=d_outprod_hid,
            num_heads=num_heads,
            num_qk_points=num_qk_points,
            num_v_points=num_v_points,
            transition_n=transition_n,
            s_dropout=s_dropout,
            z_dropout=z_dropout,
            use_rel_pts_update=use_rel_pts_update,
            use_point_bias=use_point_bias,
            eps=eps,
            **base_args,
        )

        self.zormer_dropout = nn.Dropout(ipa_dropout)
        self.zormer_layernorm = LayerNorm(d_single)
        if use_gated_linear:
            self.bb_update = GatedLinear(d_single*2, 6, init="final")
        else:
            self.bb_update = BackboneUpdate(d_single*2)
        # self.angle_resnet = SidechainAngleResnet(d_single, d_angle, num_angle_blocks, num_angles)

    def forward(
        self,
        s: torch.Tensor,            # *is
        z: torch.Tensor,            # *ijz
        t_emb: torch.Tensor,     # *it
        pair_mask: torch.Tensor,    # *ij
        attn_mask: torch.Tensor,    # *ij
        frames: Frame = None,       # *i
        quats: Quaternion = None,   # *i
        frame_gen_mask: torch.Tensor = None,    # *i
    ) -> torch.Tensor:
        
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)

        # s_init, z_init = s, z
        s_init = s.clone()

        frame_dtype = torch.float if self.use_float_structures else s.dtype
        if not self.decode_from_input_frames:
            assert frames is None and quats is None, \
                "must NOT provide frames / quaternions if not decoding from frames."
            assert frame_gen_mask is None, "regenerate all frames."
            quats = Quaternion.identity(
                s.shape[:-1], frame_dtype, s.device,
                requires_grad=False,
            )
            frames = Frame.identity(
                s.shape[:-1], frame_dtype, s.device,
                requires_grad=False,
            )
        else:
            # raise NotImplementedError("do not support decode from given currently.")
            assert quats is not None and frames is not None, \
                "must provide frames & quaternions if decoding from frames."
            quats = quats.to(dtype=frame_dtype)
            frames = frames.to(dtype=frame_dtype)

        outputs = []
        for i in range(self.num_iter):
            s, z, t_emb = self.zormer(
                s, z, t_emb, frames, pair_mask, attn_mask
            )    # notably zormer block is residual.
            s = self.zormer_dropout(s)
            s = self.zormer_layernorm(s)
            update_vec = self.bb_update(torch.cat((s, s_init), dim=-1))
            # in v6 we skip angles first.
            # unnormed_angles, angles = self.angle_resnet(s, s_init)

            if self.use_hard_motif:
                assert frame_gen_mask is not None, "must provide frame gen mask if use hard motif."
                update_vec *= frame_gen_mask[..., None]
            update_vec = update_vec.to(frame_dtype)

            quats = quats.compose_update_vec(
                update_vec.to(frame_dtype), pre_rot_mat=frames.get_rots()
            )
            frames = Frame(Rotation(quats.get_rot_mats()), quats.get_trans())

            preds = {
                # "single": s,
                "frames": frames.scale_translation(self.scale_factor).to_tensor_4x4(),
                # "unnormalized_angles": unnormed_angles,
                # "angles": angles,
            }
            outputs.append(preds)

            if i < self.num_iter - 1:
                frames = frames.stop_rot_gradient()
                quats = quats.stop_rot_gradient()
            
        ret = dict_multimap(torch.stack, outputs)

        return ret


class AllAtomModule(RobustModule):
    def __init__(self, **base_args):
        super().__init__(**base_args)

    def forward():
        raise NotImplementedError("in v6 we do not consider all atom.")


class GatedLinear(RobustModule):
    def __init__(self, d_in, d_out, init="default", **base_args):
        super().__init__(**base_args)
        self.linear_gate = Linear(d_in, d_out, init="gating")
        self.linear_out = Linear(d_in, d_out, init=init)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        g = self.linear_gate(x)
        g = self.sigmoid(g)
        o = g * self.linear_out(x)
        return o
