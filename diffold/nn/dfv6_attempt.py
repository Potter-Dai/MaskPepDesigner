import torch
import torch.nn as nn
from typing import Tuple, Optional

import math

from unicore.modules import LayerNorm
# from torch.nn import LayerNorm
from unicore.utils import permute_final_dims

from unifold.modules.common import (
    Linear,
    residual,
    SimpleModuleList,
    Resnet,
    ResnetBlock,
)

from unifold.modules.frame import Frame, Rotation, Quaternion
from unifold.modules.structure_module import (
    StructureModuleTransition,
    BackboneUpdate,
)

from .base import RobustModule

from unicore.modules import LayerNorm, softmax_dropout


class EvolutionaryPointsAttention(nn.Module):
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_hid: int,
        num_evo_pts: int,
        num_heads: int,
        num_qk_points: int,
        num_v_points: int,
        separate_kv: bool = False,
        bias: bool = True,
        eps: float = 1e-8,
    ):
        super(EvolutionaryPointsAttention, self).__init__()

        self.d_hid = d_hid
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.eps = eps

        hc = self.d_hid * self.num_heads
        self.linear_q = Linear(d_single, hc, bias=bias)
        self.separate_kv = separate_kv
        if self.separate_kv:
            self.linear_k = Linear(d_single, hc, bias=bias)
            self.linear_v = Linear(d_single, hc, bias=bias)
        else:
            self.linear_kv = Linear(d_single, 2 * hc, bias=bias)

        hpq = self.num_heads * self.num_qk_points * 3
        self.linear_q_points = Linear(d_single, hpq)
        hpk = self.num_heads * self.num_qk_points * 3
        hpv = self.num_heads * self.num_v_points * 3
        if self.separate_kv:
            self.linear_k_points = Linear(d_single, hpk)
            self.linear_v_points = Linear(d_single, hpv)
        else:
            hpkv = hpk + hpv
            self.linear_kv_points = Linear(d_single, hpkv)

        self.linear_b = Linear(d_pair, self.num_heads)

        self.head_weights = nn.Parameter(torch.zeros((num_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.num_heads * (d_pair + self.d_hid + self.num_v_points * 4)
        self.linear_out = Linear(concat_out_dim, d_single, init="final")

        self.softplus = nn.Softplus()

        heq = self.num_heads * self.num_qk_points
        self.linear_q_evo = Linear(num_evo_pts, heq)
        hek = self.num_heads * self.num_qk_points
        hev = self.num_heads * self.num_v_points
        if self.separate_kv:
            self.linear_k_evo = Linear(num_evo_pts, hek)
            self.linear_v_evo = Linear(d_single, hev)
        else:
            hekv = hek + hev
            self.linear_kv_evo = Linear(d_single, hekv)

        self.linear_out_e = Linear(num_v_points, num_evo_pts)
        self.linear_out_z = Linear(self.num_heads, d_pair)

    def forward(
        self,
        s: torch.Tensor,    # [*, L, ds]
        z: torch.Tensor,    # [*, L, L, dz]
        e: torch.Tensor,    # [*, L, 3, ne]
        f: Frame,           # [*, L]
        square_mask: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor:
        q = self.linear_q(s)

        q = q.view(q.shape[:-1] + (self.num_heads, -1))

        if self.separate_kv:
            k = self.linear_k(s)
            v = self.linear_v(s)
            k = k.view(k.shape[:-1] + (self.num_heads, -1))
            v = v.view(v.shape[:-1] + (self.num_heads, -1))
        else:
            kv = self.linear_kv(s)
            kv = kv.view(kv.shape[:-1] + (self.num_heads, -1))
            k, v = torch.split(kv, self.d_hid, dim=-1)

        def hidden_to_points(x):
            # [*, L, H*3*np] -> [*, L, H, np, 3]
            return x.view(*x.shape[:-1], self.num_heads, -1, 3)

        def evo_hidden_to_points(x):
            # [*, L, 3, H*np] -> [*, L, H, np, 3]
            x = x.view(*x.shape[:-1], self.num_heads, -1)
            x = permute_final_dims(x, [1, 2, 0])
            return x

        q_pts = (
            hidden_to_points(self.linear_q_points(s)) +
            evo_hidden_to_points(self.linear_q_evo(e))
        )
        if self.separate_kv:
            k_pts = (
                hidden_to_points(self.linear_k_points(s)) +
                evo_hidden_to_points(self.linear_k_evo(e))
            )
            v_pts = (
                hidden_to_points(self.linear_v_points(s)) +
                evo_hidden_to_points(self.linear_v_evo(e))
            )
        else:
            kv_pts = (
                hidden_to_points(self.linear_kv_points(s)) +
                evo_hidden_to_points(self.linear_kv_evo(e))
            )
            k_pts, v_pts = torch.split(
                kv_pts, [self.num_qk_points, self.num_v_points], dim=-2
            )

        def local_to_global(pts):
            return f[..., None, None].apply(pts)

        def global_to_local(pts):
            return f[..., None, None].invert_apply(pts)

        bias = self.linear_b(z)

        attn = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),
            permute_final_dims(k, (1, 2, 0)),
        )   # [*, H, Lq, Lk]

        if self.training:
            attn = attn * math.sqrt(1.0 / (3 * self.d_hid))
            attn = attn + (math.sqrt(1.0 / 3) * permute_final_dims(bias, (2, 0, 1)))
            pt_att = local_to_global(q_pts).unsqueeze(-4) - local_to_global(k_pts).unsqueeze(-5)  # [*, L, L, H, np, 3]
            pt_att = pt_att.float() ** 2
        else:
            attn *= math.sqrt(1.0 / (3 * self.d_hid))
            attn += (math.sqrt(1.0 / 3) * permute_final_dims(bias, (2, 0, 1)))
            pt_att = local_to_global(q_pts).unsqueeze(-4) - local_to_global(k_pts).unsqueeze(-5)
            pt_att *= pt_att

        pt_att = pt_att.sum(dim=(-1, -2))     # [*, L, L, H]
        head_weights = self.softplus(self.head_weights) * math.sqrt(
            1.0 / (3 * (self.num_qk_points * 9.0 / 2))
        )
        pt_att *= head_weights * (-0.5)

        pt_att = permute_final_dims(pt_att, (2, 0, 1))  # [*, H, L, L]
        attn += square_mask
        attn_out = attn
        attn = softmax_dropout(attn, 0, self.training, bias=pt_att.type(attn.dtype))
        del pt_att, q_pts, k_pts, bias
        o = torch.matmul(attn, v.transpose(-2, -3)).transpose(-2, -3)
        o = o.contiguous().view(*o.shape[:-2], -1)  # [*, L, H, D]
        del q, k, v
        o_pts_global = torch.sum(
            (
                attn[..., None, :, :, None]     # [*, H, 1, Lq, Lk, 1]
                * permute_final_dims(local_to_global(v_pts), (1, 3, 0, 2))[..., None, :, :]  # [*, H, 3, 1, Lk, N]
            ),
            dim=-2,
        )   # [*, H, 3, Lk, N]

        o_pts_global = permute_final_dims(o_pts, (2, 0, 3, 1))     # [*, Lq, H, N, 3]

        o_pts = global_to_local(o_pts_global)
        if self.training:
            o_pts_norm = torch.sqrt(torch.sum(o_pts.float() ** 2, dim=-1) + self.eps).type(
                o_pts.dtype
            )   # [*, Lq, H, N]
        else:
            o_pts_norm = torch.sqrt(torch.sum(o_pts ** 2, dim=-1) + self.eps).type(
                o_pts.dtype
            )

        o_pts_norm = o_pts_norm.view(*o_pts_norm.shape[:-2], -1)    # [*, Lq, H*N]

        o_pts = o_pts.view(*o_pts.shape[:-3], -1, 3)        # [*, Lq, H*N, 3]

        # ...hij,...ijd->ihd
        o_pair = torch.matmul(attn.transpose(-2, -3), z)

        o_pair = o_pair.view(*o_pair.shape[:-2], -1)

        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pts, dim=-1), o_pts_norm, o_pair), dim=-1)
        )

        e = self.linear_out_e(o_pts.transpose(-1, -2))
        z = self.linear_out_z(attn)

        return s, e, z
