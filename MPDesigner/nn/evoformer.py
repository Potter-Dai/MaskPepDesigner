import torch
import torch.nn as nn
from typing import Tuple, Optional
from functools import partial

from unifold.modules.common import (
    Transition,
    OuterProductMean,
    SimpleModuleList,
    residual,
    bias_dropout_residual,
    tri_mul_residual,
)
from unifold.modules.attentions import (
    MSAAttention,
    TriangleAttentionStarting,
    TriangleAttentionEnding,
)
from unifold.modules.triangle_multiplication import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from unicore.utils import checkpoint_sequential

from .base import RobustModule


class SequenceAttentionWithPairBias(MSAAttention):
    def __init__(
        self,
        d_in: int,
        d_hid: int,
        num_heads: int,
        d_pair: int,
    ):
        super().__init__(d_in, d_hid, num_heads, True, d_pair)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        seq_attn_mask: Optional[torch.Tensor] = None,   # [*, L]
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        m = s.unsqueeze(dim=-3)     # [*, 1, L, c]
        mask = seq_attn_mask
        for _ in range(3):
            mask = mask.unsqueeze(dim=-2)  # [*, 1, 1, 1, L]
        m = super().forward(m, z, mask, chunk_size)
        s_res = m.squeeze(dim=-3)       # [*, L, c]

        return s_res


class SequenceAttentionBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hid: int,
        num_heads: int,
        d_pair: int,
        transition_n: int,
        dropout: float,
    ):
        super().__init__()
        self.attn = SequenceAttentionWithPairBias(
            d_in, d_hid, num_heads, d_pair
        )
        self.transition = Transition(
            d_in=d_in,
            n=transition_n,
        )
        self.dropout = dropout

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        seq_attn_mask: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:

        s = bias_dropout_residual(
            self.attn,
            s,
            self.attn(s, z, seq_attn_mask, chunk_size),
            dropout_shared_dim=None,
            prob=self.dropout,
            training=self.training,
        )

        s = residual(
            self.transition(s), s,
            self.training
        )

        return s


class OuterProduct(OuterProductMean):
    def __init__(self, d_single, d_pair, d_hid, eps=0.001):
        super().__init__(
            d_msa=d_single,
            d_pair=d_pair,
            d_hid=d_hid,
            eps=eps,
        )

    def forward(self, s: torch.Tensor, mask: Optional[torch.Tensor] = None, chunk_size: Optional[int] = None) -> torch.Tensor:
        m = s.unsqueeze(-3)         # [*, 1, L, c]
        mask = mask.unsqueeze(-2)   # [*, 1, L]
        return super().forward(m, mask, chunk_size)


class MPDesignerEvoformerBlock(RobustModule):
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        d_hid_seq_attn: int,
        d_hid_opm: int,
        d_hid_mul: int,
        d_hid_pair_att: int,
        num_seq_attn_blocks: int,
        num_heads_seq: int,
        num_heads_pair: int,
        transition_n: int,
        seq_dropout: float,
        pair_dropout: float,
        use_triangle_attn: bool,
        use_triangle_update: bool,
        **base_args
    ):
        super().__init__(**base_args)

        self.seq_att_blocks = SimpleModuleList([
            SequenceAttentionBlock(
                d_in=d_single,
                d_hid=d_hid_seq_attn,
                num_heads=num_heads_seq,
                d_pair=d_pair,
                transition_n=transition_n,
                dropout=seq_dropout,
            ) for _ in range(num_seq_attn_blocks)
        ])

        self.outer_product = OuterProduct(
            d_single,
            d_pair,
            d_hid_opm,
        )

        self.use_triangle_update = use_triangle_update
        if self.use_triangle_update:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                d_pair,
                d_hid_mul,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                d_pair,
                d_hid_mul,
            )

        self.use_triangle_attn = use_triangle_attn
        if self.use_triangle_attn:
            self.tri_att_start = TriangleAttentionStarting(
                d_pair,
                d_hid_pair_att,
                num_heads_pair,
            )
            self.tri_att_end = TriangleAttentionEnding(
                d_pair,
                d_hid_pair_att,
                num_heads_pair,
            )

        self.pair_transition = Transition(
            d_in=d_pair,
            n=transition_n,
        )

        self.row_dropout_share_dim = -3
        self.col_dropout_share_dim = -2

        self.pair_dropout = pair_dropout

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        seq_mask: torch.Tensor,
        seq_attn_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        tri_start_attn_mask: torch.Tensor,
        tri_end_attn_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s_prev = s
        seq_blocks = [
            partial(
                b,
                z=z,
                seq_attn_mask=seq_attn_mask,
                chunk_size=chunk_size,
            )
            for b in self.seq_att_blocks
        ]
        for b in seq_blocks:
            s = b(s)
        self.check_nan(s, s_prev, "seq_attn")

        z_prev = z
        z = residual(
            z, self.outer_product(s, mask=seq_mask, chunk_size=chunk_size),
            self.training
        )
        self.check_nan(z, z_prev, "out_prod")

        if self.use_triangle_update:
            z_prev = z
            z = tri_mul_residual(
                self.tri_mul_out,
                z,
                self.tri_mul_out(z, mask=pair_mask, block_size=block_size),
                self.row_dropout_share_dim,
                self.pair_dropout,
                self.training,
                block_size=block_size,
            )
            z = tri_mul_residual(
                self.tri_mul_in,
                z,
                self.tri_mul_in(z, mask=pair_mask, block_size=block_size),
                self.row_dropout_share_dim,
                self.pair_dropout,
                self.training,
                block_size=block_size,
            )
            self.check_nan(z, z_prev, "tri_mul")

        if self.use_triangle_attn:
            z_prev = z
            z = bias_dropout_residual(
                self.tri_att_start,
                z,
                self.tri_att_start(z, attn_mask=tri_start_attn_mask, chunk_size=chunk_size),
                self.row_dropout_share_dim,
                self.pair_dropout,
                self.training,
            )
            z = bias_dropout_residual(
                self.tri_att_end,
                z,
                self.tri_att_end(z, attn_mask=tri_end_attn_mask, chunk_size=chunk_size),
                self.col_dropout_share_dim,
                self.pair_dropout,
                self.training,
            )
            self.check_nan(z, z_prev, "tri_att")

        z = residual(
            z, self.pair_transition(z, chunk_size=chunk_size),
            self.training
        )
        return s, z


class MPDesignerEvoformerStack(nn.Module):
    def __init__(
        self,
        num_blocks,
        use_gradient_ckpt: bool = True,
        **block_kwargs,
    ):
        super(MPDesignerEvoformerStack, self).__init__()
        self.use_gradient_ckpt = use_gradient_ckpt
        self.blocks = SimpleModuleList([
            MPDesignerEvoformerBlock(**block_kwargs)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        **block_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        blocks = [
            partial(b, **block_kwargs)
            for b in self.blocks
        ]
        if self.use_gradient_ckpt:
            s, z = checkpoint_sequential(
                blocks,
                input=(s, z),
            )
        else:
            for b in blocks:
                s, z = b(s, z)
        return s, z


# Evoformer modules for MPDesigner
