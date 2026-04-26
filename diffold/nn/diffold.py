import torch
import torch.nn as nn

from unicore.utils import tensor_tree_map

from unifold.modules.attentions import (
    gen_attn_mask,
    gen_tri_attn_mask,
)
from unifold.modules.frame import Frame, Quaternion
from unifold.modules.featurization import atom14_to_atom37


from diffold.data.utils import TensorDict
from .modules import (
    InputEmbedder,
    ZormerStack,
    SidechainModule,
)

from .heads import DiffoldAuxHeads
from .base import RobustModule

class Diffold(RobustModule):
    def __init__(self, config):
        self.config = config
        model_cfg = config.model
        self.globals_ = globals_ = model_cfg.globals
        base_cfg = {
            "enable_nan_check": (globals_.nan_check > 0),
            "tolerance": globals_.extreme_value_tolerance,
            "strict": (globals_.nan_check == 2),
        }
        super().__init__(**base_cfg)

        self.input_embedder = InputEmbedder(
            **model_cfg.input_embedder,
            **base_cfg,
        )

        self.zormer_stack = ZormerStack(
            **model_cfg.zormer,
            **base_cfg,
        )

        self.sidechain_module = SidechainModule(
            **model_cfg.sidechain_module,
        )

        self.aux_heads = DiffoldAuxHeads(
            model_cfg.heads,
        )

        self.dtype = torch.float

    def forward(self, batch):
        if "dummy" in batch:
            return None
        # process inputs.
        batch = self.convert_input_dtype(batch, self.dtype, self.globals_.use_float_input_embedder)
        masks = self.generate_masks(batch, self.globals_.inf)

        target_feat = batch["target_feat"]
        f_tens = batch["diffused_frames"]
        quat_tens = batch["diffused_quats"]

        self.check_nan([target_feat, f_tens, quat_tens], key="inputs")

        # input embedder.
        s, z = self.input_embedder(
            target_feat,
            batch["relative_positions"],
            batch["diffused_distogram"],
        )

        s = s.type(self.dtype)
        z = z.type(self.dtype)

        # zormer.
        f = Frame.from_tensor_4x4(f_tens).scale_translation(1. / self.globals_.scale_factor)
        quat = Quaternion.from_tensor_7(quat_tens).scale_translation(1. / self.globals_.scale_factor)

        s, z, f, _quat, traj, s_stacked = self.zormer_stack(
            s, z, f, quat, **masks,
            chunk_size=self.globals_.chunk_size,
            block_size=self.globals_.block_size,
        )

        f = f.scale_translation(self.globals_.scale_factor)          # scale back frames.
        traj = traj.scale_translation(self.globals_.scale_factor)    # scale back bb traj.


        outs = {
            "single": s,
            "single_per_layer": s_stacked,
            "pair": z,
            "pred_frames": f.to_tensor_4x4(),      
            "pred_frames_per_layer": traj.to_tensor_4x4(),
        }
        self.check_nan(list(outs.values()), key="outs")

        # heads
        aux_outs = self.aux_heads(s, z, s_stacked)
        pred_aatype = torch.argmax(aux_outs["seq_logits"], dim=-1).detach()
        self.check_nan(list(aux_outs.values()), key="aux")

        # sidechain.
        sidechain_outs = self.sidechain_module(s, f, pred_aatype)   # TODO: sample aatype in inference.
        self.check_nan(list(sidechain_outs.values()), key="sc")

        outs.update({
            "pred_aatype": pred_aatype,
            "sm": sidechain_outs,
            "aux": aux_outs,
        })

        outs = self.post_process(outs, batch)

        return outs

    @staticmethod
    def convert_input_dtype(
        batch: TensorDict,
        dtype: torch.dtype,
        use_float_input_embedder: bool,
    ):
        for key in (
            "seq_mask",
            "diffused_frames",
            "diffused_quats",
        ):
            batch[key] = batch[key].type(dtype)
        dtype = (
            dtype
            if not use_float_input_embedder
            else torch.float
        )
        for key in (
            "target_feat",
        ):
            batch[key] = batch[key].type(dtype)
        return batch
    
    @staticmethod
    def convert_input_device(batch, device):
        return tensor_tree_map(lambda x: x.to(device), batch)

    @staticmethod
    def generate_masks(batch, inf):
        seq_mask = batch["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        seq_attn_mask = gen_attn_mask(seq_mask, neg_inf=-inf)
        pair_attn_mask = gen_attn_mask(pair_mask, neg_inf=-inf)
        tri_start_attn_mask, tri_end_attn_mask = gen_tri_attn_mask(pair_mask, inf)

        return {
            "seq_mask": seq_mask,
            "pair_mask": pair_mask,
            "seq_attn_mask": seq_attn_mask,
            "pair_attn_mask": pair_attn_mask,
            "tri_start_attn_mask": tri_start_attn_mask,
            "tri_end_attn_mask": tri_end_attn_mask,
        }

    def post_process(self, outs, batch):
        # outs["final_atom_positions"] = atom14_to_atom37(outs["sm"]["atom14_positions"], batch)
        # outs["final_atom_exists"] = batch["atom37_atom_exists"]
        outs["aux"].update(
            self.aux_heads.get_confidence_scores(outs["aux"], batch)
        )
        return outs

    def _make_input_float(self):
        self.input_embedder.float()

    def half(self):
        super().half()
        self.dtype = torch.half
        if self.globals_.use_float_input_embedder:
            self._make_input_float()
        return self

    def bfloat16(self):
        super().bfloat16()
        self.dtype = torch.bfloat16
        if self.globals_.use_float_input_embedder:
            self._make_input_float()
        return self

    def float(self):
        super().float()
        self.dtype = torch.float
        return self
