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

from .heads import Dfv6AuxHeads
from .base import RobustModule

from .dfv6_modules import (
    InputEmbedder,
    ZormerEncoderStack,
    ZormerDecoderStack,
    GatedLinear,
)

class Dfv6(RobustModule):
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

        self.scale_factor = model_cfg.scale_factor

        self.input_embedder = InputEmbedder(
            **model_cfg.input_embedder,
            **base_cfg,
        )

        self.time_embedder = GatedLinear(
            **model_cfg.time_embedder,
            **base_cfg,
        )

        self.zormer_encoder = ZormerEncoderStack(
            **model_cfg.zormer,
            **model_cfg.zormer_encoder,
            **base_cfg,
        )

        if model_cfg.enable_decoder:
            self.zormer_decoder = ZormerDecoderStack(
                **model_cfg.zormer,
                **model_cfg.zormer_decoder,
                **base_cfg,
            )
            self.decode_from_input_frames = model_cfg.zormer_decoder.decode_from_input_frames
        else:
            self.zormer_decoder = None

        self.aux_heads = Dfv6AuxHeads(
            model_cfg.heads,
        )

        self.dtype = torch.float

    def forward(self, batch: TensorDict):
        if "dummy" in batch:
            return None
        # process inputs. frames & quats are float.
        f1d, f2d, time_feat, seq_mask, frames, quats, frame_gen_mask = self.unpack_inputs(batch)
        self.check_nan((f1d, f2d, time_feat), key="inputs")
        self.check_nan((frames, quats), key="structures")

        pair_mask = seq_mask[..., :, None] * seq_mask[..., None, :]
        attn_mask = torch.where(
            pair_mask > 0., 0., -self.globals_.inf
        )

        s, z = self.input_embedder(f1d, f2d)
        s = s.to(self.dtype)
        z = z.to(self.dtype)
        _prev = (s, z)

        t_emb = self.time_embedder(time_feat)
        t_emb = t_emb.to(self.dtype)

        frames = Frame.from_tensor_4x4(frames).scale_translation(1. / self.scale_factor)

        s, z = self.zormer_encoder(
            s, z, t_emb, frames, pair_mask, attn_mask,
        )
        self.check_nan((s, z), _prev, key="encoder")

        outs = {
            "single": s,
            "pair": z,
        }

        if self.zormer_decoder is not None:
            if not self.decode_from_input_frames:
                sm_out = self.zormer_decoder(
                    s, z, t_emb, pair_mask, attn_mask, frames=None, quats=None, frame_gen_mask=None,
                )
            else:
                assert self.scale_factor == self.zormer_decoder.scale_factor
                assert quats is not None, "must provide initial quaterions"
                quats = Quaternion.from_tensor_7(quats).scale_translation(1. / self.scale_factor)
                sm_out = self.zormer_decoder(
                    s, z, t_emb, pair_mask, attn_mask, frames=frames, quats=quats, frame_gen_mask=frame_gen_mask,
                )
            outs.update(sm_out)
            self.check_nan(list(sm_out.values()), (s, z), key="decoder")

        # heads
        aux_outs = self.aux_heads(s, z)
        self.check_nan(list(aux_outs.values()), (s, z), key="aux")

        outs.update(aux_outs)
        outs.update(self.aux_heads.get_confidence_scores(aux_outs, batch))

        return outs

    def unpack_inputs(self, batch: TensorDict):
        f1d = batch["feat_1d"]
        f2d = batch["feat_2d"]
        time_feat= batch["time_feat"]
        seq_mask = batch["seq_mask"]
        frames = batch["diffused_frames"]
        quats = batch.get("diffused_quats", None)
        frame_gen_mask = batch["frame_gen_mask"]

        if not self.globals_.use_float_input_embedder:
            f1d = f1d.to(self.dtype)
            f2d = f2d.to(self.dtype)
            time_feat = time_feat.to(self.dtype)

        seq_mask = seq_mask.to(self.dtype)
        frame_gen_mask = frame_gen_mask.to(self.dtype)

        return f1d, f2d, time_feat, seq_mask, frames, quats, frame_gen_mask

    @staticmethod
    def convert_input_device(batch, device):
        return tensor_tree_map(lambda x: x.to(device), batch)

    def post_process(self, outs, batch):
        # outs["final_atom_positions"] = atom14_to_atom37(outs["sm"]["atom14_positions"], batch)
        # outs["final_atom_exists"] = batch["atom37_atom_exists"]
        outs.update(
            self.aux_heads.get_confidence_scores(outs, batch)
        )
        return outs

    def _make_input_float(self):
        self.input_embedder.float()
        self.time_embedder.float()

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
