import torch
import torch.nn as nn
from typing import *
from ml_collections import ConfigDict

from unicore.utils import one_hot, masked_mean
from unicore.modules import LayerNorm

from unifold.modules.auxillary_heads import (
    EnhancedHeadBase,
    DistogramHead,
    PredictedLDDTHead,
    PredictedAlignedErrorHead,
    ExperimentallyResolvedHead,
    predicted_lddt,
    predicted_aligned_error,
    predicted_tm_score,
)
from unifold.modules.common import Linear
from unifold.data import residue_constants as rc


class SequenceDenoiseHead(EnhancedHeadBase):
    def __init__(
        self,
        d_single: int,
        num_classes: int,
        **kwargs,
    ) -> None:
        super().__init__(d_in=d_single, d_out=num_classes, disable_enhance_head=False)



class MPDesignerAuxHeads(nn.Module):
    def __init__(
        self,
        head_cfg: ConfigDict,
    ):
        super().__init__()

        self.enable_plddt = head_cfg.predicted_lddt.enabled
        if self.enable_plddt:
            self.plddt = PredictedLDDTHead(**head_cfg["predicted_lddt"])

        self.enable_distogram = head_cfg.distogram.enabled
        if self.enable_distogram:
            self.distogram = DistogramHead(**head_cfg["distogram"])

        self.enable_seq_denoise = head_cfg.sequence_denoise.enabled
        if self.enable_seq_denoise:
            self.seq_denoise = SequenceDenoiseHead(**head_cfg["sequence_denoise"])

        # self.enable_exp_res = head_cfg.experimentally_resolved.enabled
        # if self.enable_exp_res:
        #     self.experimentally_resolved = ExperimentallyResolvedHead(
        #         **head_cfg["experimentally_resolved"]
        #     )

        self.enable_pae = head_cfg.predicted_aligned_error.enabled
        if self.enable_pae:
            self.pae = PredictedAlignedErrorHead(**head_cfg["predicted_aligned_error"])

        self.pae_cfg = head_cfg.predicted_aligned_error

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
    ):
        aux_outs = {}

        if self.enable_plddt:
            plddt_logits = self.plddt(s)
            aux_outs["plddt_logits"] = plddt_logits

        if self.enable_distogram:
            distogram_logits = self.distogram(z)
            aux_outs["distogram_logits"] = distogram_logits

        if self.enable_seq_denoise:
            aalogits = self.seq_denoise(s)
            aux_outs["aalogits"] = aalogits

        # if self.enable_exp_res:
        #     exp_res_logits = self.experimentally_resolved(s)
        #     aux_outs["experimentally_resolved_logits"] = exp_res_logits

        if self.enable_pae:
            pae_logits = self.pae(z)
            aux_outs["pae_logits"] = pae_logits

        return aux_outs

    def get_confidence_scores(self, aux_outs, batch):
        ret = {}
        if self.enable_plddt:
            plddt_logits = aux_outs["plddt_logits"].detach()
            plddt = predicted_lddt(plddt_logits)
            ret["plddt"] = plddt

        if self.enable_pae:
            pae_logits = aux_outs["pae_logits"].detach()
            pae = predicted_aligned_error(pae_logits, **self.pae_cfg, return_values_only=True)
            ptm = predicted_tm_score(
                pae_logits, interface=False, **self.pae_cfg
            )
            ret.update({"pae": pae, "ptm": ptm,})
            iptm_weight = self.pae_cfg.get("iptm_weight", 0.)
            if iptm_weight > 0.:
                iptm = predicted_tm_score(
                    pae_logits,
                    interface=True,
                    asym_id=batch["chain_id"],
                    **self.pae_cfg,
                )
                iptm_p_ptm = (
                    iptm_weight * iptm + (1.0 - iptm_weight) * ptm
                )
                ret.update({
                    "iptm": iptm, "iptm_plus_ptm": iptm_p_ptm,
                })

        return ret


# Compatibility names for existing downstream code.
Dfv6AuxHeads = MPDesignerAuxHeads
