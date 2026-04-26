from unifold.modules.alphafold import *
from typing import *
from .sm import RefineStructureModule

class RefineFold(AlphaFold):
    def __init__(self, config):
        super(RefineFold, self).__init__(config)
        self.structure_module = RefineStructureModule(
            use_chain_pooling=False,
            **config.model["structure_module"],
        )
    
    def forward(self, batch):

        m_1_prev = batch.get("m_1_prev", None)
        z_prev = batch.get("z_prev", None)
        x_prev = batch.get("input_atom_positions", None)  # get init x_prev from batch.
        frames_prev = batch.get("input_rigidgroups_gt_frames", None)
        
        if x_prev is not None:
            x_prev = x_prev[0]
        if frames_prev is not None:
            # rigidgroups_gt_frames are sc frames. extract its first as bb frames.
            frames_prev = frames_prev[0, ..., 0, :, :]

        is_grad_enabled = torch.is_grad_enabled()

        num_iters = int(batch["num_recycling_iters"]) + 1
        num_ensembles = int(batch["msa_mask"].shape[0]) // num_iters
        if self.training:
            assert num_ensembles == 1, "don't use ensemble during training"

        # convert dtypes in batch
        batch = self.__convert_input_dtype__(batch)
        for cycle_no in range(num_iters):
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                (
                    outputs,
                    m_1_prev,
                    z_prev,
                    x_prev,
                    frames_prev,
                ) = self.iteration_evoformer_structure_module(
                    batch,
                    m_1_prev,
                    z_prev,
                    x_prev,
                    frames_prev,
                    cycle_no=cycle_no,
                    num_recycling=num_iters,
                    num_ensembles=num_ensembles,
                )
            if not is_final_iter:
                del outputs

        if "asym_id" in batch:
            # used for aux head calculation.
            outputs["asym_id"] = batch["asym_id"][0, ...]
        
        outputs.update(self.aux_heads(outputs))
        
        return outputs

    
    def iteration_evoformer_structure_module(
        self, batch, m_1_prev, z_prev, x_prev, frames_prev, cycle_no, num_recycling, num_ensembles=1
    ):
        # most of this functions is unaltered.
        z, s = 0, 0
        n_seq = batch["msa_feat"].shape[-3]
        assert num_ensembles >= 1
        for ensemble_no in range(num_ensembles):
            idx = cycle_no * num_ensembles + ensemble_no
            fetch_cur_batch = lambda t: t[min(t.shape[0] - 1, idx), ...]
            feats = tensor_tree_map(fetch_cur_batch, batch)
            m, z0, s0, msa_mask, m_1_prev_emb, z_prev_emb = self.iteration_evoformer(
                feats, m_1_prev, z_prev, x_prev
            )
            z += z0
            s += s0
            del z0, s0
        if num_ensembles > 1:
            z /= float(num_ensembles)
            s /= float(num_ensembles)

        outputs = {}

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        # norm loss
        if (not getattr(self, "inference", False)) and num_recycling == (cycle_no + 1):
            delta_msa = m
            delta_msa[..., 0, :, :] = delta_msa[..., 0, :, :] - m_1_prev_emb.detach()
            delta_pair = z - z_prev_emb.detach()
            outputs["delta_msa"] = delta_msa
            outputs["delta_pair"] = delta_pair
            outputs["msa_norm_mask"] = msa_mask

        outputs["sm"] = self.structure_module(
            s,
            z,
            feats["aatype"],
            bb_frames=frames_prev,
            seq_mask=feats["seq_mask"],
            asym_mask=None,         # make this work for chain pooling.
            angles_sin_cos=None,    # make this work for fixed sidechains.
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["pred_frame_tensor"] = outputs["sm"]["frames"][-1]

        # use float32 for numerical stability
        if (not getattr(self, "inference", False)):
            m_1_prev = m[..., 0, :, :].float()
            z_prev = z.float()
            x_prev = outputs["final_atom_positions"].float()
        else:
            m_1_prev = m[..., 0, :, :]
            z_prev = z
            x_prev = outputs["final_atom_positions"]
        
        # explicitly extract frames_prev as retval.
        frames_prev = outputs["sm"]["sidechain_frames"][..., 0, :, :]

        return outputs, m_1_prev, z_prev, x_prev, frames_prev

