from unifold.modules.structure_module import *
from unifold.modules.frame import rot_to_quat


class RefineStructureModule(StructureModule):
    def __init__(self, use_chain_pooling, **kwargs):
        super(RefineStructureModule, self).__init__(**kwargs)
        self.use_chain_pooling = use_chain_pooling
        if self.use_chain_pooling:
            self.chain_frame_update = BackboneUpdate(kwargs["d_single"])

    def forward(
        self,
        s,
        z,
        aatype,
        bb_frames: torch.Tensor = None,
        seq_mask=None,
        asym_mask=None,         # for chain-level pooling.
        angles_sin_cos=None,    # for fixed torsion angles.
    ):
        if seq_mask is None:
            seq_mask = s.new_ones(s.shape[:-1])

        # generate square mask
        square_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)
        square_mask = gen_attn_mask(square_mask, -self.inf).unsqueeze(-3)
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        initial_s = s
        s = self.linear_in(s)

        if bb_frames is None:
            quat_encoder = Quaternion.identity(
                s.shape[:-1],
                s.dtype,
                s.device,
                requires_grad=False,
            )
            backb_to_global = Frame(
                Rotation(
                    mat=quat_encoder.get_rot_mats(),
                ),
                quat_encoder.get_trans(),
            )
        else:
            backb_to_global = Frame.from_tensor_4x4(
                bb_frames
            )[-1]   # get last bb_frames among 1 / 8 frames.
            quat_encoder = Quaternion(
                rot_to_quat(backb_to_global.get_rots()._mat),
                backb_to_global.get_trans()
            )        # TODO: this rot2quat is slow.

        outputs = []

        for i in range(self.num_blocks):
            s = residual(s, self.ipa(s, z, backb_to_global, square_mask), self.training)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # update quaternion encoder
            # use backb_to_global to avoid quat-to-rot conversion
            quat_encoder = quat_encoder.compose_update_vec(
                self.bb_update(s), pre_rot_mat=backb_to_global.get_rots()
            )

            # initial_s is always used to update the backbone
            if angles_sin_cos is None:
                unnormalized_angles, angles = self.angle_resnet(s, initial_s)
            else: 
                unnormalized_angles = angles_sin_cos.clone().detach()
                angles = angles_sin_cos.clone().detach()

            # convert quaternion to rotation matrix
            backb_to_global = Frame(
                Rotation(
                    mat=quat_encoder.get_rot_mats(),
                ),
                quat_encoder.get_trans(),
            )

            if self.use_chain_pooling:
                assert asym_mask is not None, "must provide asym mask for chain pooling."
                chain_repr = self.chain_pooling(s)
                chain_frames = self.chain_frame_update(chain_repr)
                backb_to_global = self.apply_chain_frames(backb_to_global, chain_frames, asym_mask)

            if i == self.num_blocks - 1:
                all_frames_to_global = self.torsion_angles_to_frames(
                    backb_to_global.scale_translation(self.trans_scale_factor),
                    angles,
                    aatype,
                )

                pred_positions = self.frames_and_literature_positions_to_atom14_pos(
                    all_frames_to_global,
                    aatype,
                )

            preds = {
                "frames": backb_to_global.scale_translation(
                    self.trans_scale_factor
                ).to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
            }

            outputs.append(preds)
            if i < (self.num_blocks - 1):
                # stop gradient in iteration
                quat_encoder = quat_encoder.stop_rot_gradient()
                backb_to_global = backb_to_global.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)
        outputs["sidechain_frames"] = all_frames_to_global.to_tensor_4x4()
        outputs["positions"] = pred_positions
        outputs["single"] = s

        return outputs

    @staticmethod
    def chain_pooling(self, s, asym_mask):
        raise NotImplementedError()

    @staticmethod
    def apply_chain_frames(bb_frames, chain_frames, asym_mask):
        raise NotImplementedError