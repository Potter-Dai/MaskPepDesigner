import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

from scipy.spatial.transform import Rotation as R

from unicore.data.data_utils import numpy_seed
from unicore.utils import one_hot

from unifold.data.data_ops import gumbel_max_sample
from unifold.modules.frame import Frame, Rotation
# from .angle import AngleDistribution

from diffold.data.utils import torch_seed

raise NotImplementedError("do not import this.")


def check_gamma_consistency(
    gamma_t: torch.Tensor,
    gamma_s: torch.Tensor,
    threshold: float = 1e-3,
) -> None:
    assert torch.all((gamma_t >= 0.) & (gamma_t <= 1.)), f"gamma_t must be in [0, 1]."
    assert torch.all((gamma_s >= 0.) & (gamma_t <= 1.)), f"gamma_s must be in [0, 1]."

    assert torch.all(gamma_s > threshold), \
        f"gamma_s should be larger than {threshold}. Got min={torch.min(gamma_s)}."
    assert torch.all(gamma_s > gamma_t), \
        f"denoised results should have larger gamma than input."


def class_id_to_normalized_logits(
    aatype: torch.Tensor,
    num_classes: int,
):
    aatype_pos = one_hot(aatype, num_classes)
    pos_coef = math.sqrt(num_classes - 1)
    neg_coef = -1. / pos_coef
    return aatype_pos * pos_coef + (1. - aatype_pos) * neg_coef


def make_gaussian_posterior_coefficients(
    gamma_t: torch.Tensor,
    gamma_s: torch.Tensor,
    threshold: float = 1e-3
):
    """
    generate the coefficients of x_0_hat, x_t and eps in the backward process.
    we require gamma_s > gamma_t and gamma_s not too small (> threshold).
    """
    check_gamma_consistency(gamma_t, gamma_s, threshold)

    denom = (1. - gamma_t).clamp_min(threshold)
    gamma_t_to_s = gamma_t / gamma_s

    c_0 = (
        gamma_s.clamp_min(0.).sqrt() * \
        (1. - gamma_t_to_s)
    ) / denom

    c_1 = (
        gamma_t_to_s.clamp_min(0.).sqrt() * \
        (1. - gamma_s)
    ) / denom

    c_2 = (
        (1. - gamma_s) * (1. - gamma_t_to_s)
    ) / denom

    # for regions with approx. 1 gamma_t, i.e. with no noises, we use x_t always.
    # clamping is only for safety reason, e.g. rounding.
    use_binary = gamma_t > (1. - threshold)
    c_0 = torch.where(use_binary, 0., c_0).clamp(min=0., max=1.)
    c_1 = torch.where(use_binary, 1., c_1).clamp(min=0., max=1.)
    c_2 = torch.where(use_binary, 0., c_2).clamp(min=0., max=1.)

    return c_0, c_1, c_2


class PositionTransition(nn.Module):
    def __init__(
        self,
        threshold: float = 0.001,
        scale_factor: Union[float, str] = "auto",
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.threshold = threshold
    
    @staticmethod
    def _auto_scale_factor(seq_len: int):
        k, b = 5.65525102, -6.1982153705882315
        r = k * (seq_len ** (1. / 3)) + b     # estimated volume radius
        return max(r, 4)

    def get_scale_factor(self, seq_len: int = None):
        if self.scale_factor == "auto":
            assert seq_len is not None
            return self._auto_scale_factor(seq_len)
        else:
            return self.scale_factor
    
    def prior(
        self,
        shape: Iterable,
        seq_len: int = None,
        dtype: torch.dtype = torch.float,
        device: torch.device = "cpu",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        assert shape[-1] == 3, "shape {shape} incompatible as 3D coordinates."
        
        seq_len = seq_len if seq_len else shape[-2]
        
        with numpy_seed(seed, 0, key="position_prior"):
            err = torch.tensor(np.random.randn(*shape), dtype=dtype, device=device)
        
        scale_factor = self.get_scale_factor(seq_len)
        return err * scale_factor
    
    def add_noise(
        self,
        xyz_0: torch.Tensor,        # [*, L, 3]
        gamma_t: torch.Tensor,      # [*, L]
        seed: Optional[int] = None,
    ):
        with numpy_seed(seed, 0, key="position_add_noise"):
            err = torch.tensor(np.random.randn(*xyz_0.shape), dtype=xyz_0.dtype, device=xyz_0.device)

        scale_factor = self.get_scale_factor(xyz_0.shape[-2])

        x_t = gamma_t.sqrt()[..., None] * xyz_0 + (1. - gamma_t).sqrt()[..., None] * err * scale_factor

        return x_t, err

    def denoise(
        self,
        xyz_t: torch.Tensor,
        xyz_0_hat: torch.Tensor,    # [*, L, 3]
        gamma_t: torch.Tensor,      # [*, L]
        gamma_s: torch.Tensor,      # [*, L]
        seed: Optional[int] = None,
    ):
        with numpy_seed(seed, 0, key="position_denoise"):
            err = torch.tensor(np.random.randn(*xyz_t.shape), dtype=xyz_t.dtype, device=xyz_t.device)

        if self.scale_factor == "auto":
            scale_factor = self._auto_scale_factor(xyz_t.shape[-2])
        else:
            scale_factor = self.scale_factor

        assert torch.all(gamma_t <= gamma_s)

        c_0, c_1, c_2 = make_gaussian_posterior_coefficients(gamma_t, gamma_s, self.threshold)

        x_s = c_0[..., None] * xyz_0_hat + c_1[..., None] * xyz_t + c_2[..., None] * err * scale_factor

        return x_s, err


class RotationTransition(nn.Module):
    def __init__(
        self, 
        threshold: float = 0.001,
        **angle_distribution_kwargs
    ):
        super().__init__()
        self.threshold = threshold
        self.angle_distrib = AngleDistribution(**angle_distribution_kwargs)

    def prior(
        self,
        shape,
        dtype: torch.dtype = torch.float,
        device: torch.device = "cpu",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        random_rots = self.uniform_random_rotations(
            shape, dtype, device, seed
        )
        return random_rots
    
    def add_noise(
        self,
        rotmats_0: torch.Tensor,    # [*, 3, 3]
        gamma_t: torch.Tensor,      # [*]
        seed: Optional[int] = None,
    ):
        random_rotations = self.igso3_random_rotations(
            (1. - gamma_t).clamp_min(0.).sqrt(),
            self.angle_distrib,
            seed,
            dtype=rotmats_0.dtype,
            device=rotmats_0.device
        )

        scaled_rotations = self.scale_rotation(
            rotmats_0, gamma_t.clamp_min(0.).sqrt()
        )

        return random_rotations @ scaled_rotations, random_rotations

    def denoise(
        self,
        rotmats_t: torch.Tensor,    # [*, L, 3, 3]
        rotmats_0_hat: torch.Tensor,
        gamma_t: torch.Tensor,      # [*, L]
        gamma_s: torch.Tensor,
        seed: Optional[int] = None,
    ):
        c_0, c_1, c_2 = make_gaussian_posterior_coefficients(gamma_t, gamma_s, self.threshold)
        random_rotations = self.igso3_random_rotations(
            c_2, self.angle_distrib, seed, dtype=rotmats_t.dtype, device=rotmats_t.device
        )
        R_delta = rotmats_0_hat @ rotmats_t.transpose(-1, -2)
        scaled_R_delta = self.scale_rotation(
            R_delta, c_0 / (c_0 + c_1)
        )
        scaled_rotations = scaled_R_delta @ rotmats_t

        return random_rotations @ scaled_rotations, random_rotations

    @staticmethod
    def scale_rotation(
        rotmats: torch.Tensor,       # [*, 3, 3]
        coefs: torch.Tensor,         # [*]
    ) -> torch.Tensor:
        batch_shape = rotmats.shape[:-2]
        dtype, device = rotmats.dtype, rotmats.device
        assert coefs.shape == batch_shape
        rotmats = rotmats.reshape(-1, 3, 3).float().numpy()
        coefs = coefs.reshape(-1, 1).float().numpy()
        rotvecs = R.as_rotvec(R.from_matrix(rotmats))
        scaled_rotvecs = coefs * rotvecs
        scaled_rotmats = R.as_matrix(R.from_rotvec(scaled_rotvecs))
        scaled_rotmats = torch.tensor(
            scaled_rotmats, dtype=dtype, device=device
        ).reshape(*batch_shape, 3, 3)
        return scaled_rotmats
    
    @staticmethod
    def uniform_random_rotations(
        shape: Iterable,
        dtype: torch.dtype = torch.float,
        device: torch.device = "cpu",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        assert shape[-1] == shape[-2] == 3, "shape {shape} incompatible as rotation matrices."
        batch_shape = shape[:-2]
        with numpy_seed(seed, 0, key="uniform_random_rotations"):
            unnormed_axes = np.random.randn(*batch_shape, 3)
        unnormed_axes = np.reshape(unnormed_axes, (-1, 3))
        rotmats = R.as_matrix(R.from_rotvec(unnormed_axes))
        rotmats = torch.tensor(rotmats, dtype=dtype, device=device).reshape(*batch_shape, 3, 3)
        return rotmats

    @staticmethod
    def igso3_random_rotations(
        stddevs: torch.Tensor,      # [*]
        angle_distrib: AngleDistribution,
        seed: Optional[int] = None,
        **tensor_args,
    ) -> torch.Tensor:
        batch_shape = stddevs.shape
        stddevs = stddevs.reshape(-1)
        with numpy_seed(seed, 0, key="igso3_random_rotations"):
            angles = angle_distrib.sample(stddevs).float().numpy()  
            unnormed_axes = np.random.randn(*angles.shape, 3)
        normed_axes = unnormed_axes / np.linalg.norm(unnormed_axes, ord=2, axis=-1, keepdims=True)
        rotvecs = angles[..., None] * normed_axes
        rotmats = R.as_matrix(R.from_rotvec(rotvecs))
        rotmats = torch.tensor(rotmats, **tensor_args).reshape(*batch_shape, 3, 3)
        return rotmats

uniform_random_rotations = RotationTransition.uniform_random_rotations

# depreciated
class SequenceGaussianTransition(nn.Module):
    def __init__(
        self, 
        denoise_threshold: float = 0.001,
        num_classes=21
    ):
        super().__init__()
        self.threshold = denoise_threshold
        self.num_classes = num_classes

    @staticmethod
    def sample(logits):
        return gumbel_max_sample(logits)

    def prior(
        self,
        shape: Iterable,
        dtype: torch.dtype = torch.float,
        device: torch.device = "cpu",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        assert shape[-1] == self.num_classes, "shape {shape} incompatible as sequence logits."
        
        with numpy_seed(seed, 0, key="sequence_prior"):
            err = torch.tensor(np.random.randn(*shape), dtype=dtype, device=device)
        
        return err

    def add_noise(
        self,
        logits_0: torch.Tensor,        # [*, L, 3]
        gamma_t: torch.Tensor,      # [*, L]
        seed: Optional[int] = None,
    ):
        with numpy_seed(seed, 0, key="sequence_add_noise"):
            err = torch.tensor(np.random.randn(*logits_0.shape), dtype=logits_0.dtype, device=logits_0.device)

        logits_t = gamma_t.sqrt()[..., None] * logits_0 + (1. - gamma_t).sqrt()[..., None] * err

        return logits_t, err

    def denoise(
        self,
        logits_t: torch.Tensor,
        logits_0_hat: torch.Tensor,    # [*, L, 3]
        gamma_t: torch.Tensor,      # [*, L]
        gamma_s: torch.Tensor,      # [*, L]
        seed: Optional[int] = None,
    ):
        with numpy_seed(seed, 0, key="sequence_denoise"):
            err = torch.tensor(np.random.randn(*logits_t.shape), dtype=logits_t.dtype, device=logits_t.device)

        assert torch.all(gamma_t <= gamma_s)

        c_0, c_1, c_2 = make_gaussian_posterior_coefficients(gamma_t, gamma_s, self.threshold)

        logits_s = c_0[..., None] * logits_0_hat + c_1[..., None] * logits_t + c_2[..., None] * err

        return logits_s, err
    

class SequenceCategoricalTransition(nn.Module):
    def __init__(
        self,
        denoise_threshold: float = 0.001,
        num_classes=21,
        prior_type: str = "masking",
        mask_index: int = -1,
        customized_prior: torch.Tensor = None,
    ):
        super().__init__()
        self.threshold = denoise_threshold
        self.num_classes = num_classes
        self.prior_type = prior_type
        if prior_type == "masking":
            self.mask_index = mask_index
            self.prior_distrib = self.build_prior(
                num_classes,
                type_="masking",
                mask_index=mask_index
            )
        elif prior_type == "uniform":
            self.prior_distrib = self.build_prior(
                num_classes,
                type_="uniform",
            )
        elif prior_type == "customized":
            self.prior_distrib = self.build_prior(
                num_classes,
                type_="customized",
                customized_prior=customized_prior
            )


    @staticmethod
    def build_prior(num_classes: int, type_: str, **kwargs):
        if type_ == "masking":
            prior = torch.zeros([num_classes])
            mask_index = kwargs["mask_index"] % num_classes
            prior[mask_index] = 1.
        elif type_ == "uniform":
            prior = torch.ones([num_classes]) / num_classes
        elif type_ == "customized":
            prior = torch.as_tensor(kwargs["customized_prior"])
            assert prior.shape == (num_classes,)
        else:
            raise NotImplementedError(f"{type_} prior not implemented.")
        return prior
    
    @staticmethod
    def _sample(probs: torch.Tensor):
        batch_shape = probs.shape[:-1]
        num_classes = probs.shape[-1]
        probs = probs.reshape(-1, num_classes)
        out = torch.multinomial(probs, num_samples=1)
        out = out.reshape(batch_shape)
        return out

    def posterior(
        self,
        aatype_t: torch.Tensor, # [*, L]
        probs_s: torch.Tensor,  # [*, L, Nc]
        gamma_t: torch.Tensor,  # [*, L]
        gamma_s: torch.Tensor,  # [*, L]
    ):
        prior = self.prior_distrib.to(dtype=probs_s.dtype, device=probs_s.device)
        check_gamma_consistency(gamma_t, gamma_s, threshold=self.threshold)
        gamma_t_to_s = gamma_t / gamma_s    # < 1
        posterior = (
            gamma_t_to_s[..., None] * one_hot(aatype_t, self.num_classes, dtype=probs_s.dtype) +
            (1. - gamma_t_to_s[..., None]) * prior
        ) * (
            gamma_s[..., None] * probs_s +
            (1. - gamma_s[..., None]) * prior
        )
        posterior = posterior / posterior.sum(dim=-1, keepdim=True)
        return posterior

    def add_noise(
        self,
        aatype_0: torch.Tensor,        # [*, L]
        gamma_t: torch.Tensor,      # [*, L]
        seed: Optional[int] = None,
    ):
        prior = self.prior_distrib.to(dtype=gamma_t.dtype, device=gamma_t.device)
        probs_t = (
            gamma_t[..., None] * one_hot(aatype_0, self.num_classes) +
            (1. - gamma_t[..., None]) * prior
        )
        with torch_seed(seed, 0, key="sequence_add_noise"):
            aatype_t = self._sample(probs_t)
        return aatype_t, probs_t

    def denoise(
        self,
        aatype_t: torch.Tensor,
        probs_s_hat: torch.Tensor,    # [*, L, 3]
        gamma_t: torch.Tensor,      # [*, L]
        gamma_s: torch.Tensor,      # [*, L]
        seed: Optional[int] = None,
    ):
        posterior = self.posterior(aatype_t, probs_s_hat, gamma_t, gamma_s)
        with torch_seed(seed, 0, key="sequence_denoise"):
            aatype_s = self._sample(posterior)
        return aatype_s, posterior


class DiffusionTransition:
    def __init__(self, config):
        self.config = config
        self.pos_trans = PositionTransition(config.denoise_threshold, config.scale_factor)
        self.rot_trans = RotationTransition(config.denoise_threshold, **config.angle_distribution)
        self.seq_trans = SequenceCategoricalTransition(config.denoise_threshold, **config.sequence)

    def add_noise(self, protein, seed):
        seq_gammas = protein["seq_gammas"]              # [L,]
        frame_gammas = protein["frame_gammas"]          # [L,]

        s_0 = protein["aatype"]                # [L]
        s_t, _ = self.seq_trans.add_noise(s_0, seq_gammas, seed)

        f_0 = protein["frames"]                         # [L, 4, 4]
        r_0, p_0 = torch.split(f_0[..., :3, :], (3, 1), dim=-1)  # [L, 3, 3/1]
        p_0 = p_0.squeeze(-1)                           # [L, 3]
        r_t, _ = self.rot_trans.add_noise(r_0, frame_gammas, seed)
        p_t, _ = self.pos_trans.add_noise(p_0, frame_gammas, seed)
        f_t = Frame(Rotation(r_t), p_t).to_tensor_4x4()

        return {
            "diffused_aatype": s_t,
            "diffused_frames": f_t,
        }
    
    def denoise(self, protein, out):
        seed = int(protein["diffusion_seed"])

        seq_gammas = protein["seq_gammas"]              # [L,]
        frame_gammas = protein["frame_gammas"]          # [L,]

        seq_gammas_to = protein["seq_gammas_to"]
        frame_gammas_to = protein["frame_gammas_to"]

        s_t = protein["diffused_aatype"]
        s_0_hat = out["predicted_seq_logits"]
        probs_hat = torch.softmax(s_0_hat, dim=-1)
        s_s, _ = self.seq_trans.denoise(s_t, probs_hat, seq_gammas, seq_gammas_to, seed)

        f_t = protein["diffused_frames"]                # [L, 4, 4]
        f_0_hat = out["predicted_frames"]               # [L, 4, 4]
        r_t, p_t = torch.split(                         # [L, 3/1, 3]
            f_t[..., :3, :], (3, 1), dim=-1
        )
        r_0_hat, p_0_hat = torch.split(                 # [L, 3/1, 3]
            f_0_hat[..., :3, :], (3, 1), dim=-1
        )
        p_t = p_t.squeeze(-1)
        p_0_hat = p_0_hat.squeeze(-1)
        r_s, _ = self.rot_trans.denoise(r_t, r_0_hat, frame_gammas, frame_gammas_to, seed)
        p_s, _ = self.pos_trans.denoise(p_t, p_0_hat, frame_gammas, frame_gammas_to, seed)
        f_s = Frame(Rotation(r_s), p_s).to_tensor_4x4()

        return {
            "next_frames": f_s,
            "next_seq_logits": s_s,
        }
