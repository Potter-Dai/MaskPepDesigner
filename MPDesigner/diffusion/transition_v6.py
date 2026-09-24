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
from .cos_angle import IGSO3

from MPDesigner.data.utils import to_numpy

from ml_collections import ConfigDict

from . import scipy_so3


class EuclideanDiffuser(nn.Module):
    def __init__(
        self,
        scale_factor: float = 1.,
        beta_clip: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.scale_factor = scale_factor or 1.
        self.beta_clip = beta_clip
        self.eps = eps

    def beta(self, t, s):
        assert torch.all(t > s)
        return 1. -self.gamma(t) / self.gamma(s)

    def gamma(
        self,
        t: torch.Tensor
    ) -> torch.Tensor:
        '''
        mapping from t to gamma_t. Make sure gamma(0) = 1 and gamma(1) = 0.
        '''
        return 1. - t

    def prior(
        self,
        shape,
        dtype: torch.dtype = torch.float,
        device: torch.device = "cpu",
    ) -> torch.Tensor:
        z = np.random.randn(*shape)
        z = torch.tensor(z, dtype=dtype, device=device)
        return z * self.scale_factor

    def addnoise(
        self,
        x_s: torch.Tensor,          # [*, L, D]
        diff_mask: torch.Tensor,    # [*, L]
        t: torch.Tensor,            # [*]
        s: torch.Tensor = None,     # [*]
    ):
        '''
        forward transition from timestamp `s` to `t` (s < t).
        '''
        if s is None:   # by default s=0 and gamma(s)=1.
            s = torch.zeros_like(t)
        else:
            assert torch.all(s < t)

        z = np.random.randn(*x_s.shape)
        z = torch.tensor(z, dtype=x_s.dtype, device=x_s.device)

        g = self.gamma(t) / self.gamma(s)

        x_t = g.sqrt() * x_s + (1. - g).sqrt() * z * self.scale_factor
        if diff_mask is not None:
            x_t = torch.where(
                diff_mask[..., None] > 0,
                x_t,
                x_s
            )

        return x_t, z

    def denoise(
        self,
        x_t: torch.Tensor,          # [*, L, D]
        xh_0: torch.Tensor,          # [*, L, D]
        diff_mask: torch.Tensor,    # [*, L]
        t: torch.Tensor,            # [*]
        s: torch.Tensor,     # [*]
    ):
        '''
        backward transition from timestamp `t` to `s` (s < t).
        '''
        assert torch.all(s < t)

        z = np.random.randn(*x_t.shape)
        z = torch.tensor(z, dtype=x_t.dtype, device=x_t.device)

        gt, gs = self.gamma(t), self.gamma(s)

        beta = (1. - gt / gs)[..., None].clamp_max(self.beta_clip)

        score = self.score(x_t, xh_0, t)

        # x_s = (2. - (1. - beta).sqrt()) * x_t + beta * score + beta.sqrt() * z * self.scale_factor
        x_s = (1/(1-beta).sqrt())*(x_t + beta * score * self.scale_factor**2) + ((1 - gs)/(1-gt)*beta).sqrt() * z * self.scale_factor
        # x_s = torch.where(
        #     s[..., None, None] > 1e-12, x_s, x_t
        # )

        if diff_mask is not None:
            x_s = torch.where(
                diff_mask[..., None] > 0,
                x_s,
                x_t
            )

        return x_s, beta

    def score(
        self,
        x_t: torch.Tensor,          # [*, L, D]
        x_0: torch.Tensor,          # [*, L, D]
        t: torch.Tensor,            # [*]
    ):
        '''
        stein score function.
        '''
        g = self.gamma(t)[..., None, None]
        score = (g.sqrt() * x_0 - x_t) / (1. - g) / self.scale_factor**2
        return score


def improved_ddpm_kernel(t, s=0.01):
    T = torch.ones_like(t)
    fn = lambda x: torch.cos(x * torch.pi / 2) ** 2
    normed_t = (t + s) / (T + s)
    normed_zero = s / (T + s)
    return fn(normed_t) / fn(normed_zero)


class PositionDiffuser(EuclideanDiffuser):
    def __init__(
        self,
        scale_factor: float,
        exponential_coef: float = 1.,
        cos_s: float = 0.01,
        beta_clip: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(scale_factor, beta_clip, eps)
        self.cos_s = cos_s
        self.exp = exponential_coef

    def gamma(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        '''
        mapping from t to gamma_t. Make sure gamma(0) = 1 and gamma(1) = 0.
        '''
        return improved_ddpm_kernel(t ** self.exp, self.cos_s)


class SequenceDiffuser(EuclideanDiffuser):
    def __init__(
        self,
        num_aatype: int,
        scale_factor: float,
        g_0: float,
        t_eps: float,
        cos_s: float,
        beta_clip: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(scale_factor, beta_clip, eps)
        self.num_aatype = num_aatype
        self.t_eps = t_eps
        self.g_0 = g_0
        self.cos_s = cos_s
        self.k_eps = (1 - self.g_0) / self.t_eps  # \approx infty
        # self.k2 = self.gamma_ub / (1. - self.gamma_eps) # \approx gamma_ub

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        the gamma function decrease from 1 to g_0 linearly between [0, t_eps),
        and use the (properly scaled) improved_ddpm_kernel afterwards.
        """
        return torch.where(
            t < self.t_eps,
            1. - self.k_eps * t,
            self.g_0 * improved_ddpm_kernel((t - self.t_eps) / (1 - self.t_eps), self.cos_s)
        )


def _shape_to_size(shape):
    n = 1
    for i in shape:
        n *= i
    return n

def uniform_random_rotations(
    shape: Iterable,
    dtype: torch.dtype = torch.float,
    device: torch.device = "cpu",
) -> torch.Tensor:
    np_random_rots = scipy_so3.random(_shape_to_size(shape))
    random_rots = torch.tensor(
        np_random_rots, dtype=dtype, device=device
    ).view(*shape, 3, 3)
    return random_rots


class RotationDiffuser(nn.Module):
    def __init__(
        self,
        max_sigma_sq: float = 9.,
        exponential_coef: float = 1.,
        rw_approx_thres: float = 0.6,
        left: bool = True,
        num_omega_bins: int = 1000,
        l_cutoff: int = 100,
        igso3_gaussian_thres: float = 0.2,
        igso3_grad_gaussian_thres: float = 0.6,
        eps: float = 1e-12,
    ):
        # approx_thres:
        #   for sigma > approx_thres we use IGSO3 angle distribution.
        #   for sigma < approx_thres we use Brownian motion in the tangent space.
        super().__init__()
        self.max_sigma_sq = max_sigma_sq
        self.exponential_coef = exponential_coef
        self.rw_approx_thres = rw_approx_thres
        self.left = left
        self.igso3 = IGSO3(
            num_omega_bins=num_omega_bins,
            l_cutoff=l_cutoff,
            gaussian_thres=igso3_gaussian_thres,
            grad_gaussian_thres=igso3_grad_gaussian_thres,
            eps=eps,
        )

    @staticmethod
    def prior(
        shape,
        dtype: torch.dtype = torch.float,
        device: torch.device = "cpu",
    ) -> torch.Tensor:
        return uniform_random_rotations(shape, dtype, device)

    def sigma_sq(self, t: torch.Tensor):
        '''
        compute sigma^2_t.
        must make sigma(0) = 0.
        making sigma(T=1) >= 3 guarantees that R_T=1 is from a uniform so3 prior.
        '''
        return self.max_sigma_sq * (t ** self.exponential_coef)

    def addnoise(
        self,
        r_s: torch.Tensor,          # * L 3 3
        diff_mask: torch.Tensor,    # * L
        t: torch.Tensor,            # *
        s: torch.Tensor = None,     # *
    ):
        if s is None:
            s = torch.zeros_like(t)
        else:
            assert torch.all(s <= t), (s, t)

        betas = self.sigma_sq(t) - self.sigma_sq(s) # *
        delta_r = self.random_rotations(
            sigmas=betas.sqrt(),
            size=r_s.shape[-3],
            igso3_obj=self.igso3,
            rw_approx_thres=self.rw_approx_thres
        )   # * L 3 3

        r_t = delta_r @ r_s if self.left else r_s @ delta_r

        if diff_mask is not None:
            r_t = torch.where(diff_mask[..., None, None] > 0, r_t, r_s)

        return r_t, delta_r

    def score(
        self,
        r_t: torch.Tensor,      # [*, L, 3, 3]
        rh_0: torch.Tensor,     # [*, L, 3, 3]
        sigma_t: torch.Tensor,        # [*]
    ) -> torch.Tensor:          # [*, L, 3]

        if self.left:
            r_z = r_t @ rh_0.transpose(-1, -2)  # * L 3 3
        else:
            r_z = rh_0.transpose(-1, -2) @ r_t

        v_z = self.Log(r_z) # * L 3
        rw_approx = -v_z / sigma_t[..., None, None] ** 2
        if torch.all(sigma_t < self.rw_approx_thres):
            return rw_approx

        w = v_z.norm(dim=-1)  # * L
        dlogf_dw = self.igso3.grad_log_pdf(sigma_t, w)  # * L

        score = torch.where(
            sigma_t[..., None, None] > self.rw_approx_thres,
            v_z * dlogf_dw[..., None] / w[..., None], rw_approx
        )

        return score

    def denoise(
        self,
        r_t: torch.Tensor,      # * L 3 3
        rh_0: torch.Tensor,     # * L 3 3
        diff_mask: torch.Tensor,# * L
        t: torch.Tensor,        # *
        s: torch.Tensor,        # *
    ):
        assert torch.all(s <= t), (s, t)
        sigma_sqs_t = self.sigma_sq(t)

        betas = sigma_sqs_t - self.sigma_sq(s)
        delta_so3 = self.random_so3(
            sigmas=betas.sqrt(),
            size=r_t.shape[-3],
            igso3_obj=self.igso3,
            rw_approx_thres=self.rw_approx_thres
        )   # * L 3
        delta_so3 = torch.where(
            s[..., None, None] > 0, delta_so3, delta_so3.new_zeros(*delta_so3.shape)
        )

        score = self.score(r_t, rh_0, sigma_sqs_t.sqrt())   # * L 3
        update_vec = score * betas[..., None, None]
        noisy_update_vec = update_vec + delta_so3
        update_mat = self.Exp(noisy_update_vec)

        if self.left:
            r_s = update_mat @ r_t
        else:
            r_s = r_t @ update_mat

        if diff_mask is not None:
            r_s = torch.where(
                diff_mask[..., None, None] > 0.,
                r_s, r_t
            )

        return r_s, update_vec

    @staticmethod
    def Log(r: torch.Tensor):
        batch_shape = r.shape[:-2]
        np_r = to_numpy(r.reshape(-1, 3, 3))
        np_v = scipy_so3.Log(np_r)
        v = r.new_tensor(np_v).view(*batch_shape, 3)
        return v

    @staticmethod
    def Exp(v: torch.Tensor):
        batch_shape = v.shape[:-1]
        np_v = to_numpy(v.reshape(-1, 3))
        np_r = scipy_so3.Exp(np_v)
        r = v.new_tensor(np_r).view(*batch_shape, 3, 3)
        return r

    @staticmethod
    def random_so3(
        sigmas: torch.Tensor,   # *
        size: int,      # N=size
        igso3_obj: IGSO3 = None,
        rw_approx_thres: float = 0.6,
    ) -> torch.Tensor:
        if igso3_obj is None:
            print("warning: igso3 obj not provided.")
            igso3_obj = IGSO3()

        z = sigmas.new_tensor(np.random.randn(*sigmas.shape, size, 3))   # * N 3
        rotvec_approx = z * sigmas[..., None, None]
        if torch.all(sigmas <= rw_approx_thres):
            return rotvec_approx

        angles = igso3_obj.sample(sigmas, size)     # * N
        rotvec = torch.where(
            sigmas[..., None, None] > rw_approx_thres,
            z * angles[..., None] / z.norm(dim=-1, keepdim=True),
            rotvec_approx,
        )

        return rotvec

    @staticmethod
    def random_rotations(
        sigmas: torch.Tensor,   # *
        size: int,      # N=size
        igso3_obj: IGSO3 = None,
        rw_approx_thres: float = 0.6,
    ) -> torch.Tensor:
        rotvec = RotationDiffuser.random_so3(sigmas, size, igso3_obj, rw_approx_thres)
        rotmat = RotationDiffuser.Exp(rotvec)
        return rotmat


class Diffuser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_trans = PositionDiffuser(**config.position)
        self.rot_trans = RotationDiffuser(**config.rotation)
        self.seq_trans = SequenceDiffuser(**config.sequence)

    def prior(
        self,
        batch_shape: Tuple[int],
        seq_len: int,
        dtype: torch.dtype = torch.float,
        device: torch.device = "cpu",
    ):
        s_T = self.seq_trans.prior((*batch_shape, seq_len, self.seq_trans.num_aatype), dtype, device)
        r_T = self.rot_trans.prior((*batch_shape, seq_len), dtype, device)
        p_T = self.pos_trans.prior((*batch_shape, seq_len, 3), dtype, device)
        f_T = Frame(Rotation(r_T), p_T).to_tensor_4x4()
        return s_T, f_T

    def addnoise(
        self,
        s_s: torch.Tensor,              # [*, L, 21]
        f_s: torch.Tensor,              # [*, L, 4, 4]
        seq_gen_mask: torch.Tensor,     # [*, L]
        frame_gen_mask: torch.Tensor,   # [*, L]
        t: torch.Tensor,                # [*]
        s: torch.Tensor = None          # [*]
    ):
        s_t, _ = self.seq_trans.addnoise(s_s, seq_gen_mask, t, s)

        r_s, p_s = torch.split(f_s[..., :3, :], (3, 1), dim=-1)  # [L, 3, 3/1]
        p_s = p_s.squeeze(-1)                           # [L, 3]

        r_t, _ = self.rot_trans.addnoise(r_s, frame_gen_mask, t, s)
        p_t, _ = self.pos_trans.addnoise(p_s, frame_gen_mask, t, s)
        f_t = Frame(Rotation(r_t), p_t).to_tensor_4x4()

        return s_t, f_t

    def denoise(
        self,
        s_t: torch.Tensor,              # [*, L, 21]
        f_t: torch.Tensor,              # [*, L, 4, 4]
        sh_0: torch.Tensor,
        fh_0: torch.Tensor,
        seq_gen_mask: torch.Tensor,     # [*, L]
        frame_gen_mask: torch.Tensor,   # [*, L]
        t: torch.Tensor,                # [*]
        s: torch.Tensor = None          # [*]
    ):
        s_s, _ = self.seq_trans.denoise(s_t, sh_0, seq_gen_mask, t, s)

        r_t, p_t = torch.split(f_t[..., :3, :], (3, 1), dim=-1)  # [L, 3, 3/1]
        rh_0, ph_0 = torch.split(fh_0[..., :3, :], (3, 1), dim=-1)  # [L, 3, 3/1]
        p_t = p_t.squeeze(-1)                           # [L, 3]
        ph_0 = ph_0.squeeze(-1)

        r_s, _ = self.rot_trans.denoise(r_t, rh_0, frame_gen_mask, t, s)
        p_s, _ = self.pos_trans.denoise(p_t, ph_0, frame_gen_mask, t, s)
        f_s = Frame(Rotation(r_s), p_s).to_tensor_4x4()

        return s_s, f_s

    def denoise_fwd(
        self,
        s_t: torch.Tensor,              # [*, L, 21]
        f_t: torch.Tensor,              # [*, L, 4, 4]
        sh_0: torch.Tensor,
        fh_0: torch.Tensor,
        seq_gen_mask: torch.Tensor,     # [*, L]
        frame_gen_mask: torch.Tensor,   # [*, L]
        t: torch.Tensor,
        s: torch.Tensor,                # [*]
    ):
        s_s, _ = self.seq_trans.addnoise(sh_0, None, t=s, s=None)

        rh_0, ph_0 = torch.split(fh_0[..., :3, :], (3, 1), dim=-1)  # [L, 3, 3/1]
        ph_0 = ph_0.squeeze(-1)                           # [L, 3]

        r_s, _ = self.rot_trans.addnoise(rh_0, None, t=s, s=None)
        p_s, _ = self.pos_trans.addnoise(ph_0, None, t=s, s=None)
        f_s = Frame(Rotation(r_s), p_s).to_tensor_4x4()

        if seq_gen_mask is not None:
            s_s = torch.where(
                seq_gen_mask[..., None] > 0.,
                s_s, s_t.to(dtype=s_s.dtype),
            )
        if frame_gen_mask is not None:
            f_s = torch.where(
                frame_gen_mask[..., None, None] > 0.,
                f_s, f_t.to(dtype=f_s.dtype)
            )

        return s_s, f_s
