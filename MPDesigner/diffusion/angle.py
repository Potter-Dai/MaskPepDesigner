import math
import torch
import torch.nn as nn
import numpy as np

from unifold.data.data_ops import gumbel_noise
from unicore.data.data_utils import numpy_seed
from unicore.utils import batched_gather

raise NotImplementedError("do not import this.")

class AngleDistribution(nn.Module):

    def __init__(
        self,
        num_omega_bins: int = 1000,
        l_cutoff: int = 50,
        gaussian_threshold: float = 0.1,
        grad_gaussian_threshold: float = 0.7,
        eps: float = 1e-12,
    ):
        # for shape annotations, N = num_omega_bins, L = l_cutoff
        super().__init__()

        self.num_omega_bins = num_omega_bins
        self.gaussian_threshold = gaussian_threshold
        self.grad_gaussian_threshold = grad_gaussian_threshold
        self.eps = eps

        omegas = torch.linspace(0, math.pi, num_omega_bins+1)[:-1]
        ls = torch.arange(l_cutoff+1, dtype=torch.float)
        sino = self._make_sinosoid_components(ls, omegas, eps)
        self.register_buffer('ls', ls)          # [L]
        self.register_buffer('omegas', omegas)  # [N]
        self.register_buffer('sino', sino)      # [N, L]

        self.bin_width = float(self.omegas[1])

    @staticmethod
    def _make_sinosoid_components(ls, omegas, eps):
        """
        [L], [N], float -> [N, L]
        """
        l = ls[None, :]
        o = omegas[:, None]
        return (2*l+1) * (
            torch.where(
                o > eps,
                torch.sin((l+.5)*o) / torch.sin(.5*o).clamp_min(eps),
                2*l+1
            )
        )

    def _pdf(self, sigmas: torch.Tensor, normalize: bool = True):
        """
        [*] -> [*, N]
        """
        expand_shape = [1 for _ in sigmas.shape]
        ls = self.ls.view(*expand_shape, 1, *self.ls.shape)     # [*, 1, L]
        sino = self.sino.view(*expand_shape, *self.sino.shape)  # [*, N, L]
        expon = torch.exp( -ls*(ls+1) * sigmas[..., None, None] ** 2 / 2)    # [*, 1, L]
        pdf = (sino * expon).sum(dim=-1)                        # [*, N]
        if normalize:
            pdf = pdf / pdf.sum(dim=-1, keepdim=True)
        return pdf

    def sample(self, sigmas: torch.Tensor, size: int):
        """
        [*] -> [*, L=size]
        """

        # use gaussian approx. for small stds:
        #   1. to eliminate numerical issues of igso3 density;
        #   2. to eliminate histogram artifacts.
        #   3. mod pi for very very rare events.
        omega_small_z = sigmas.new_tensor(data=np.random.randn(*sigmas.shape, size))
        omega_small = (omega_small_z * sigmas).abs() % math.pi
        if torch.all(sigmas <= self.gaussian_threshold):        # no need for pdf estim.
            return omega_small

        # histogram sampling on normalized igso3 density.
        pdf = self._pdf(sigmas, normalize=True)         # [*, N]
        logits = torch.log(pdf.clamp_min(self.eps)).float()
        logits = logits[..., None, :].tile(size, 1)     # [*, L, N]

        # random factors.
        gumbel = gumbel_noise(logits.shape, logits.device)
        bin_shift = logits.new_tensor(np.random.rand(*logits.shape[:-1]))   # [*, L]

        omega_index = (logits + gumbel).argmax(dim=-1)          # [*, L]
        omega_start = batched_gather(self.omegas, omega_index)  # [*, L]
        omega = omega_start + bin_shift * self.bin_width        # [*, L]

        omega = torch.where(
            sigmas > self.gaussian_threshold,
            omega, omega_small
        )

        return omega

    def grad_log_pdf(self, sigmas: torch.Tensor, omegas: torch.Tensor):
        """
        return dlogf(w, sigma^2) / dw
        [*], [*, L] -> [*, L]
        notably this <= 0.
        """
        assert torch.all(omegas >= 0.) and torch.all(omegas <= np.pi), omegas

        gaussian_grad = -omegas / (sigmas ** 2) # [*, L]
        if torch.all(sigmas <= self.grad_gaussian_threshold):    # no need for pdf est.
            return gaussian_grad

        pdf = self._pdf(sigmas, normalize=True) # [*, N]
        pdf = torch.cat((pdf, pdf[..., -2:-1]), dim=-1).clamp_min(self.eps)     # [*, N+1]
        omega_index = ((omegas[..., None] > self.omegas).sum(dim=-1).clamp(max=self.num_omega_bins, min=1) - 1).long()  # [*, L]
        pdf = pdf[..., None, :].tile(omegas.shape[-1], 1)   # [*, L, N+1]

        pdf_this = batched_gather(pdf, omega_index, dim=-1, num_batch_dims=len(omegas.shape))   # [*, L]
        pdf_next = batched_gather(pdf, (omega_index+1), dim=-1, num_batch_dims=len(omegas.shape))   # [*, L]
        # print(pdf_this, pdf_next)
        # print(omega_index, pdf_this, pdf_next)
        grad_log_pdf = (pdf_next / pdf_this).log() / self.bin_width       # [*, L], < 0
        # print(grad_log_pdf)
        return torch.where(
            sigmas[..., None] > self.gaussian_threshold,
            grad_log_pdf, gaussian_grad,
        )


def plot_pdf(path="./img/omega_pdf.png", fmt="png", **kwargs):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    distrib = AngleDistribution(**kwargs)
    stds = torch.arange(0.04, 3, 0.02)
    pdf = distrib._pdf(stds, normalize=True)
    # print(pdf.shape)
    cmap =  mpl.colormaps["viridis"]
    def gaussian(x, std):
        y = torch.exp(-(x/std)**2/2)
        y = y / y.sum(dim=-1, keepdim=True)
        return y
    for i, s in enumerate(stds):
        pdf_i = pdf[i] if s > 0.2 else gaussian(distrib.omegas, s)
        plt.plot(distrib.omegas.numpy(), pdf_i.numpy(), color=cmap(float(s) / 2))
    plt.title("omega pdf colored by eps")
    plt.tight_layout()
    plt.savefig(path, format=fmt)
