import math
import torch
import torch.nn as nn
import numpy as np

from unicore.utils import batched_gather

class IGSO3(nn.Module):

    def __init__(
        self,
        num_omega_bins: int = 1000,
        l_cutoff: int = 100,
        gaussian_thres: float = 0.2,
        grad_gaussian_thres: float = 0.6,
        eps: float = 1e-12,
    ):
        # W = num_omega_bins, L = l_cutoff
        super().__init__()
        self.num_omega_bins = num_omega_bins
        self.gaussian_thres = gaussian_thres
        self.grad_gaussian_thres = grad_gaussian_thres
        self.eps = eps

        # create uniform bins {0.5h, 1.5h, ..., pi-0.5h}.
        omegas = torch.linspace(0., math.pi, num_omega_bins+1)
        self.bin_width = float(omegas[1])
        omegas = omegas[:-1] + .5 * self.bin_width

        # constants for pdf evaluation
        ls = torch.arange(l_cutoff+1, dtype=torch.float)
        sine_terms = self._make_sine_terms(ls, omegas)

        self.register_buffer('omegas', omegas)  # W
        self.register_buffer('ls', ls)          # L
        self.register_buffer('sine_terms', sine_terms)      # W L

    def _make_sine_terms(self, ls, omegas):
        """
        L, W -> W L
        sine_terms(wi, lj) = (2lj+1) * sin((lj+0.5)*wi) / sin(0.5wi)
        """
        l = ls[None, :]     # 1 L
        w = omegas[:, None] # W 1
        sine_terms = (2*l+1) * torch.sin((l+.5)*w) / torch.sin(.5*w)  # W L
        # lhospital approx over w=0
        sine_terms = torch.where(
            w > self.eps, sine_terms, (2*l+1.)**2
        )
        return sine_terms

    def pdf(
        self,
        sigmas: torch.Tensor,
        normalize: bool = True,
        use_cosine: bool = True,
    ):
        """
        * -> * W
        if use_cosine calculates the igso3 density (Maxwell-Boltzmann).
        if not use_cosine calculates the igso3 without (1-cos) term for grad score eval.
            (i.e. the Haar measure is divided.)
        """
        expon = torch.exp(-.5*self.ls*(self.ls+1.)*(sigmas[..., None]**2))   # * L

        pdf = torch.where(
            sigmas[..., None] > self.gaussian_thres,
            (expon[..., None, :]*self.sine_terms).sum(dim=-1),  # * 1 L, W L -> * W
            torch.exp(-.5*(self.omegas/sigmas[..., None])**2)   # * W
        )   # * W

        if use_cosine:
            pdf = pdf * (1. - self.omegas.cos())    # * W, W -> * W
        if normalize:
            pdf /= (pdf.sum(dim=-1, keepdim=True) * self.bin_width)

        return pdf

    def cdf(self, sigmas: torch.Tensor, normalize: bool = True, use_cosine: bool = True):
        """
        * -> * W
        """
        pdf = self.pdf(sigmas, normalize=False, use_cosine=use_cosine) # * W, to be normalized by cumsum.
        cdf = torch.cumsum(pdf, dim=-1)
        if normalize:
            cdf = cdf / cdf[..., -1, None]  # * W
        return cdf

    def sample(self, sigmas: torch.Tensor, size: int):
        """
        * -> * N=size
        """
        # create u = random (0, 1) to sample from cdf.
        cdf = self.cdf(sigmas, normalize=True)  # * W
        u = cdf.new_tensor(data=np.random.rand(*sigmas.shape, size))        # * N
        sampled_bins = torch.sum(u[..., None] > cdf[..., None, :], dim=-1)  # * N W, * 1 W -> * N, values {0, ..., N-1}
        # add a random linear shift inside bins.
        bin_shift = sigmas.new_tensor(data=np.random.rand(*sigmas.shape, size)) -.5 # *, N
        omega = (sampled_bins.float() + bin_shift) * self.bin_width
        # assert torch.all(omega < torch.pi) and torch.all(omega > 0.)
        return omega

    def grad_log_pdf(self, sigmas: torch.Tensor, omegas: torch.Tensor):
        """
        *, * N -> * N
        return d log pdf(w; sigma^2) / dw
        """
        assert torch.all(omegas >= 0.) and torch.all(omegas <= np.pi), omegas
        omega_index = (omegas[..., None] > self.omegas).sum(dim=-1) # * N, values {0, ..., W}
        omega_shift = omegas - (omega_index.float()+0.5) * self.bin_width # * N, values [-.5, .5]*bw

        pdf = self.pdf(sigmas, normalize=True, use_cosine=False)        # * W
        pdf = torch.cat((pdf[..., 0:1], pdf, pdf[..., -2:-1]), dim=-1)  # * W+2, manual replicate pad
        pdf = pdf[..., None, :].tile(omegas.shape[-1], 1)   # * N W+2

        pdf_left = batched_gather(pdf, omega_index, dim=-1, num_batch_dims=len(omegas.shape))   # * N
        pdf_right = batched_gather(pdf, omega_index+1, dim=-1, num_batch_dims=len(omegas.shape))  # * N

        # dlogf/dw = df / f
        grad_pdf = (pdf_right - pdf_left) / self.bin_width
        pdf_accurate = pdf_left + omega_shift * grad_pdf
        grad_log_pdf = grad_pdf / pdf_accurate

        gaussian_grad_log_pdf = -omegas / sigmas[..., None]**2
        grad_log_pdf = torch.where(
            sigmas[..., None] > self.grad_gaussian_thres, grad_log_pdf, gaussian_grad_log_pdf,
        )

        return grad_log_pdf


def plot_pdf(path, fmt="png", **kwargs):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    igso3 = IGSO3(**kwargs)

    stds = torch.linspace(0.1, torch.pi, 10)
    pdf = igso3.pdf(stds, normalize=True)

    cmap =  mpl.colormaps["rainbow"]
    for s, pdf_i in zip(stds, pdf):
        plt.plot(igso3.omegas.numpy(), pdf_i.numpy(), color=cmap(float(s)/torch.pi))

    plt.title("igso3 pdf colored by sigma")
    plt.tight_layout()
    plt.savefig(path, format=fmt)
