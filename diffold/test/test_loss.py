import torch

from diffold.data.utils import recur_print
from diffold.losses.auxillary import (
    plddt_loss,
    supervised_chi_loss,
    distogram_loss,
    pae_loss,
    sequence_denoise_loss_v3,
)
from diffold.losses.structure import (
    compute_frame_loss,
    compute_ca_metrics,
    backbone_clash_loss,
)

from unifold.modules.frame import Frame, Rotation
from diffold.diffusion.transition import uniform_random_rotations

from diffold.data.data_ops import compute_distogram

from unifold.losses.auxillary import (
    supervised_chi_loss as uf_chi_loss,
    plddt_loss as uf_plddt_loss,
    distogram_loss as uf_distogram_loss,
    pae_loss as uf_pae_loss,
    softmax_cross_entropy,
    one_hot,
    masked_mean
)

from diffold.config import model_config

config = model_config("tiny")

bsz, nres = 2, 12

def equal(a, b, eps=1e-6):
    return (a.shape == b.shape) and (torch.all((a-b).abs() < eps))

def test_distogram_loss():
    print("test distogram loss.")
    num_bins = config.loss.distogram.num_bins
    min_bin = 1
    max_bin = 10
    ca = torch.randn((bsz, nres, 3))
    true_bins = compute_distogram(ca, min_bin, max_bin, num_bins)
    loss_dict1 = {}
    loss_dict2 = {}
    logits = torch.randn((bsz, nres, nres, num_bins))
    frame_mask = torch.randint(0, 2, size=(bsz, nres)).float()
    loss1 = distogram_loss(
        logits, true_bins, frame_mask,
        num_bins,
        eps=1e-10,
        smooth_factor=0.,
        loss_dict=loss_dict1
    )
    loss2 = uf_distogram_loss(
        logits, 
        ca,
        frame_mask,
        min_bin,
        max_bin,
        num_bins,
        eps=1e-10,
        loss_dict=loss_dict2
    )
    print("diffold", loss1, loss_dict1)
    print("unifold", loss2, loss_dict2)
    assert equal(loss1, loss2)

    return True


def test_chi_loss():
    print("test chi loss.")
    loss_dict1 = {}
    loss_dict2 = {}
    aatype = torch.randint(0, 21, (bsz, nres))
    pred_unnormed = torch.randn((bsz, nres, 7, 2))
    pred = pred_unnormed / pred_unnormed.norm(dim=-1, keepdim=True)
    true = torch.randn((bsz, nres, 7, 2))
    true /= true.norm(dim=-1, keepdim=True)

    torsion_angle_mask = torch.randint(0, 2, size=(bsz, nres, 7)).float()
    seq_mask = torch.randint(0, 2, size=(bsz, nres)).float()
    bfactor = seq_mask

    # true = pred
    loss1 = supervised_chi_loss(
        pred,
        pred_unnormed,
        true,
        aatype,
        bfactor,
        torsion_angle_mask,
        seq_mask,
        chi_weight=1,
        angle_norm_weight=0.01,
        eps=1e-6,
        loss_dict=loss_dict1,
    )
    loss2 = uf_chi_loss(
        pred[None],
        pred_unnormed[None],
        true[..., 3:, :],
        aatype,
        seq_mask,
        torsion_angle_mask[..., 3:] * seq_mask[..., None],
        chi_weight=1,
        angle_norm_weight=0.01,
        eps=1e-6,
        loss_dict=loss_dict2,
    )
    print("diffold", loss1, loss_dict1)
    print("unifold", loss2, loss_dict2)
    assert equal(loss1, loss2)
    return True


def test_plddt_loss():
    print("test plddt loss.")
    num_bins = config.loss.predicted_lddt.num_bins
    loss_dict1 = {}
    loss_dict2 = {}
    logits = torch.randn((bsz, nres, num_bins))
    pred_ca = torch.randn((bsz, nres, 3))
    true_ca = torch.randn((bsz, nres, 3))
    pred_all_atom = torch.zeros((bsz, nres, 37, 3))
    pred_all_atom[..., 1, :] = pred_ca
    true_all_atom = torch.zeros((bsz, nres, 37, 3))
    true_all_atom[..., 1, :] = true_ca
    frame_mask = torch.randint(0, 2, size=(bsz, nres)).float()
    all_atom_mask = torch.zeros((bsz, nres, 37))
    all_atom_mask[..., 1] = frame_mask
    loss1 = plddt_loss(
        logits,
        pred_ca,
        true_ca,
        frame_mask,
        num_bins=num_bins,
        smooth_factor=0.,
        resolution=torch.tensor([2.]),
        eps=1e-10,
        loss_dict=loss_dict1
    )
    loss2 = uf_plddt_loss(
        logits,
        pred_all_atom,
        true_all_atom,
        all_atom_mask,
        resolution=torch.tensor([2.]),
        num_bins=num_bins,
        loss_dict=loss_dict2
    )
    print("diffold", loss1, loss_dict1)
    print("unifold", loss2, loss_dict2)
    assert equal(loss1, loss2)
    return True


def test_pae_loss():
    print("test pae loss.")
    num_bins = config.loss.predicted_aligned_error.num_bins
    loss_dict1 = {}
    loss_dict2 = {}
    logits = torch.randn((bsz, nres, nres, num_bins))
    pred_frames = Frame(
        Rotation(uniform_random_rotations((bsz, nres, 3, 3))),
        torch.randn((bsz, nres, 3))
    ).to_tensor_4x4()
    true_frames = Frame(
        Rotation(uniform_random_rotations((bsz, nres, 3, 3))),
        torch.randn((bsz, nres, 3))
    ).to_tensor_4x4()
    frame_mask = torch.randint(0, 2, size=(bsz, nres)).float()
    loss1 = pae_loss(
        logits, pred_frames, true_frames, frame_mask,
        resolution=torch.tensor([2.]),
        smooth_factor=0.,
        eps=1e-10,
        loss_dict=loss_dict1,
    )
    loss2 = uf_pae_loss(
        logits,
        pred_frames,
        true_frames,
        frame_mask,
        resolution=torch.tensor([2.]),
        eps=1e-10,
        loss_dict=loss_dict2
    )
    loss_dict2.pop("aligned_error")
    print("diffold", loss1, loss_dict1)
    print("unifold", loss2, loss_dict2)
    assert equal(loss1, loss2)
    return True


def test_sequence_loss():
    num_classes = 21
    num_layers = 4
    loss_dict1 = {}
    logits = torch.randn((num_layers, bsz, nres, num_classes))
    true_aatype = torch.randint(0, num_classes, (bsz, nres))
    diff_aatype = true_aatype
    seq_mask = torch.randint(0, 2, size=(bsz, nres))
    loss1 = sequence_denoise_loss_v3(
        logits,
        true_aatype,
        diff_aatype,
        seq_mask,
        smooth_factor=0.,
        eps=1e-10,
        layer_weighting="uniform",
        loss_dict=loss_dict1,
    )
    loss2 = softmax_cross_entropy(
        logits, one_hot(true_aatype[None], num_classes)
    ).mean(dim=0)
    loss2 = masked_mean(seq_mask, loss2, dim=-1, eps=1e-10)
    print("diffold", loss1, loss_dict1)
    print("unifold", loss2)
    assert equal(loss1, loss2)
    return True


def test_frame_loss():
    num_layers = 4
    pred_frames = Frame(
        Rotation(uniform_random_rotations((num_layers, bsz, nres, 3, 3))),
        torch.randn((num_layers, bsz, nres, 3))
    )
    true_frames = Frame(
        Rotation(uniform_random_rotations((bsz, nres, 3, 3))),
        torch.randn((bsz, nres, 3))
    )
    bfactor = torch.rand((bsz, nres))
    loss = compute_frame_loss(
        pred_frames,
        true_frames,
        bfactor,
        reference_pts=None,
        clamp_distance=1.,
        stack_dim=0,
        eps=1e-10
    )
    print(loss)
    return True


def test_ca_metrics():
    num_layers = 4
    pred_ca = torch.randn((num_layers, bsz, nres, 3))
    true_ca = torch.randn((bsz, nres, 3))
    bfactor = torch.rand((bsz, nres))
    metrics = compute_ca_metrics(
        pred_ca,
        true_ca,
        bfactor,
    )
    print(metrics)
    return True


def test_backbone_clash_loss():
    print("test bb clash.")
    num_layers = 4
    loss_dict = {}
    pred_ca = torch.linspace(0, 1 * (nres-1), nres)
    print(pred_ca)
    pred_ca = torch.tile(pred_ca, (num_layers, bsz, 3, 1))
    pred_ca = pred_ca.transpose(-1, -2)
    # pred_ca = torch.randn((num_layers, bsz, nres, 3))
    seq_mask = torch.randint(0, 2, size=(bsz, nres)).float()
    loss = backbone_clash_loss(
        pred_ca,
        seq_mask,
        2.0,
        "linear",
        1e-10,
        loss_dict
    )
    print(loss)
    print(loss_dict)
    return True


if __name__ == "__main__":
    test_distogram_loss()
    test_chi_loss()
    test_plddt_loss()
    test_pae_loss()
    test_sequence_loss()
    # test_frame_loss()
    # test_ca_metrics()
    test_backbone_clash_loss()
