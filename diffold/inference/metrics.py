import json
from matplotlib import pyplot as plt

from unifold.losses.geometry import kabsch_rmsd

def eval_metrics(
    sh_0, fh_0, s_0, f_0,
    seq_mask, frame_mask,
    t, plddt,
    eps = 1e-12
):
    ph_0 = fh_0[..., :3, 3]
    p_0 = f_0[..., :3, 3]               # L, 3
    # err_sq = ((ph_0 - p_0) ** 2).sum(dim=-1)
    # msd = (err_sq * frame_mask).sum() / (frame_mask.sum() + eps)
    # rmsd = msd.sqrt()
    rmsd = kabsch_rmsd(p_0, ph_0, frame_mask > 0.)
    correct = (sh_0.argmax(dim=-1) == s_0.argmax(dim=-1)).float()
    acc = (correct * seq_mask).sum() / (seq_mask.sum() + eps)
    plddt = (plddt * seq_mask).sum() / (seq_mask.sum() + eps)
    return {
        "t": float(t), "rmsd": float(rmsd), "acc": float(acc), "plddt": float(plddt),
    }


def save_metrics(path, metrics):
    metrics = {k: v.tolist() for k, v in metrics.items()}
    with open(path, "w") as fp:
        json.dump(metrics, fp, indent=2)


def plot_metrics(path, metrics):
    fig = plt.figure(figsize=(10, 10))
    (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2, sharex="all")
    ax1.plot(metrics["rmsd"])
    ax1.set_ylabel("rmsd")
    ax2.plot(metrics["acc"])
    ax2.set_ylabel("acc")
    ax3.plot(metrics["plddt"])
    ax3.set_ylabel("plddt")
    ax4.plot(metrics["t"])
    ax4.set_ylabel("t")
    fig.tight_layout()
    fig.savefig(path, dpi=300, format="png")
    return

