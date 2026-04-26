import torch
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss

from .losses.dfv6_losses import dfv6_loss


@register_loss("diffold")
class DiffoldLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, batch, reduce=True):
        if "dummy" in batch:
            device = batch["dummy"].device
            return (
                torch.tensor(0., dtype=torch.float, device=device, requires_grad=True),
                0,
                {"sample_size": 0}
            )
        out, loss_cfg = model(batch)    # return loss config in model.
        # out, batch = self.to_float(out, batch)
        return dfv6_loss(out, batch, loss_cfg)

    @staticmethod
    def to_float(out, batch):
        for k in out["sm"]:
            out["sm"][k] = out["sm"][k].float()
        for k in out["aux"]:
            if "logits" in k:
                out["aux"][k] = out["aux"][k].float()
        for k in out:
            if "frames" in k:
                out[k] = out[k].float()
        for k in (
            "bfactor",
            "seq_mask",
        ):
            batch[k] = batch[k].float()
        return out, batch

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        logging_outputs = [log for log in logging_outputs if log.get("sample_size", 0) > 0]
        sample_size_sum = sum(log.pop("sample_size") for log in logging_outputs)
        if sample_size_sum == 0:
            return

        loss_sum = sum(log.get("loss", 0.) for log in logging_outputs)
        sample_weight_sum = sum(log.get("sample_weight", 0.) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size_sum, sample_size_sum, round=4)
        metrics.log_scalar("loss_by_weight", loss_sum / sample_weight_sum, sample_size_sum, round=4)

        for key in logging_outputs[0]:
            if key in ["sample_size", "bsz", "sample_weight", "loss"]:
                continue
            loss_sum = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(key, loss_sum / sample_size_sum, sample_size_sum, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
