from unicore.models import register_model, register_model_architecture, BaseUnicoreModel

from .nn.dfv6 import Dfv6
from .config_v6 import model_config


@register_model("diffold")
class DiffoldModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--model-name",
            help="choose the model config",
        )

    def __init__(self, args):
        super().__init__()
        arch_v4(args)
        self.args = args
        config = model_config(
            self.args.model_name,
        )
        config.model.zormer_encoder.checkpoint_ratio = args.checkpoint_ratio
        self.model = Dfv6(config)
        self.config = config

    def half(self):
        self.model = self.model.half()
        return self

    def bfloat16(self):
        self.model = self.model.bfloat16()
        return self

    def float(self):
        self.model = self.model.float()
        return self

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    def forward(self, batch, **kwargs):
        outputs = self.model.forward(batch)
        return outputs, self.config.loss

@register_model_architecture("diffold", "diffold_v6")
def arch_v4(args):
    args.model_name = getattr(args, "model_name", "diffold_v6")
