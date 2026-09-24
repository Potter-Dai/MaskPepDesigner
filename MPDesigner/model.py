from unicore.models import register_model, register_model_architecture, BaseUnicoreModel

from .nn.mpdesigner import MPDesigner
from .config_v6 import model_config


@register_model("MPDesigner")
class MPDesignerModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--model-name",
            help="choose the model config",
        )

    def __init__(self, args):
        super().__init__()
        mpdesigner_architecture(args)
        self.args = args
        config = model_config(
            self.args.model_name,
        )
        config.model.zormer_encoder.checkpoint_ratio = args.checkpoint_ratio
        self.model = MPDesigner(config)
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

@register_model_architecture("MPDesigner", "MPDesigner_v6")
def mpdesigner_architecture(args):
    if not getattr(args, "model_name", None):
        raise ValueError("Specify --model-name with the exact checkpoint configuration "
                         "(for example MPDesigner_v65); the architecture name alone "
                         "does not identify a trained configuration.")


@register_model("diffold")
class DiffoldModel(MPDesignerModel):
    """Compatibility registration for old training commands/checkpoints."""


@register_model_architecture("diffold", "diffold_v6")
def arch_v4(args):
    mpdesigner_architecture(args)
