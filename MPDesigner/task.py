from unicore.tasks import (
    register_task,
    UnicoreTask
)
from unicore.distributed.utils import get_data_parallel_world_size

from .dataset import MPDesignerDataset

import logging
logger = logging.getLogger(__name__)

@register_task("MPDesigner")
class MPDesignerTask(UnicoreTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "--unifold-db-path",
            type=str,
            default="",
        )
        parser.add_argument(
            "--uf-distill-db-path",
            type=str,
            default="",
        )
        parser.add_argument(
            "--esmatlas-db-path",
            type=str,
            default="",
        )
        parser.add_argument(
            "--unifold-db-prefix",
            type=str,
            default="",
        )
        parser.add_argument(
            "--uf-distill-db-prefix",
            type=str,
            default="sd_",
        )
        parser.add_argument(
            "--valid-samples-per-step",
            type=int,
            default=128,
        )
        parser.add_argument(
            "--checkpoint-ratio",
            type=float,
            default=1.0,
        )

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        batch_size = (
            self.args.batch_size
            * get_data_parallel_world_size()
            * self.args.update_freq[0]
        )
        logger.info(f"loading dataset {split}")
        if split == "train":
            dataset = MPDesignerDataset(
                config=self.config,
                mode="train",
                task="train",
                unifold_db_path=self.args.unifold_db_path,
                uf_distill_db_path=self.args.uf_distill_db_path,
                esmatlas_db_path=self.args.esmatlas_db_path,
                unifold_db_prefix=self.args.unifold_db_prefix,
                uf_distill_db_prefix=self.args.uf_distill_db_prefix,
                max_samples=self.args.max_update * batch_size,
                seed=self.seed
            )
        else:
            dataset = MPDesignerDataset(
                config=self.config,
                mode="runtime_eval",
                task=split,
                unifold_db_path=self.args.unifold_db_path,
                uf_distill_db_path=self.args.uf_distill_db_path,
                esmatlas_db_path=self.args.esmatlas_db_path,
                unifold_db_prefix=self.args.unifold_db_prefix,
                uf_distill_db_prefix=self.args.uf_distill_db_prefix,
                max_samples=self.args.valid_samples_per_step,
                seed=self.seed
            )

        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        self.config = model.config
        import os
        os.makedirs(args.save_dir, exist_ok=True)
        with open(f"{args.save_dir}/config.json", 'w') as f:
            f.write(self.config.to_json(indent=2))
        os.system(f"echo `date` branch `git rev-parse --abbrev-ref HEAD` commit `git rev-parse HEAD` >> {args.save_dir}/HEAD")
        return model

    def disable_shuffling(self) -> bool:
        return True


@register_task("diffold")
class DiffoldTask(MPDesignerTask):
    """Compatibility registration for old training commands/checkpoints."""
