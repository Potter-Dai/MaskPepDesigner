import torch
import numpy as np

import json

import time

from ml_collections import ConfigDict
from typing import *

import logging
logger = logging.getLogger(__name__)

# from unicore.data import data_utils
from unicore.data import UnicoreDataset
from unicore.data.data_utils import numpy_seed, collate_dict

from unifold.data import utils


from .data import (
    load,
    db,
    process,
    utils,
    data_ops
)
# from .diffusion import transition
from .diffusion.transition_v6 import Diffuser

DUMMY_BATCH = {
    "dummy": torch.tensor([1.])
}

class DiffoldDataset(UnicoreDataset):
    def __init__(
        self,
        config: ConfigDict,
        mode: str,
        task: str,
        unifold_db_path: Optional[str] = None,
        uf_distill_db_path: Optional[str] = None,
        esmatlas_db_path: Optional[str] = None,
        unifold_db_prefix: str = "",
        uf_distill_db_prefix: str = "sd_",
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.task = task
        self.seed = seed
        self.max_samples = max_samples or 1_000_000

        self.mode_cfg = mode_cfg = config.mode[mode]

        release_date_cutoff = mode_cfg.get("train_max_date", None)
        max_chains = mode_cfg.get("max_chains", None)

        self.eval_on_train = config.data.evaluate_on_training_set

        # get db sampling weights
        if mode == "train" or self.eval_on_train:
            sampling_probs = list(config.mode["train"].database_weights.items())
            self.db_keys = [s[0] for s in sampling_probs]
            self.db_probs = np.array([s[1] for s in sampling_probs], dtype=float)
            self.db_probs /= np.sum(self.db_probs)
        else:
            self.db_keys = ["multimer"]
            self.db_probs = [1.]

        if mode == "train":
            mode_prefix = "train_"
        elif "eval" in mode:
            if self.eval_on_train:    # only for debug
                mode_prefix = "train_"
            else:
                mode_prefix = "eval_"
        else:
            mode_prefix = ""

        if "monomer" in self.db_keys:
            self.uf_monomer_db = db.UnifoldDatabase(
                unifold_db_path,
                name=f"{task}_monomer",
                prefix=f"{unifold_db_prefix}{mode_prefix}",
                use_multimer=False,
                release_date_cutoff=release_date_cutoff,
                max_chains=max_chains
            )
        else:
            self.uf_monomer_db = None

        if "multimer" in self.db_keys:
            self.uf_multimer_db = db.UnifoldDatabase(
                unifold_db_path,
                name=f"{task}_multimer",
                prefix=f"{unifold_db_prefix}{mode_prefix}",
                use_multimer=True,
                release_date_cutoff=release_date_cutoff,
                max_chains=max_chains
            )
        else:
            self.uf_multimer_db = None

        if "afdb" in self.db_keys:
            self.uf_distill_db = db.UFDistillDatabase(
                uf_distill_db_path,
                name=f"{task}_afdb",
                prefix=uf_distill_db_prefix,
            )
        else:
            self.uf_distill_db = None

        if "esmdb" in self.db_keys:
            self.esmatlas_db = db.ESMAtlasDatabase(
                esmatlas_db_path,
                name=f"{task}_esmdb",
                save_to_self=config.data.save_esmdb_env,
            )
        else:
            self.esmatlas_db = None

        self.dbs = {
            "monomer": self.uf_monomer_db,
            "multimer": self.uf_multimer_db,
            "afdb": self.uf_distill_db,
            "esmdb": self.esmatlas_db,
        }

        # cached to accelerate.
        self.diffuser = Diffuser(config.data.diffusion)

        # logging
        dbs_print = json.dumps(
            {k: str(v) for k, v in self.dbs.items()}, indent=4
        )
        probs_print = json.dumps(
            {k: f"{float(v):.2f}" for k, v in zip(self.db_keys, self.db_probs)}, indent=4
        )
        logger.info(f"database loaded: {dbs_print}\nsampling probabilities: {probs_print}")

    def __getitem__(self, idx: int, random: Optional[bool] = None):
        random = (self.mode == "train" or self.eval_on_train) if random is None else random
        return self.sample(idx, random, self.seed)

    def sample(self, idx: int, random: bool, seed: int):
        if random:
            with numpy_seed(self.seed, idx, key="get_db"):
                db_key = self.db_keys[np.random.choice(
                    len(self.db_probs), replace=False, p=self.db_probs
                )]
        else:
            # alteratively iter dbs
            db_key = self.db_keys[idx % len(self.db_keys)]
            idx = idx // len(self.db_keys)

        target_db: db.BaseDatabase = self.dbs.get(db_key, None)
        if target_db is None:
            raise ValueError(f"database {db_key} does not exist.")

        raw_feat = target_db.sample(idx, random, seed)
        if raw_feat is None:
            logger.warning(f"{db_key} getting item timeout. return None.")
            return DUMMY_BATCH
        uid = db.decompress_uid(raw_feat["uid"])

        with numpy_seed(seed, idx, key="get_process_seed"):
            process_seed = np.random.randint(1 << 31)

        try:
            batch = process.process_features(
                raw_feat,
                self.config.data,
                self.mode_cfg,      # currently only "crop" used.
                self.task,
                self.diffuser,
                seed=process_seed,
            )
        except utils.TimeoutError:
            logger.warning(f"processing {uid} timeout. return None.")
            return DUMMY_BATCH

        return batch

    def __len__(self):  # return infty
        return self.max_samples

    def collater(self, samples):
        if not len(samples):
            return collate_dict(samples, dim=0)

        # 1. 过滤掉 dummy batch
        samples = [s for s in samples if "aatype" in s]
        if not len(samples):
            return collate_dict(samples, dim=0)

        # 2. 计算 batch 内最大残基数，作为统一的 pad 目标
        seqlen = [s["aatype"].shape[0] for s in samples]
        target_size = max(seqlen)
        target_size = data_ops.get_pad_size(target_size, self.config.data.crop.pad_size)

        schema = self.config.data.batch_features

        padded_samples = []
        for s in samples:
            cur_len = s["aatype"].shape[0]
            ps = {}
            for k, v in s.items():
                if k not in schema:
                    continue
                dims = schema[k]
                # 构建 padding：对 N_RES 或 None 维度 pad 到 target_size
                pad_widths = []
                for d, sz in zip(dims, v.shape):
                    if d == data_ops.N_RES or d is None:
                        pad_widths.append((0, target_size - sz))
                    else:
                        pad_widths.append((0, 0))
                pad_widths.reverse()
                from itertools import chain as ichain
                flat_pad = list(ichain(*pad_widths))
                ps[k] = torch.nn.functional.pad(v, flat_pad, value=0.)
            padded_samples.append(ps)

        return collate_dict(padded_samples, dim=0)
    
    def __str__(self) -> str:
        return str(self.dbs)

