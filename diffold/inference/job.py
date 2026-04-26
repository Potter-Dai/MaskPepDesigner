import os
import numpy as np
from unicore.data.data_utils import numpy_seed

import re

import json

from diffold.data import process

from . import contig as ctg, utils

from hashlib import md5

def hash(*args, mod=100_000):
    i = (";".join(str(a) for a in args)).encode()
    o = int.from_bytes(md5(i).digest(), "big") % mod
    return o


class Job(object):
    def __init__(
        self,
        id: str,
        contig: str = None,
        pdb: str = None,
        initial_t: float = 1.,
        num_steps: int = 1,
        num_replica: int = 1,
        seed: int = 0,
        save_trajectory: bool = True,
        save_embedding: str = "none",
        calc_metrics: bool = False,
        ignore_missing_residues: bool = False,
    ) -> None:
        self.id = id
        self.contig = contig
        self.pdb = pdb
        if type(initial_t) == list:
            self.initial_t = initial_t
        elif type(initial_t) == str:
            s, e, n = initial_t.strip().split("/")
            self.initial_t = np.linspace(float(s), float(e), int(n)).tolist()
        else:
            self.initial_t = [initial_t for _ in range(num_replica)]
        self.num_steps = num_steps
        self.num_replica = num_replica
        self.meta_seed = seed
        self.seeds = [hash(seed, r) for r in range(num_replica)]
        self.save_trajectory = save_trajectory
        self.save_embedding = save_embedding
        self.calc_metrics = calc_metrics
        self.ignore_missing_residues = ignore_missing_residues
        self._check()

    def _check(self):
        assert self.contig is not None or self.pdb is not None, \
            f"When no contig is provided, a pdb must be specified."
        assert re.match(r"^[\w]+$", self.id), \
            f"Job id should only contain letters, digits and `_`. Got `{self.id}`."
        assert len(self.initial_t) == self.num_replica, \
            f"Provided initial_t has incompatible length ({len(self.initial_t)}) " \
            f"with num replica ({self.num_replica})."
        assert np.all((np.array(self.initial_t) >= 0.) * (np.array(self.initial_t) <= 1.)), \
            f"Initial time should be in [0, 1]. Got `{self.initial_t}`."
        assert self.save_embedding in ("none", "single", "single+pair"), \
            f"Wrong save_embedding option {self.save_embedding}"

    def to_dict(self) -> dict:
        return {
            "contig": self.contig,
            "pdb": self.pdb,
            "initial_t": self.initial_t,
            "num_steps": self.num_steps,
            "num_replica": self.num_replica,
            "seed": self.meta_seed,
            "replica_seeds": self.seeds,
            "save_trajectory": self.save_trajectory,
        }

def load_job_batchs(
    job: Job,
    config,
    input_pdb_dir: str = None,
    diffuser = None,
    contig_parser: ctg.ContigParser = ctg.default_contig_parser,
    silent: bool = True,
    **contig_kwargs,
):
    if job.pdb is not None:
        pdb_path = os.path.join(input_pdb_dir, f"{job.pdb}.pdb")
        pdb_ref = utils.load_pdb_ref(name=job.id, path=pdb_path, silent=silent)
    else:
        pdb_ref = None

    contig_parser = contig_parser or ctg.ContigParser(**contig_kwargs)
    contig = contig_parser.parse(job.contig)

    batchs = []
    for seed, t in zip(job.seeds, job.initial_t):
        # resample length.
        with numpy_seed(seed, 0, key="length_sampling"):
            residx = ctg.contig_to_residx(contig)
        sample = utils.build_sample_from_residx(residx, pdb_ref)
        sample["diffusion_t"] = np.array(t, dtype=float)

        batch = process.process_features(
            sample,
            data_cfg=config.data,
            mode_cfg=config.mode.predict,
            task="predict",
            diffusion_transition=diffuser,
            seed=seed,
        )
        batchs.append(batch)

    return batchs


def load_replica_batch(
    # job: Job,
    pdb: str,
    contig: str,
    seed: int,
    init_t: float,
    config,
    input_pdb_dir: str = None,
    diffuser = None,
    contig_parser: ctg.ContigParser = ctg.default_contig_parser,
    silent: bool = True,
    ignore_missing_residues: bool = False,
    **contig_kwargs,
):
    if pdb is not None:
        pdb_path = os.path.join(input_pdb_dir, f"{pdb}.pdb")
        pdb_ref = utils.load_pdb_ref(name="", path=pdb_path, silent=silent)
    else:
        pdb_ref = None

    contig_parser = contig_parser or ctg.ContigParser(**contig_kwargs)
    contig = contig_parser.parse(contig)

    with numpy_seed(seed, 0, key="length_sampling"):
        residx = ctg.contig_to_residx(contig)
    sample = utils.build_sample_from_residx(residx, pdb_ref, ignore_missing_residues)
    sample["diffusion_t"] = np.array(init_t, dtype=float)
    batch = process.process_features(
        sample,
        data_cfg=config.data,
        mode_cfg=config.mode.predict,
        task="predict",
        diffusion_transition=diffuser,
        seed=seed,
    )

    return batch
