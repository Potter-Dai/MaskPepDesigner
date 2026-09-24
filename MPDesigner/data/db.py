import numpy as np
import lmdb
import os
import pickle, gzip
import datetime
from typing import *

from collections import Counter
from itertools import chain
from unicore.data.data_utils import numpy_seed

from unifold.data.utils import lru_cache
from unifold.data.lmdb_dataset import LMDBDataset

from .utils import NumpyDict, MultiMap, load_json, inverse_multi_map, timeout, TimeoutError
from .load import preprocess, load_unifold_reference

import logging

logger = logging.getLogger(__name__)
LOAD_TIMEOUT = 30

import string

class BaseDatabase:
    def __init__(self, directory: str, name: str=''):
        assert type(directory) == str and os.path.exists(directory), f"{directory} is not a path."
        self.directory = directory
        self.name = name

    def get(self, *args, **kwargs) -> NumpyDict:
        raise NotImplementedError()

    def sample(self, data_index: int, random: bool = True, seed: Optional[int] = None) -> NumpyDict:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"[{self.name} @ {self.directory}]"

    def handle_timeout(self, key, seconds):
        logger.warning(f"{self.name} getting {key} timeout after {seconds} seconds")

def compress_uid(uid: str) -> np.ndarray:
    arr = np.array([ord(c) for c in uid], dtype=int)
    return arr

def decompress_uid(uid: np.ndarray) -> str:
    assert len(uid.shape) == 1, uid.shape
    chars = [chr(int(i) & 0xff) for i in uid]
    return "".join(c for c in chars if c in string.printable)

class ESMAtlasDatabase(BaseDatabase):
    # cls consts
    _num_lmdbs = 1000
    _num_int_bytes = 4
    _byte_order  = "big"

    def __init__(
        self,
        directory: str,
        name: str = "esm_database",
        save_to_self: bool = False,
    ):
        super().__init__(directory, name)
        self._envs = [None] * self._num_lmdbs
        self._env_sizes = [None] * self._num_lmdbs
        self.save_to_self = save_to_self
        logger.info(
            f"{self.name} loaded {self._num_lmdbs} sub-lmdbs."
        )

    def _lmdb_path(self, lmdb_id: int):
        return os.path.join(self.directory, f"{lmdb_id:03d}.lmdb")

    def _lmdb_size(self, lmdb_id: int):
        self._connect_db(lmdb_id)
        return self._env_sizes[lmdb_id]

    def _connect_db(self, lmdb_id: int):
        if self._envs[lmdb_id] is None:
            env = lmdb.Environment(
                self._lmdb_path(lmdb_id),
                subdir=False,
                readonly=True,
                readahead=False,
                meminit=False,
                lock=False,
                max_readers=256,
            )
            self._env_sizes[lmdb_id] = env.stat()["entries"]
            if self.save_to_self:
                self._envs[lmdb_id] = env
        else:
            env = self._envs[lmdb_id]
        return env

    def _int_to_key(self, key: int):
        return int(key).to_bytes(self._num_int_bytes, self._byte_order)

    @lru_cache(maxsize=16, deepcopy=True)
    @timeout(LOAD_TIMEOUT, "esmdb_get")
    def get(
        self,
        lmdb_id: int,
        key: int,
        allow_larger_key: bool = False,
    ) -> NumpyDict:
        env = self._connect_db(lmdb_id)

        if key > self._lmdb_size(lmdb_id):
            if not allow_larger_key:
                raise ValueError(f"key {key} exceeded lmdb size {self._env_sizes[lmdb_id]}")
            else:
                key = key % self._lmdb_size(lmdb_id)

        val = env.begin().get(
            self._int_to_key(key),
            default=None
        )

        if val is None:
            raise ValueError(
                f"{key} does not exist in esm database {self.directory}, "
                f"which is unexpected. Database should be checked."
            )

        val = pickle.loads(gzip.decompress(val))
        uid = "MGPY_" + val.get("esm_id", "UNKNOWN")
        val = preprocess(val)
        val["uid"] = compress_uid(uid)

        return val

    def sample(
        self,
        data_index: int,
        random: bool = True,
        seed: Optional[int] = None,
    ):
        if random:
            with numpy_seed(seed, data_index, key=f"sample_{self.name}"):
                lmdb_id = np.random.randint(0, self._num_lmdbs)
                key = np.random.randint(0, self._lmdb_size(lmdb_id))
        else:
            lmdb_id = data_index % self._num_lmdbs
            key = lmdb_id // self._num_lmdbs
        try:
            ret = self.get(lmdb_id, key, allow_larger_key=True)
        except TimeoutError:
            self.handle_timeout(f"{lmdb_id}/{key}", LOAD_TIMEOUT)
            ret = None
        return ret

def strpdate(date: str):
    return datetime.datetime.strptime(date, "%Y-%m-%d")

AssemblyInfo = Dict[str, Dict[str, List[Any]]]

class UnifoldDatabase(BaseDatabase):
    def __init__(
        self,
        directory: str,
        name: str = "unifold_database",
        prefix: str = "",
        use_multimer: bool = True,
        release_date_cutoff: Optional[str] = None,
        max_chains: Optional[int] = None,
    ):
        super().__init__(directory, name)
        self.prefix = prefix

        # 1. load lmdbs (lazy initializing)
        self.feature_lmdb = LMDBDataset(os.path.join(self.directory, "features.lmdb"))
        self.label_lmdb = LMDBDataset(os.path.join(self.directory, "labels.lmdb"))

        # 2. load sequence -> chain id mapping. This maps from unique sequence id
        # (e.g. "PDB_AAACTU") to multiple pdb chain id (e.g. ["7uas_A", "7uas_B"])
        self.sequence_to_chains: MultiMap = load_json(
            os.path.join(self.directory, prefix + "multi_label.json")
        )

        self.release_date_cutoff = release_date_cutoff
        if self.release_date_cutoff is not None:
            self._filter_by_date(release_date_cutoff)

        self.chain_to_sequence = inverse_multi_map(self.sequence_to_chains)

        # 3. load multimer metadata and sample weights
        self.use_multimer = use_multimer
        if self.use_multimer:
            self.assembly_info: AssemblyInfo = load_json(
                os.path.join(self.directory, self.prefix + "mmcif_assembly.json")
            )
            self._filter_assembly_by_known_chains(self.chain_to_sequence.keys())
            self.max_chains = max_chains
            if self.max_chains is not None:
                self._filter_assembly_by_max_chains(max_chains)
            sample_weights = self._compute_assembly_sample_weights(
                self.assembly_info, clamp_max=2.0
            )
            logger.info(
                f"{self.name} loaded "
                f"{len(self.assembly_info)} assemblies, "
                f"{len(self.chain_to_sequence)} chains with "
                f"{len(self.sequence_to_chains)} sequences."
            )
        else:
            sample_weights = self._compute_chain_sample_weights(self.sequence_to_chains.keys())
            logger.info(
                f"{self.name} loaded "
                f"{len(self.chain_to_sequence)} chains with "
                f"{len(self.sequence_to_chains)} sequences."
            )

        # 4. normalize sample weights as probabilities
        (
            self.num_samples,
            self.sample_keys,
            self.sample_probs,
        ) = self._make_sampling_probabilities(sample_weights)

    def _filter_by_date(self, release_date_cutoff: str):
        """filter out chains that were released after given date (fmt YYYY-MM-DD)"""
        if not release_date_cutoff:
            return

        release_date_cutoff = strpdate(release_date_cutoff)
        if release_date_cutoff > datetime.datetime.now():
            return

        release_dates_path = os.path.join(self.directory, "release_date.json")

        if not os.path.exists(release_dates_path):
            logger.warn(f"{release_dates_path} does not exist. All chains kept.")
            return

        release_dates: MultiMap = load_json(release_dates_path)

        def before_cutoff(chain_id: str):
            pdb_id = chain_id.split("_")[0]
            date = release_dates.get(pdb_id, None)
            if date is not None:
                date = strpdate(date[0])
                return date <= release_date_cutoff
            else:
                logger.warn(f"cannot find release dates of {pdb_id}. Sample ignored.")
                return False

        ret = {
            seq_id: [c for c in chains if before_cutoff(c)]
            for seq_id, chains in self.sequence_to_chains.items()
        }

        ret = {
            seq_id: chains
            for seq_id, chains in ret.items()
            if len(chains)
        }

        self.sequence_to_chains = ret
        return self.sequence_to_chains

    def _filter_assembly_by_known_chains(self, known_chains: Iterable[str]):
        """
        This function removes unknown chains the assembly_info json, making it contain
        only known chains in the database to avoid possible failures.
        """
        if not self.assembly_info:
            return

        def filter_pdb(pdb_id, pdb_item):
            out = [
                (ch, op)
                for ch, op in zip(pdb_item["chains"], pdb_item["opers"])
                if f"{pdb_id}_{ch}" in known_chains
            ]
            if len(out):
                return {
                    "chains": [v[0] for v in out],
                    "opers": [v[1] for v in out],
                }
            else:
                return None

        ret = {
            k: filter_pdb(k, v)
            for k, v in self.assembly_info.items()
        }
        ret = {
            k: v
            for k, v in ret.items()
            if v is not None
        }   # skip empty pdbs

        self.assembly_info = ret
        return self.assembly_info

    def _filter_assembly_by_max_chains(self, max_chains: int):
        """ignores pdbs with chains above given threshold."""
        ret = {
            k: v
            for k, v in self.assembly_info.items()
            if len(v["chains"]) <= max_chains
        }
        self.assembly_info = ret
        return self.assembly_info

    def _compute_chain_sample_weights(self, sequences: Iterable[str]):
        # 1. count appearances of each sequences
        sequence_counts = {
            k: len(v) for k, v in self.sequence_to_chains.items()
        }

        # 2. load and adjust cluster sizes
        cluster_sizes: Dict[str, int] = load_json(
            os.path.join(self.directory, self.prefix + "cluster_size.json")
        )
        cluster_sizes = {
            k: cluster_sizes[k] * sequence_counts.get(k, 1)
            for k in cluster_sizes
        }
        sequence_lengths: Dict[str, int] = load_json(
            os.path.join(self.directory, self.prefix + "seq_length.json")
        )

        # 3. calculate sample weights of chains
        def compute_sequence_sample_weight(length: int, cluster_size: int):
            p1 = max(min(length, 512), 256) / 512
            p2 = length ** 2 / 1024
            return min(p1, p2) / max(cluster_size, 1)

        sequence_sample_weights = {
            seq: compute_sequence_sample_weight(
                sequence_lengths[seq], cluster_sizes[seq]
            ) for seq in sequences
        }
        chain_sample_weights = {
            ch: sequence_sample_weights[seq]
            for ch, seq in self.chain_to_sequence.items()
        }

        return chain_sample_weights

    def _compute_assembly_sample_weights(self, assembly_info: AssemblyInfo, clamp_max: float = 2.0):
        # 1. count appearances of each sequences
        assembly_sequences: MultiMap = {
            k: [self.chain_to_sequence[f"{k}_{ch}"] for ch in v["chains"]]
            for k, v in assembly_info.items()
        }
        sequence_counts = Counter(chain(*assembly_sequences.values()))

        # 2. load and adjust cluster sizes
        cluster_sizes: Dict[str, int] = load_json(
            os.path.join(self.directory, self.prefix + "cluster_size.json")
        )
        cluster_sizes = {
            k: cluster_sizes[k] * sequence_counts.get(k, 1)
            for k in cluster_sizes
        }

        # 3. calculate sample weights of assemblies
        def compute_assembly_sample_weight(seqs):
            total_sample_weight = sum(
                1. / cluster_sizes[s] for s in seqs
            )
            return min(total_sample_weight, clamp_max)

        assembly_sample_weights = {
            k: compute_assembly_sample_weight(v)
            for k, v in assembly_sequences.items()
        }

        return assembly_sample_weights

    @staticmethod
    def _make_sampling_probabilities(sample_weights: Dict[str, float]):
        num_samples = len(sample_weights)
        sample_keys = list(sample_weights.keys())
        weights_array = np.array(list(sample_weights.values()))
        sample_probs = weights_array / np.sum(weights_array)
        return num_samples, sample_keys, sample_probs

    def sample(
        self,
        data_index: int,
        random: bool = True,
        seed: Optional[int] = None,
    ) -> NumpyDict:
        if random:
            with numpy_seed(seed, data_index, key=f"sample_{self.name}"):
                idx = np.random.choice(self.num_samples, p=self.sample_probs)
            key = self.sample_keys[idx]
        else:
            key = self.sample_keys[data_index % len(self.sample_keys)]

        try:
            ret = self.get(key)
        except TimeoutError:
            self.handle_timeout(str(key), LOAD_TIMEOUT)
            ret = None
        return ret

    @lru_cache(maxsize=16, deepcopy=True)
    @timeout(LOAD_TIMEOUT, "ufdb_get")
    def get(
        self,
        key: str,
    ):
        if self.use_multimer:
            assembly = self.assembly_info[key]
            chain_ids = [f"{key}_{ch}" for ch in assembly["chains"]]
            seq_ids = [self.chain_to_sequence[ch] for ch in chain_ids]
            val = load_unifold_reference(
                sequence_ids=seq_ids,
                feature_lmdb=self.feature_lmdb,
                label_ids=chain_ids,
                label_lmdb=self.label_lmdb,
                symmetry_ops=assembly["opers"]
            )
            val["uid"] = compress_uid(f"{self.name}_{key}")
            return val
        else:
            seq_id = self.chain_to_sequence[key]
            val = load_unifold_reference(
                sequence_ids=[seq_id],
                feature_lmdb=self.feature_lmdb,
                label_ids=[key],
                label_lmdb=self.label_lmdb,
                symmetry_ops=None
            )
            val["uid"] = compress_uid(f"{self.name}_{key}")
            return val


class UFDistillDatabase(UnifoldDatabase):
    def __init__(
        self,
        directory: str,
        name: str = "uf_distill_database",
        prefix: str = "",
    ):
        super(UnifoldDatabase, self).__init__(directory, name)
        self.prefix = prefix

        # 1. load lmdbs (lazy initializing)
        self.feature_lmdb = LMDBDataset(os.path.join(self.directory, "features.lmdb"))
        self.label_lmdb = LMDBDataset(os.path.join(self.directory, "labels.lmdb"))

        # 2. load sample weights
        sample_weights = self._load_sample_weights()
        logger.info(
                f"{self.name} loaded "
                f"{len(sample_weights)} sequences."
            )

        # 3. normalize sample weights as probabilities
        (
            self.num_samples,
            self.sample_keys,
            self.sample_probs,
        ) = self._make_sampling_probabilities(sample_weights)

    def _load_sample_weights(self):
        cluster_sizes: Dict[str, int] = load_json(
            os.path.join(self.directory, self.prefix + "cluster_size.json")
        )
        sequence_lengths: Dict[str, int] = load_json(
            os.path.join(self.directory, self.prefix + "seq_length.json")
        )

        def compute_sequence_sample_weight(length: int, cluster_size: int):
            p1 = max(min(length, 512), 256) / 512
            p2 = length ** 2 / 1024
            return min(p1, p2) / max(cluster_size, 1)

        sample_weights = {
            seq: compute_sequence_sample_weight(
                sequence_lengths[seq], cs
            ) for seq, cs in cluster_sizes.items()
        }

        return sample_weights

    @lru_cache(maxsize=16, deepcopy=True)
    @timeout(30, "afdb_get")
    def get(
        self,
        key: str,
    ):
        val = load_unifold_reference(
            sequence_ids=[key],
            feature_lmdb=self.feature_lmdb,
            label_ids=[key],
            label_lmdb=self.label_lmdb,
            symmetry_ops=None
        )
        val["uid"] = compress_uid(f"{self.name}_{key}")
        return val
