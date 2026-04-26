import torch
import numpy as np
from typing import *
import json
import lmdb
import os
from unifold.data.lmdb_dataset import LMDBDataset
from unifold.dataset import Operation as SymmetryOp
from unifold.dataset import process_label as apply_symmetry_op_to_coord
from unifold.data import residue_constants as rc
import pickle, gzip

NumpyDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]
MultiMap = Dict[str, List[str]]

def recur_abstract(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        if len(x.shape):
            return f"{x.shape}_{x.dtype}"
        else:
            return f"{x.shape}_{x.dtype}={float(x):.6f}"
    elif isinstance(x, dict):
        return {k: recur_abstract(v) for k, v in x.items()}
    elif type(x) in (tuple, list):
        return [recur_abstract(v) for v in x]
    elif type(x) == str:
        return x
    else:
        return f"{type(x)}={x}"

def recur_print(x, key=''):
    if key:
        print(key)
    print(json.dumps(recur_abstract(x), indent=2))


def apply_symmetry_op(label: NumpyDict, op: SymmetryOp):
    label["all_atom_positions"] = apply_symmetry_op_to_coord(label["all_atom_positions"], label["all_atom_mask"], op)
    return label




_seq_map_trans = str.maketrans(
    ''.join(rc.restypes_with_x),
    ''.join(chr(v) for v in range(len(rc.restypes_with_x)))
)

def sequence_to_aatype(sequence: str, dtype: np.dtype = np.uint8):
    return np.frombuffer(
        (bytes(sequence.translate(_seq_map_trans), "ascii")), dtype=dtype
    )

def residue_to_all_atom_info(res):
    bfac = []
    mask = np.zeros((rc.atom_type_num,), dtype=bool)        # [37]
    pos = np.zeros((rc.atom_type_num, 3), dtype=float)      # [37, 3]
    for a in res.get_atoms():
        aidx = rc.atom_order.get(a.get_name(), None)
        if aidx is not None:
            pos[aidx, :] = a.get_coord()
            mask[aidx] = True
            bfac.append(a.get_bfactor())
        else:
            print(f"warning: {aidx} is not a valid atom type and is ignored.")
    mean_bfac = sum(bfac) / len(bfac)
    return mask, pos, mean_bfac

def load_json(path):
    return json.load(open(path))

def load_gzipped_pickle(path):
    return pickle.load(gzip.open(path, 'rb'))

from itertools import chain
def inverse_multi_map(map: MultiMap) -> Dict[str, str]:
    pairs = [
        [(k, v) for v in vals]
            for k, vals in map.items()
    ]
    inverse_map = {v: k for k, v in chain(*pairs)}
    return inverse_map


import signal
import functools

class TimeoutError(Exception):
    pass

def timeout(seconds=10, func_name=""):
    def decorator(func):
        error_message = f"function {func_name} timeout after {seconds} seconds."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handle_timeout(signum, frame):
                raise TimeoutError(error_message)

            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)

            return result

        return wrapper

    return decorator



def str_hash(text:str):
  hash=0
  for ch in text:
    hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
  return hash


import contextlib
@contextlib.contextmanager
def torch_seed(seed, *addl_seeds, key: str = None, cuda: bool = False):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    def check_seed(s):
        assert type(s) == int or type(s) == np.int32 or type(s) == np.int64
    check_seed(seed)
    if len(addl_seeds) > 0:
        for s in addl_seeds:
            check_seed(s)
        seed = int(hash((seed, *addl_seeds)) % 1e8)
        if key is not None:
            seed = int(hash((seed, str_hash(key))) % 1e8)
    cpu_state = torch.get_rng_state()
    torch.manual_seed(seed)
    if cuda:
        gpu_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda:
            torch.cuda.set_rng_state(gpu_state)


from unifold.data import protein
import numpy as np

def frames_to_pdb(frames: torch.Tensor, frames_mask: torch.Tensor, path: str):
    frames = frames.detach().float().cpu().numpy()
    frames_mask = frames_mask.detach().float().cpu().numpy()
    assert len(frames.shape) == 3
    seqlen = frames.shape[-3]
    aatype = np.ones([seqlen], dtype=int) * 20
    pos = np.zeros([seqlen, 37, 3])
    msk = np.zeros([seqlen, 37])
    pos[:, 1, :] = frames[:, :3, 3]
    msk[:, 1] = frames_mask
    residx = np.arange(seqlen, dtype=int) + 1
    asym_id = np.zeros(seqlen, dtype=int)
    prot = protein.Protein(
        atom_positions=pos,
        aatype=aatype,
        atom_mask=msk,
        residue_index=residx,
        chain_index=asym_id,
        b_factors=msk
    )
    pdb = protein.to_pdb(prot)
    with open(path, 'w') as f:
        f.write(pdb)


def to_numpy(x: torch.Tensor, reduce_batch_dim: bool = False):
    if reduce_batch_dim:
        x = x.squeeze(0)
    if x.dtype in (torch.float, torch.bfloat16, torch.float16):
        x = x.detach().cpu().float().numpy()
    elif x.dtype in (torch.long, torch.int, torch.int64):
        x = x.detach().cpu().long().numpy()
    else:
        raise ValueError(f"unknown dtype {x.dtype}")
    return x


def rigid_transform_3D(A, B, verbose=False):
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected


aatype_to_seq = lambda aatype: ''.join([
        rc.restypes_with_x[x] for x in aatype])