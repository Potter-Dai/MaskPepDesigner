import torch
import numpy as np

raise NotImplementedError("do not use this.")

from scipy.spatial.transform import Rotation as R

def angle(v: np.ndarray) -> np.ndarray:
    ret = np.linalg.norm(v, axis=-1)
    if type(ret) == float:
        ret = np.array(ret)
    return ret

def log(r: np.ndarray) -> np.ndarray:
    batch_shape = r.shape[:-2]
    r = r.reshape(-1, 3, 3)
    v = R.as_rotvec(R.from_matrix(r))
    return v.reshape(*batch_shape, 3)

def exp(v: np.ndarray) -> np.ndarray:
    batch_shape = v.shape[:-1]
    v = v.reshape(-1, 3)
    r = R.as_matrix(R.from_rotvec(v))
    return r.reshape(*batch_shape, 3, 3)

def transpose_last(r: np.ndarray) -> np.ndarray:
    d = len(r.shape)
    return r.transpose(*range(d-2), d-1, d-2)

def left_grad(r: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    left gradients of d||log(r.T @ rh)|| / dr.
    """
    v = log(r @ transpose_last(rh))
    return v / angle(v)[..., None]

def right_grad(r: np.ndarray, rh: np.ndarray) -> np.ndarray:
    v = log(transpose_last(rh) @ r)
    return v / angle(v)[..., None]
