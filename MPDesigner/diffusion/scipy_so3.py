import numpy as np
from scipy.spatial.transform.rotation import Rotation as R

def Log(m: np.ndarray) -> np.ndarray:
    return R.as_rotvec(R.from_matrix(m))

def Exp(v: np.ndarray) -> np.ndarray:
    return R.as_matrix(R.from_rotvec(v))

def theta_and_axis(v):
    theta = np.linalg.norm(v, axis=-1)
    axis = v / theta[..., None]
    return theta, axis

def so3_sampling(n, sigma):
    z = np.random.randn(n, 3) * sigma
    mat = R.as_matrix(R.from_rotvec(z))
    return mat

def random(n):
    return R.as_matrix(R.random(n))
