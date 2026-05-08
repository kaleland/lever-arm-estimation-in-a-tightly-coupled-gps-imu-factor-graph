
from util import so3, se3
import numpy as np
from numba import njit

@njit
def _se23_from_components(R: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Create a SE_2(3) extended pose matrix from rotation matrix R, velocity v, and position p."""
    T = np.eye(5, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3] = v
    T[:3,4] = p
    return T

@njit 
def _components_from_se23(T: np.ndarray) -> tuple:
    """Extract the rotation matrix, velocity, and position from a SE_2(3) extended pose matrix."""
    R = T[:3,:3]
    v = T[:3,3]
    p = T[:3,4]
    return R, v, p

@ njit
def adjoint_se23(T: np.ndarray) -> np.ndarray:
    """Adjoint representation of a SE_2(3) extended pose matrix."""
    if T.ndim == 2:
        adjoint = np.zeros((9, 9), dtype=np.float64)
        adjoint[:3, :3] = T[:3, :3]
        adjoint[3:6,3:6] = T[:3, :3]
        adjoint[6:9, 6:9] = T[:3, :3]
        adjoint[3:6,:3] = so3.skew(T[:3,3]) @ T[:3,:3]
        adjoint[6:9,:3] = so3.skew(T[:3,4]) @ T[:3,:3]
        return adjoint
    adjoint = np.zeros((T.shape[0], 9, 9), dtype=np.float64)
    adjoint[:,:3,:3] = T[:,:3,:3]
    adjoint[:,3:6,3:6] = T[:,:3,:3]
    adjoint[:,6:9,6:9] = T[:,:3,:3]
    for i in range(T.shape[0]):
        adjoint[i,3:6,:3] = so3.skew(T[i,:3,3]) @ T[i,:3,:3]
        adjoint[i,6:9,:3] = so3.skew(T[i,:3,4]) @ T[i,:3,:3]

    return adjoint

@njit
def skew_se23(v) -> np.ndarray:
    """"Skew-symmetric" matrix from a 9-dimensional vector in se_2(3). Note
    that the convention is different in that the rotation vector comes first. 
    This follows the definition from "Associating Uncertainty to Extended Poses
    for on Lie Group IMU Preintegration with Rotating Earth" by Brossard et al.
    """
    return np.array([[0.0, -v[2], v[1], v[3], v[6]],[v[2], 0.0, -v[0], v[4], v[7]],[-v[1], v[0], 0.0, v[5], v[8]],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0]])

@njit
def inv_se23(T: np.ndarray) -> np.ndarray:
    """Inverse of a SE_2(3) extended pose matrix."""
    Rinv = T[:3,:3].T
    vinv = -Rinv @ T[:3,3]
    pinv = -Rinv @ T[:3,4]
    return _se23_from_components(Rinv, vinv, pinv)

@njit
def diamond_minus_se23(T1,T2):
    
    return np.concatenate((so3.box_minus_right(T1[:3,:3], T2[:3,:3]),
                      T1[:3,3] - T2[:3,3],
                      T1[:3,4] - T2[:3,4]))

@njit
def diamond_plus_se23(T1,tau2):
    
    Rnew = T1[:3,:3] @ so3.expmap(tau2[:3])
    vnew = T1[:3,3]+tau2[3:6]
    pnew = T1[:3,4]+tau2[6:9]
    return _se23_from_components(Rnew, vnew, pnew)

## The function below is  from Micro Lie Theory, but looks wrong.
# def lie_jacobian_inverse_T_se23(T):
#     out = np.zeros((9, 9), dtype=np.float64)
#     out[:3,:3] = -T[:3,:3]
#     out[:3,3:6] = -so3.skew(T[:3,3]) @ T[:3,:3]
#     out[:3,6:9] = -so3.skew(T[:3,4]) @ T[:3,:3]
#     out[3:6,3:6] = -T[:3,:3]
#     out[6:9,6:9] = -T[:3,:3]

#     return out