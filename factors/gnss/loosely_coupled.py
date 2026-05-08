from factors import Factor, RobustFactor
from util import so3, gnss
import numpy as np
from dataclasses import dataclass
from functools import partial
from typing import Callable, Any

@dataclass
class LooselyCoupledGNSSFactor(Factor):

    weight: float # Weight for the factor (standard scalar or matrix)

@dataclass
class RobustLooselyCoupledGNSSFactor(LooselyCoupledGNSSFactor, RobustFactor):

    pass

def loosely_coupled_gnss_error(state_tuple, p_meas, v_meas, wb_raw, weight = np.eye(6)):
    """Calculate the loosely coupled GNSS error.
    
    Inputs:
    - state_tuple:
        - pw: The position of the IMU in world frame.
        - vw: The velocity of the IMU in world frame.
        - wRb: The rotation matrix from the body frame to the world frame.
        - bg: The gyroscope bias in the body frame.
        - lb: The lever arm vector from the IMU to the antenna in the body frame.
    - p_meas: The measured antenna position in the world frame.
    - v_meas: The measured antenna velocity in the world frame.
    - wb_raw: The raw gyroscope measurement in the body frame.
    - weight: The weight for the error calculation (Cholesky factor of inverse covariance).
    
    Returns:
    - The weighted error (6x1).

    """
    pw, vw, wRb, bg, lb = state_tuple
    wb = wb_raw - bg
    
    p_hat = pw + wRb @ lb
    v_hat = vw + wRb @ np.cross(wb, lb)
    
    error = np.concatenate([p_meas - p_hat, v_meas - v_hat])
    return weight @ error  

def loosely_coupled_gnss_jacobians(state_tuple, p_meas, v_meas, wb_raw, weight = np.eye(6)):
    """Calculate the loosely coupled GNSS Jacobians.
    
    Inputs:
    - state_tuple:
        - pw, vw, wRb, bg, lb: Same as loosely_coupled_gnss_error.
    - p_meas, v_meas, wb_raw: Same as loosely_coupled_gnss_error.
    - weight: The weight matrix.
    
    Returns:
    - A tuple containing the Jacobians of the error with respect to pw, vw, attitude, bg, and lb.

    """
    pw, vw, wRb, bg, lb = state_tuple
    wb = wb_raw - bg
    
    # Pre-calculate skew symmetric matrices
    skew_lb = so3.skew(lb)
    skew_wb = so3.skew(wb)
    skew_v_lever = so3.skew(np.cross(wb, lb))
    
    # 6x15 Jacobian (or individual Jacobians for each 3x1 state component)
    # Error = [ p_meas - (pw + R*lb) ]
    #         [ v_meas - (vw + R*(wb x lb)) ]
    
    # d_error / d_pw
    de_dpw = np.zeros((6, 3))
    de_dpw[0:3, :] = -np.eye(3)
    
    # d_error / d_vw
    de_dvw = np.zeros((6, 3))
    de_dvw[3:6, :] = -np.eye(3)
    
    # d_error / d_theta (body-frame attitude error: R <- R * exp(theta))
    de_dtheta = np.zeros((6, 3))
    # d(p_meas - R*lb) / d_theta = - R * [-lb]x = R * [lb]x
    de_dtheta[0:3, :] = wRb @ skew_lb
    # d(v_meas - R*(wb x lb)) / d_theta = - R * [-(wb x lb)]x = R * [wb x lb]x
    de_dtheta[3:6, :] = wRb @ skew_v_lever
    
    # d_error / d_bg
    de_dbg = np.zeros((6, 3))
    # d(v_meas - R*( (wb_raw - bg) x lb )) / d_bg
    # = - R * d( (wb_raw - bg) x lb ) / d_bg
    # = - R * d( -lb x (wb_raw - bg) ) / d_bg
    # = R * d( lb x (wb_raw - bg) ) / d_bg
    # = R * [lb]x * (-I) = - R * [lb]x
    de_dbg[3:6, :] = -wRb @ skew_lb
    
    # d_error / d_lb
    de_dlb = np.zeros((6, 3))
    # d(p_meas - R*lb) / d_lb = -R
    de_dlb[0:3, :] = -wRb
    # d(v_meas - R*(wb x lb)) / d_lb = - R * [wb]x
    de_dlb[3:6, :] = -wRb @ skew_wb
    
    return weight @ de_dpw, weight @ de_dvw, weight @ de_dtheta, weight @ de_dbg, weight @ de_dlb


def create_loosely_coupled_gnss_factor(
        p_meas,
        v_meas,
        wb_raw,
        time,
        pw_key,
        vw_key,
        R_key,
        bg_key,
        lb_key,
        weight = np.eye(6)) -> LooselyCoupledGNSSFactor:
    """Create a loosely coupled GNSS factor."""
    return LooselyCoupledGNSSFactor(
        time = time,
        states = [pw_key, vw_key, R_key, bg_key, lb_key],
        n_rows = 6,
        jacobians_size = 6*3+6*3+6*3+6*3+6*3, # 6 rows, 5 state components each with 3 dimensions
        error_func = partial(loosely_coupled_gnss_error, p_meas=p_meas, v_meas=v_meas, wb_raw=wb_raw, weight=weight),
        jacobian_func = partial(loosely_coupled_gnss_jacobians, p_meas=p_meas, v_meas=v_meas, wb_raw=wb_raw, weight=weight),
        weight = weight
    )

def create_robust_loosely_coupled_gnss_factor(
        p_meas,
        v_meas,
        wb_raw,
        time,
        pw_key,
        vw_key,
        R_key,
        bg_key,
        lb_key,
        robust_weight_func: Callable[..., Any],
        weight = np.eye(6)) -> RobustLooselyCoupledGNSSFactor:
    """Create a robust loosely coupled GNSS factor."""
    return RobustLooselyCoupledGNSSFactor(
        time = time,
        states = [pw_key, vw_key, R_key, bg_key, lb_key],
        n_rows = 6,
        jacobians_size = 6*3+6*3+6*3+6*3+6*3, # 6 rows, 5 state components each with 3 dimensions
        error_func = partial(loosely_coupled_gnss_error, p_meas=p_meas, v_meas=v_meas, wb_raw=wb_raw, weight=weight),
        jacobian_func = partial(loosely_coupled_gnss_jacobians, p_meas=p_meas, v_meas=v_meas, wb_raw=wb_raw, weight=weight),
        weight = weight,
        robust_weight_func = robust_weight_func
    )


