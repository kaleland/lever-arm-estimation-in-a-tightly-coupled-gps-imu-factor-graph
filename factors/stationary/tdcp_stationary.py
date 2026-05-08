import factors.gnss.tdcp as tdcp

from functools import partial
import numpy as np
from typing import Callable, Any

def tdcp_stationary_error(state_tuple, *args, **kwargs):
    
    p, cb0, cb1, R = state_tuple
    expanded_state_tuple = (p, p, cb0, cb1, R, R)
    return tdcp.tdcp_error(*args, state_tuple = expanded_state_tuple, **kwargs)

def tdcp_stationary_unknown_lever_arm_error(state_tuple, *args, **kwargs):
    
    p, cb0, cb1, R, lb = state_tuple
    expanded_state_tuple = (p, p, cb0, cb1, R, R, lb)
    return tdcp.tdcp_error_unknown_lever_arm(*args, state_tuple = expanded_state_tuple, **kwargs)

def tdcp_stationary_jacobians(state_tuple, *args, **kwargs):
    
    p, cb0, cb1, R = state_tuple
    expanded_state_tuple = (p, p, cb0, cb1, R, R)
    de_dp0, de_dp1, de_dcb0, de_dcb1, de_dR0, de_dR1 = tdcp.tdcp_jacobians(*args, state_tuple = expanded_state_tuple, **kwargs)
    de_dp = de_dp0 + de_dp1
    de_dR = de_dR0 + de_dR1
    return de_dp, de_dcb0, de_dcb1, de_dR

def tdcp_stationary_unknown_lever_arm_jacobians(state_tuple, *args, **kwargs):
    
    p, cb0, cb1, R, lb = state_tuple
    expanded_state_tuple = (p, p, cb0, cb1, R, R, lb)
    de_dp0, de_dp1, de_dcb0, de_dcb1, de_dR0, de_dR1, de_dlb = tdcp.tdcp_jacobians_unknown_lever_arm(*args, state_tuple = expanded_state_tuple, **kwargs)
    de_dp = de_dp0 + de_dp1
    de_dR = de_dR0 + de_dR1
    return de_dp, de_dcb0, de_dcb1, de_dR, de_dlb

def create_tdcp_stationary_factor(
        meas,
        time0,
        time1,
        p_key,
        cb0_key,
        cb1_key,
        R_key,
        sat_xyz0,
        sat_xyz1,
        prn: int,
        weight = 1.0,
        lb = np.zeros(3)
    ) -> tdcp.TDCPFactor:
    
    return tdcp.TDCPFactor(
            time=time1,
            states = [
                p_key,
                cb0_key,
                cb1_key,
                R_key
            ],
            n_rows = 1,  # One row for the TDCP error
            jacobians_size = 3+2+3, # 3 for position, 2 for clock bias, 3 for attitude
            error_function = partial(tdcp_stationary_error,
                                     meas=meas,
                                     sat_xyz0=sat_xyz0,
                                     sat_xyz1=sat_xyz1,
                                     weight=weight,
                                     lb=lb),
            jacobian_function = partial(tdcp_stationary_jacobians,
                                       sat_xyz0=sat_xyz0,
                                       sat_xyz1=sat_xyz1,
                                       weight=weight,
                                       lb=lb),
            prn = prn,
            weight = weight
        )

def create_robust_tdcp_stationary_factor(
        meas,
        time0,
        time1,
        p_key,
        cb0_key,
        cb1_key,
        R_key,
        sat_xyz0,
        sat_xyz1,
        prn: int,
        robust_weight_function: Callable[..., Any],
        weight = 1.0,
        lb = np.zeros(3),
    ) -> tdcp.RobustTDCPFactor:
    
    return tdcp.RobustTDCPFactor(
            time=time1,
            states = [
                p_key,
                cb0_key,
                cb1_key,
                R_key
            ],
            n_rows = 1,  # One row for the TDCP error
            jacobians_size = 3+2+3, # 3 for position, 2 for clock bias, 3 for attitude
            error_func = partial(tdcp_stationary_error,
                                     meas=meas,
                                     sat_xyz0=sat_xyz0,
                                     sat_xyz1=sat_xyz1,
                                     weight=weight,
                                     lb=lb),
            jacobian_func = partial(tdcp_stationary_jacobians,
                                       sat_xyz0=sat_xyz0,
                                       sat_xyz1=sat_xyz1,
                                       weight=weight,
                                       lb=lb),
            prn = prn,
            weight = weight,
            robust_weight_func = robust_weight_function
        )

def create_tdcp_stationary_unknown_lever_arm_factor(
        meas,
        time0,
        time1,
        p_key,
        cb0_key,
        cb1_key,
        R_key,
        lb_key,
        sat_xyz0,
        sat_xyz1,
        prn: int,
        weight = 1.0,
    ) -> tdcp.TDCPFactorUnknownLeverArm:
    
    return tdcp.TDCPFactorUnknownLeverArm(
            time=time1,
            states = [
                p_key,
                cb0_key,
                cb1_key,
                R_key,
                lb_key
            ],
            n_rows = 1,  # One row for the TDCP error
            jacobians_size = 3+2+3+3, # 3 for position, 2 for clock bias, 3 for attitude, 3 for lever arm
            error_func = partial(tdcp_stationary_unknown_lever_arm_error,
                                     meas=meas,
                                     sat_xyz0=sat_xyz0,
                                     sat_xyz1=sat_xyz1,
                                     weight=weight),
            jacobian_func = partial(tdcp_stationary_unknown_lever_arm_jacobians,
                                       sat_xyz0=sat_xyz0,
                                       sat_xyz1=sat_xyz1,
                                       weight=weight),
            prn = prn,
            weight = weight
        )

def create_robust_tdcp_stationary_unknown_lever_arm_factor(
        meas,
        time0,
        time1,
        p_key,
        cb0_key,
        cb1_key,
        R_key,
        lb_key,
        sat_xyz0,
        sat_xyz1,
        prn: int,
        robust_weight_function: Callable[..., Any],
        weight = 1.0,
    ) -> tdcp.RobustTDCPFactorUnknownLeverArm:
    
    return tdcp.RobustTDCPFactorUnknownLeverArm(
            time=time1,
            states = [
                p_key,
                cb0_key,
                cb1_key,
                R_key,
                lb_key
            ],
            n_rows = 1,  # One row for the TDCP error       
            jacobians_size = 3+2+3+3, # 3 for position, 2 for clock bias, 3 for attitude, 3 for lever arm
            error_func = partial(tdcp_stationary_unknown_lever_arm_error,
                                     meas=meas,
                                     sat_xyz0=sat_xyz0,
                                     sat_xyz1=sat_xyz1,
                                     weight=weight),
            jacobian_func = partial(tdcp_stationary_unknown_lever_arm_jacobians,
                                       sat_xyz0=sat_xyz0,
                                       sat_xyz1=sat_xyz1,
                                       weight=weight),
            prn = prn,
            weight = weight,
            robust_weight_func = robust_weight_function
        )

