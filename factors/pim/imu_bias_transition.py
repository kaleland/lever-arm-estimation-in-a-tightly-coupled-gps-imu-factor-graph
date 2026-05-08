from dataclasses import dataclass
from factors import Factor, RobustFactor
from util.constants import I3
import numpy as np
from numba import njit
from functools import partial

@dataclass
class IMUBiasTransitionFactor(Factor):
    pass
@dataclass
class RobustIMUBiasTransitionFactor(IMUBiasTransitionFactor, RobustFactor):
    pass

@njit
def bias_transition_error(state_tuple, dt, tau, weight= I3):
    """
    Calculates cost associated with change between two bias states.

    Inputs:
    state_tuple:
        bias0: np.ndarray, shape (3), first bias state
        bias1: np.ndarray, shape (3), second bias state
    dt: float, time difference between the two bias states
    tau: float, time constant for bias dynamics
    weight: np.ndarray, shape (3, 3), weight matrix for the error

    Outputs:
    error: np.ndarray, shape (3), error between the two bias states
    """
    bias0, bias1 = state_tuple
    # Calculate the error between the two bias states
    return weight@(bias1 - np.exp(-dt/tau)*bias0) 

@njit
def bias_transition_jacobian(state_tuple, dt,tau, weight = I3):
    """
    Calculates the Jacobian of the bias transition error w.r.t. the two bias states.
    The Jacobians are the derivative of the error function for
    Gauss-Newton optimization.

    Inputs:
    dt: float, time difference between the two bias states
    tau: float, time constant for bias dynamics
    weight: np.ndarray, shape (3, 3), weight matrix for the error

    Outputs:
    d_error_d_bias0: np.ndarray, shape (3,3), Jacobian of the bias transition error
        w.r.t. the first bias state
    d_error_d_bias1: np.ndarray, shape (3,3), Jacobian of the bias transition error
        w.r.t. the second bias state
    """
    return -np.exp(-dt/tau)*weight, weight

def create_bias_transition_factor(bias_time_0, bias_time_1, imu_bias_0_key, imu_bias_1_key, dt, tau, weight=I3) -> IMUBiasTransitionFactor:
    
    return IMUBiasTransitionFactor(
            time=bias_time_1, 
            states=[imu_bias_0_key, imu_bias_1_key], 
            n_rows=3,  # Three rows for the IMU bias transition error
            jacobians_size=9*2,  # 9 elements in the Jacobian for each bias state (3x3 matrix)
            error_func=partial(bias_transition_error, dt=dt, tau=tau, weight=weight), 
            jacobian_func=partial(bias_transition_jacobian, dt=dt, tau=tau, weight=weight)
    )

def create_robust_bias_transition_factor(bias_time_0, bias_time_1, imu_bias_0_key, imu_bias_1_key, dt, tau, robust_weight_function, weight = np.eye(3)) -> RobustIMUBiasTransitionFactor:
    
    return RobustIMUBiasTransitionFactor(
            time=bias_time_1, 
            states=[imu_bias_0_key, imu_bias_1_key], 
            n_rows=3,  # Three rows for the IMU bias transition error
            jacobians_size=9*2,  # 9 elements in the Jacobian for each bias state (3x3 matrix)
            error_func=partial(bias_transition_error, dt=dt, tau=tau, weight=weight), 
            jacobian_func=partial(bias_transition_jacobian, dt=dt, tau=tau, weight=weight),
            robust_weight_func=robust_weight_function
    )