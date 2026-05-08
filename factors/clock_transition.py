from dataclasses import dataclass
from typing import Any, Callable
from factors import Factor, RobustFactor

from numba import njit
from functools import partial
import numpy as np

@dataclass
class ClockTransitionFactor(Factor):
    pass

@dataclass
class RobustClockTransitionFactor(ClockTransitionFactor, RobustFactor):
    pass


def clock_transition_error(state_tuple, delta_t, weight=1.0):
    """Calculate the clock bias transition error.

    Inputs:
    - state_tuple:
        - cb0: The clock bias at time t0.
        - cd0: The clock drift at time t0.
        - cb1: The clock bias at time t1.
        - cd1: The clock drift at time t1.
        - delta_t: The time difference between t0 and t1.
    - weight: The weight for the error calculation.

    Returns:
    - The calculated error.

    """
    cb0, cd0, cb1, cd1 = state_tuple
    return weight @ np.array([
        cb1 - cb0- cd0*delta_t,
        cd1 - cd0
    ])

def clock_transition_error_jacobians(state_tuple, delta_t, weight=1.0):
    """Calculate the Jacobians for the clock bias transition factor.

    Inputs:
    - state_tuple:
        - cb0: The clock bias at time t0.
        - cd0: The clock drift at time t0.
        - cb1: The clock bias at time t1.
        - cd1: The clock drift at time t1.
        - delta_t: The time difference between t0 and t1.
    - weight: The weight for the error calculation.

    Returns:
    - A tuple containing the Jacobians of the error with respect to cb0, cd0, cb1, and cd1.

    """
    d_error_d_cb0 = weight @ np.array([[-1.0], [0.0]])
    d_error_d_cd0 = weight @ np.array([[-delta_t], [-1.0]])
    d_error_d_cb1 = weight @ np.array([[1.0], [0.0]])
    d_error_d_cd1 = weight @ np.array([[0.0], [1.0]])
    return d_error_d_cb0, d_error_d_cd0, d_error_d_cb1, d_error_d_cd1

def create_clock_transition_factor(bias_time_0, bias_time_1, clock_bias_0_key, clock_drift_0_key, clock_bias_1_key, clock_drift_1_key, delta_t, weight=1.0) -> ClockTransitionFactor:
    return ClockTransitionFactor(
            time=bias_time_1, 
            states=[clock_bias_0_key, clock_drift_0_key, clock_bias_1_key, clock_drift_1_key], 
            n_rows=2, # Two rows for the clock bias transition error 
            jacobians_size=8,  # Four elements in the Jacobian (one for each clock bias and drift)
            error_func=partial(clock_transition_error, delta_t=delta_t, weight = weight), 
            jacobian_func=partial(clock_transition_error_jacobians, delta_t=delta_t, weight=weight)
        )

def create_robust_clock_transition_factor(
        bias_time_0, 
        bias_time_1, 
        clock_bias_0_key, 
        clock_drift_0_key, 
        clock_bias_1_key, 
        clock_drift_1_key, 
        delta_t, 
        robust_weight_function: Callable[..., Any],
        weight=1.0,
        ) -> RobustClockTransitionFactor:
    
    return RobustClockTransitionFactor(
            time=bias_time_1, 
            states=[clock_bias_0_key, clock_drift_0_key, clock_bias_1_key, clock_drift_1_key], 
            n_rows=2, # Two rows for the clock bias transition error    
            jacobians_size=8,  # Four elements in the Jacobian (one for each clock bias and drift)
            error_func=partial(clock_transition_error, delta_t=delta_t, weight = weight),
            jacobian_func=partial(clock_transition_error_jacobians, delta_t=delta_t, weight=weight),
            robust_weight_func=robust_weight_function
        )