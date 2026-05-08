from dataclasses import dataclass

from factors.factor import Factor
from util.so3 import box_minus_right, lie_deriv_box_minus_right_R1, lie_deriv_box_minus_right_R2
from util.constants import I3
from numba import njit

@dataclass
class BetweenR3Factor(Factor):

    pass

def between_r3_error(state_tuple, weight=I3):
    """Calculate the error for a between factor on R^3.

    Inputs:
    - state_tuple:
        - p1: The first position vector.
        - p2: The second position vector.
    - weight: The weight for the error calculation.

    Returns:
    - The calculated error.

    """
    p1, p2 = state_tuple
    return weight @ (p1 - p2)

def between_r3_jacobian(state_tuple, weight=I3):
    """Calculate the Jacobian for a between factor on R^3.

    Inputs:
    - state_tuple:
        - p1: The first position vector.
        - p2: The second position vector
    - weight: The weight for the Jacobian calculation.

    Returns:
    - d_error_d_est1: The Jacobian of the error with respect to the first state.
    - d_error_d_est2: The Jacobian of the error with respect to the second state.

    """   
    return weight, -weight

def between_so3_error(state_tuple, weight=I3):
    """Calculate the error for a between factor on SO(3).

    Inputs:
    - state_tuple:
        - R1: The first SO(3) matrix.
        - R2: The second SO(3) matrix.
    - weight: The weight for the error calculation.

    Returns:
    - The calculated error.

    """
    R1, R2 = state_tuple
    return weight @ box_minus_right(R1, R2)

def between_so3_jacobian(state_tuple, weight=I3):
    """Calculate the Jacobian for a between factor on SO(3).

    Inputs:
    - state_tuple:
        - R1: The first SO(3) matrix.
        - R2: The second SO(3) matrix.
    - weight: The weight for the Jacobian calculation.

    Returns:
    - d_error_d_est1: The Jacobian of the error with respect to the first state.
    - d_error_d_est2: The Jacobian of the error with respect to the second state.

    """
    R1, R2 = state_tuple
    return weight@lie_deriv_box_minus_right_R1(R2, R1), weight@lie_deriv_box_minus_right_R2(R2, R1)
    

