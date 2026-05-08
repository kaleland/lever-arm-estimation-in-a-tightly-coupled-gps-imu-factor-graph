from util import so3
from util.so3 import box_minus_right, lie_deriv_box_minus_right_R1
from util.constants import I3
from numba import njit

@njit
def prior_scalar_error(prior, state_tuple, weight=1.0):
    """Calculate the error for a scalar prior factor.

    Inputs:
    - state_tuple:
        - est: The estimated value.
    - prior: The prior value.

    Returns:
    - The calculated error.

    """
    return weight*(prior - state_tuple[0])

@njit
def prior_scalar_jacobian(state_tuple, weight=1.0):
    """Calculate the Jacobian for a scalar prior factor.

    Inputs:
    - state_tuple:
        - est: The estimated value.
    - weight: The weight for the Jacobian calculation.

    Returns:
    - The Jacobian matrix.

    """
    return -weight


@njit
def prior_r3_error(prior, state_tuple, weight=I3):
    """Calculate the error for a 3D vector prior factor.

    Inputs:
    - state_tuple:
        - est: The estimated 3D vector.
    - prior: The prior 3D vector.

    Returns:
    - The calculated error.

    """
    return weight @ (prior - state_tuple[0])

@njit
def prior_r3_jacobian(state_tuple, weight=I3):
    """Calculate the Jacobian for a 3D vector prior factor.

    Inputs:
    - state_tuple:
        - est: The estimated 3D vector.
    - weight: The weight for the Jacobian calculation.

    Returns:
    - The Jacobian matrix.

    """
    return (-weight)


@njit
def prior_so3_error(prior, state_tuple, weight=I3):
    """Calculate the error for a SO(3) prior factor.

    Inputs:
    - state_tuple:
        - est: The estimated SO(3) matrix.
    - prior: The prior SO(3) matrix.

    Returns:
    - The calculated error.

    """
    return weight @ box_minus_right(R1 = state_tuple[0], R2 = prior)

# @njit
def prior_so3_jacobian(state_tuple, prior, weight=I3):
    """Calculate the Jacobian for a SO(3) prior factor.

    Inputs:
    - state_tuple:
        - est: The estimated SO(3) matrix.
        - prior: The prior SO(3) matrix.
    - weight: The weight for the Jacobian calculation.

    Returns:
    - The Jacobian matrix.

    """
    d_error_d_R0 = lie_deriv_box_minus_right_R1(state_tuple[0], prior)
    return (weight) @ d_error_d_R0

