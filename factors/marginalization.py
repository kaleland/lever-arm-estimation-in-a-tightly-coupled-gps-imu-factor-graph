from dataclasses import dataclass
from factors.factor import Factor
from util import so3

import numpy as np
from numba import njit

@dataclass
class MarginalizationFactor(Factor):
    pass

def marginalization_error(state_tuple, zm, G, x0):
    """
    Calculate the marginalization error.

    Inputs:
    - state_tuple: A tuple containing the state variables.
    - zm: The measurement vector.
    - G: The Jacobian matrix.
    - x0: A tuple containing the initial state variables.

    Returns:
    - The calculated marginalization error.

    """
    delta = np.zeros(G.shape[1])
    delta_idx = 0
    for state_idx in range(len(state_tuple)):
        if np.isscalar(state_tuple[state_idx]) or state_tuple[state_idx].ndim == 0:
            # Handle both Python scalars and 0-dimensional NumPy arrays
            delta[delta_idx] = state_tuple[state_idx] - x0[state_idx]
            delta_idx += 1

        elif state_tuple[state_idx].ndim == 1:
            delta[delta_idx:delta_idx + state_tuple[state_idx].shape[0]] = state_tuple[state_idx] - x0[state_idx]
            delta_idx += state_tuple[state_idx].shape[0]

        elif state_tuple[state_idx].ndim == 2: # Assume 2D state is SO(3) matrix
            if state_tuple[state_idx].shape[0] == 3 and state_tuple[state_idx].shape[1] == 3:
                delta[delta_idx:delta_idx + 3] = so3.box_minus_right(state_tuple[state_idx], x0[state_idx])
                # so3.box_minus_right returns a 3D vector representing the difference in SO(3) tangent space
                delta_idx += 3

            else:
                raise ValueError(f"Unsupported state dimension: {state_tuple[state_idx].shape}")
            
        else:
            raise ValueError(f"Unsupported state dimension: {state_tuple[state_idx].ndim}")
        
    return G @ delta + zm

def marginalization_jacobians(state_tuple, G, x0):
    """Calculate the Jacobians for the marginalization factor.

    Inputs:
    - state_tuple: A tuple containing the state variables.
    - G: The Jacobian matrix.

    Returns:
    - A list of Jacobians corresponding to each state variable in the state_tuple.

    """
    jacobians = []
    state_col_idx = 0
    for state_idx in range(len(state_tuple)):
        if np.isscalar(state_tuple[state_idx]) or (state_tuple[state_idx].ndim == 0):
            jacobians.append(G[:, state_col_idx])
            state_col_idx += 1

        elif state_tuple[state_idx].ndim == 1:
            jacobians.append(G[:, state_col_idx:state_col_idx + state_tuple[state_idx].shape[0]])
            state_col_idx += state_tuple[state_idx].shape[0]
            

        elif state_tuple[state_idx].ndim == 2: # Assume 2D state is SO(3) matrix
            if state_tuple[state_idx].shape[0] == 3 and state_tuple[state_idx].shape[1] == 3:
                # The Jacobian of box_minus_right(R1,R2) w.r.t. R1 is I3
                # jacobians.append(-G[:, state_col_idx:state_col_idx + 3])
                jac_R = np.zeros((G.shape[0], 3))
                d_LogA_d_A = so3.deriv_logmap(x0[state_idx].T @ state_tuple[state_idx])
                d_Expr_d_r = so3.deriv_expmap(np.zeros(3))
                d_A_d_r = np.zeros((3,3,3))
                for r_idx in range(3):
                    d_A_d_r[:,:,r_idx] = x0[state_idx].T @ state_tuple[state_idx] @ d_Expr_d_r[:,:,r_idx]

                d_LogA_d_r = np.zeros((3,3))
                for logA_idx in range(3):
                    for r_idx in range(3):
                        d_LogA_d_r[logA_idx, r_idx] = np.sum(d_LogA_d_A[logA_idx, :, :] * d_A_d_r[:, :, r_idx])

                jac_R = G[:, state_col_idx:state_col_idx + 3] @ d_LogA_d_r
                jacobians.append(jac_R)      
                state_col_idx += 3
            else:
                raise ValueError(f"Unsupported state dimension: {state_tuple[state_idx].shape}")
        
        else:
            raise ValueError(f"Unsupported state dimension: {state_tuple[state_idx].ndim}")

    return jacobians

def test_jacobians():
    np.random.seed(np.random.randint(0, 1000))

    dummy_states_keys = ['R1','a1','c1','g1','p1','v1']

    dummy_states = (
        so3.expmap(np.random.rand(3)),  # SO(3) state
        np.random.rand(3),  # Accelerometer bias
        np.random.rand(),  # Clock bias
        np.random.rand(3),  # Gyroscope bias
        np.random.rand(3),  # Position
        np.random.rand(3)   # Velocity
    )
    R1, a1, c1, g1, p1, v1 = dummy_states

    dummy_x0 = (
        so3.expmap(np.random.rand(3)),  # SO(3) state
        np.random.rand(3),  # Accelerometer bias
        np.random.rand(),  # Clock bias
        np.random.rand(3),  # Gyroscope bias
        np.random.rand(3),  # Position
        np.random.rand(3)   # Velocity
    )

    dummy_G = np.random.rand(16,16)  # Random Jacobian matrix
    dummy_zm = np.random.rand(16)  # Random measurement vector

    # Calculate the marginalization error
    error = marginalization_error(dummy_states, dummy_zm, dummy_G, dummy_x0)
    def error_wrapped(state_tuple):
        return marginalization_error(state_tuple, dummy_zm, dummy_G, dummy_x0
                                     )
    # Calculate the Jacobians
    analytical_jacobians = marginalization_jacobians(dummy_states, dummy_G, x0=dummy_x0)
    
    # Perturb each state and check the Jacobians
    eps = 1e-6
    
    def perturb_and_eval(index, direction):
        d = np.zeros(3)
        d[index] = eps * direction
        return d
    
    numerical_jacobians = []
    # Perturb R1
    numerical_jacobian_R1 = np.zeros((len(dummy_zm), 3))
    for i in range(3):
        d = perturb_and_eval(i, 1)
        R1_plus = R1 @ so3.expmap(d)
        R1_minus = R1 @ so3.expmap(-d)
        state_tuple_plus = (R1_plus, a1, c1, g1, p1, v1)
        state_tuple_minus = (R1_minus, a1, c1, g1, p1, v1)
        error_plus = error_wrapped(state_tuple_plus)
        error_minus = error_wrapped(state_tuple_minus)
        numerical_jacobian_R1[:, i] = (error_plus - error_minus) / (2 * eps)
    numerical_jacobians.append(numerical_jacobian_R1)

    # Perturb a1
    numerical_jacobian_a1 = np.zeros((len(dummy_zm), 3))
    for i in range(3):
        d = perturb_and_eval(i, 1)
        a1_plus = a1 + d
        a1_minus = a1 - d
        state_tuple_plus = (R1, a1_plus, c1, g1, p1, v1)
        state_tuple_minus = (R1, a1_minus, c1, g1, p1, v1)
        error_plus = error_wrapped(state_tuple_plus)
        error_minus = error_wrapped(state_tuple_minus)
        numerical_jacobian_a1[:, i] = (error_plus - error_minus) / (2 * eps)
    numerical_jacobians.append(numerical_jacobian_a1)

    # Perturb c1
    numerical_jacobian_c1 = np.zeros((len(dummy_zm),))
    c1_plus = c1 + eps
    c1_minus = c1 - eps
    state_tuple_plus = (R1, a1, c1_plus, g1, p1, v1)
    state_tuple_minus = (R1, a1, c1_minus, g1, p1, v1)
    error_plus = error_wrapped(state_tuple_plus)
    error_minus = error_wrapped(state_tuple_minus)
    numerical_jacobian_c1[:] = (error_plus - error_minus) / (2 * eps)
    numerical_jacobians.append(numerical_jacobian_c1)

    # Perturb g1
    numerical_jacobian_g1 = np.zeros((len(dummy_zm), 3))
    for i in range(3):
        d = perturb_and_eval(i, 1)
        g1_plus = g1 + d
        g1_minus = g1 - d
        state_tuple_plus = (R1, a1, c1, g1_plus, p1, v1)
        state_tuple_minus = (R1, a1, c1, g1_minus, p1, v1)
        error_plus = error_wrapped(state_tuple_plus)
        error_minus = error_wrapped(state_tuple_minus)
        numerical_jacobian_g1[:, i] = (error_plus - error_minus) / (2 * eps)
    numerical_jacobians.append(numerical_jacobian_g1)

    # Perturb p1
    numerical_jacobian_p1 = np.zeros((len(dummy_zm), 3))
    for i in range(3):
        d = perturb_and_eval(i, 1)
        p1_plus = p1 + d
        p1_minus = p1 - d
        state_tuple_plus = (R1, a1, c1, g1, p1_plus, v1)
        state_tuple_minus = (R1, a1, c1, g1, p1_minus, v1)
        error_plus = error_wrapped(state_tuple_plus)
        error_minus = error_wrapped(state_tuple_minus)
        numerical_jacobian_p1[:, i] = (error_plus - error_minus) / (2 * eps)
    numerical_jacobians.append(numerical_jacobian_p1)

    # Perturb v1
    numerical_jacobian_v1 = np.zeros((len(dummy_zm), 3))
    for i in range(3):
        d = perturb_and_eval(i, 1)
        v1_plus = v1 + d
        v1_minus = v1 - d
        state_tuple_plus = (R1, a1, c1, g1, p1, v1_plus)
        state_tuple_minus = (R1, a1, c1, g1, p1, v1_minus)
        error_plus = error_wrapped(state_tuple_plus)
        error_minus = error_wrapped(state_tuple_minus)
        numerical_jacobian_v1[:, i] = (error_plus - error_minus) / (2 * eps)
    numerical_jacobians.append(numerical_jacobian_v1)

    # Check if the analytical Jacobians match the numerical Jacobians
    for i, (analytical_jacobian, numerical_jacobian) in enumerate(zip(analytical_jacobians, numerical_jacobians)):
        if not np.allclose(analytical_jacobian, numerical_jacobian, rtol=1e-5):
            print(f"Jacobian mismatch at index {i}")
            print("Analytical Jacobian:")
            print(analytical_jacobian)
            print("Numerical Jacobian:")
            print(numerical_jacobian)

if __name__ == "__main__":
    test_jacobians()
