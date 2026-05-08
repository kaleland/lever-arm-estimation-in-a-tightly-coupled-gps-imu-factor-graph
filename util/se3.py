## version 1.0.1
#%%

from util.so3 import *
import numpy as np
from numba import njit
###############                    CONSTANTS                     ###############
G6 = np.zeros((6, 4, 4)) # SE3 generator matrices
G6[:3,:3,3] = np.eye(3)
G6[3:,:3,:3] = G3

I4 = np.eye(4) # 4x4 identity matrix
#%%
# Constant derivatives
dT_dT = np.zeros((4,4,4,4)) # Derivative of a 4x4 transformation matrix wrt itself
for i in range(4):
    for j in range(4):
        dT_dT[i,j,i,j] = 1

dTT_dT = np.zeros((4,4,4,4)) # Derivative of the transpose of a 4x4 transformation matrix wrt itself
for i in range(4):
    for j in range(4):
        dTT_dT[i,j,j,i] = 1

@njit
def skew_se3(v) -> np.ndarray:
    """"Skew-symmetric" matrix from a 6-dimensional vector in se3.
    """
    return np.array([[0.0, -v[5], v[4], v[0]], [v[5], 0.0, -v[3], v[1]], [-v[4], v[3], 0.0, v[2]], [0.0, 0.0, 0.0, 0.0]]) 
@njit
def skew_se23(v) -> np.ndarray:
    """"Skew-symmetric" matrix from a 9-dimensional vector in se_2(3). Note
    that the convention is different in that the rotation vector comes first. 
    This follows the definition from "Associating Uncertainty to Extended Poses
    for on Lie Group IMU Preintegration with Rotating Earth" by Brossard et al.
    """
    return np.array([[0.0, -v[2], v[1], v[3], v[6]],[v[2], 0.0, -v[0], v[4], v[7]],[-v[1], v[0], 0.0, v[5], v[8]],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0]])

@njit
def vee_se3(T) -> np.ndarray:
    """SE3 "Skew-symmetric" matrix to 6-dimensional vector in se3."""
    return np.concatenate((T[:3, 3], vee(T[:3, :3])))

@njit
def calc_V_inv(r) -> np.ndarray:
    """Calculate the inverse of the V matrix for the logarithmic map of an se3 vector."""
    r_hat = skew(r)
    theta = np.linalg.norm(r)
    if theta == 0:
        return I3
    return I3 \
        - 0.5 * r_hat \
        + (1/theta**2) * (1 - theta/(2*np.tan(theta/2))) * r_hat @ r_hat

@njit
def expmap_se3(v) -> np.ndarray:
    """Exponential map from the Lie algebra se3 to the Lie group SE3. The first
    3 elements are the translation vector and the last 3 elements are the
    rotation vector.
    """
    t = v[:3]
    r = v[3:]

    R = expmap(r)
    T = np.copy(I4)
    T[:3, :3] = R

    V = calc_V(r)
    
    T[:3, 3] = V@t
    return T

@njit
def logmap_se3(T) -> np.ndarray:
    """Logarithmic map from the Lie group SE3 to the Lie algebra se3. The first
    3 elements are the translation vector and the last 3 elements are the
    rotation vector.
    """
    r = logmap(T[:3, :3])
    # r_hat = skew(r)
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.concatenate((T[:3, 3], r))
    V_inv = calc_V_inv(r)
    # V_inv = np.eye(3) \
    #     - 0.5 * r_hat \
    #     + (1/theta**2) * (1 - theta/(2*np.tan(theta/2))) * r_hat @ r_hat
    t = V_inv @ T[:3, 3]
    return np.concatenate((t, r))

@njit
def pexpmap_se3(v) -> np.ndarray:
    """Pseudo-exponential map from the Lie algebra se3 to the Lie group SE3. The
    first 3 elements are the translation vector and the last 3 elements are
    the rotation vector.
    """
    # The line below is concise, but causes issues with numba's jit.
    # return np.row_stack(
    #     [np.hstack([expmap(v[3:]), v[:3]]), np.array([0, 0, 0, 1])])
    # The line above is equivalent to the following:
    T = I4.copy()
    T[:3, :3] = expmap(v[3:])
    T[:3, 3] = v[:3]
    return T

@njit
def plogmap_se3(T) -> np.ndarray:
    """Pseudo-logarithmic map from the Lie group SE3 to the Lie algebra se3. The
    first 3 elements are the translation vector and the last 3 elements are
    the rotation vector.
    """
    return np.concatenate((T[:3, 3], logmap(T[:3, :3])))

@njit
def plog_se3(T) -> np.ndarray:
    """Matrix pseudo-logarithm of an SE3 matrix."""
    return skew_se3(plogmap_se3(T))

@njit
def inverse_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 transformation matrix."""
    T_inv = np.empty((4, 4), dtype=np.float64)  # Preallocate matrix for Numba compatibility

    R_inv = T[:3, :3].T
    t_inv = -R_inv @ T[:3, 3]

    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    T_inv[3, :] = np.array([0, 0, 0, 1], dtype=np.float64)  # Direct assignment for last row

    return T_inv
    # The code below is more precise, but causes issues with numba's jit.
    # return np.row_stack(
    #     [np.hstack((T[:3, :3].T, (-T[:3, :3].T @ T[:3,3]).reshape(-1, 1))), [0, 0, 0, 1]])


def se3_generator_matrices() -> np.ndarray:
    
    G6 = np.zeros((6, 4, 4))
    G6[:3,:3,3] = I3
    G6[3:,:3,:3] = so3_generator_matrices()

@njit
def transform_point(Tab, pb) -> np.ndarray:
    """Transform a 3D point pb from frame b to frame a using the transformation
    matrix Tab.
    """
    return Tab[:3, :3] @ pb + Tab[:3, 3]


###############                   DERIVATIVES                    ###############
@njit
def deriv_pexpmap_se3(v) -> np.ndarray:
    """Derivative of the pseudo-exponential map from the Lie algebra se3 to the Lie
    group SE3. The first 3 elements are the translation vector and the last 3
    elements are the rotation vector. The derivative is a 4x4x6 tensor where
    the first two dimensions are the dimensions of the transformation matrix
    and the third dimension is the dimension of the se3 vector.
    """
    deriv = np.zeros((4, 4, 6))
    deriv[:3, 3, :3] = I3
    deriv_expmap_r = deriv_expmap(v[3:])
    # for i in range(3,6):
    #     deriv[:3,:3,i] = deriv_expmap_r[:,:,i-3]
    deriv[:3, :3, 3:] = deriv_expmap_r
    return deriv    

@njit
def deriv_plogmap_se3(T) -> np.ndarray:
    """Derivative of the pseudo-logarithmic map from the Lie group SE3 to the Lie
    algebra se3. The output is a 6x(4x4) tensor where the first dimension is
    the dimension of the se3 vector and the next two dimensions are the
    dimensions of the transformation matrix. The first 3 elements of the se3 
    vector are the translation vector and the last 3 elements are the rotation
    vector.
    """
    deriv = np.zeros((6, 4, 4))
    deriv[:3, :3, 3] = I3
    deriv[3:, :3, :3] = deriv_logmap(T[:3, :3])
    return deriv

@njit
def deriv_plog_se3(T) -> np.ndarray:
    """Derivative of the pseudo-logarithm of an SE3 matrix. The output is a
    (4x4)x(4x4) tensor where the first two dimensions are the dimensions of the
    output 'skew-symmetric' matrix and the last two dimensions are the dimensions
    of the input SE3 matrix.
    """
    deriv = np.zeros((4, 4, 4, 4))
    d_plogmap_se3 = deriv_plogmap_se3(T)
    for row in range(4):
        for col in range(4):
            deriv[:,:,row,col] = skew_se3(d_plogmap_se3[:,row,col])
    return deriv

@njit
def deriv_invT(T) -> np.ndarray:
    """Derivative of the inverse of a transformation matrix with respect to itself.
    The output is a (4x4)x(4x4) tensor where the first two dimensions are the
    dimensions of the output matrix and the last two dimensions are the dimensions
    of the input matrix.
    """
    deriv = np.zeros((4, 4, 4, 4))
    invT = inverse_T(T)
    for k in range(4):
        for l in range(4):
            deriv[:,:,k,l] = -invT @ dT_dT[:,:,k,l] @ invT
    return deriv

@njit
def deriv_pexpUpdateT(T) -> np.ndarray:
    """Derivative of the pseudo-exponential update of a transformation matrix with
    respect to an infestimally small exponential update. The output is a 4x4x6 
    tensor where the first two dimensions are the dimensions of the 
    transformation matrix and the third dimension is the dimension of the update
    vector. The update is via a right multiplication s.t. T_new =  T @ pexpmap(v).
    v is a zero vector of length 6.
    """
    eps = np.zeros(6)
    deriv = np.zeros((4, 4, 6))
    deriv_expEps = deriv_pexpmap_se3(eps)
    for i in range(6):
        deriv[:,:,i] = T @ deriv_expEps[:,:,i]
    return deriv

@njit
def deriv_calc_V(r) -> np.ndarray:
    """Return the Jacobian tensor of the 3x3 matrix V with respect to the 3 element
    rotation vector r. The output is a 3x3x3 tensor where the first two dimensions
    are the dimensions of the V matrix and the third dimension is the dimension of
    the rotation vector.
    """
    deriv = np.zeros((3, 3, 3))
    theta = np.linalg.norm(r)
    if theta == 0:
        return 0.5*G3
    
    # Reused terms
    r_hat = skew(r)
    r_hat_squared = r_hat @ r_hat
    d_theta_d_r = r / theta
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)
    oneMinusCosThetaOverThetaSquared = (1 - np.cos(theta)) / theta**2
    d_oneMinusCosThetaOverThetaSquared_d_r = ( sinTheta/(theta**2) - 2*(1-cosTheta)/(theta**3) ) * d_theta_d_r
    thetaMinusSinThetaOverThetaCubed = (theta - np.sin(theta)) / theta**3
    d_thetaMinusSinThetaOverThetaCubed_d_r = (-(2*theta - 3*sinTheta + theta*cosTheta)/theta**4) * d_theta_d_r
    d_rHatSquared_d_r = deriv_rHatSquared_d_r(r)

    # The derivative of skew(r) wrt r is the ith SO3 generator matrix
    for i in range(3):
        deriv[:,:,i] = \
            d_oneMinusCosThetaOverThetaSquared_d_r[i]*r_hat \
            + oneMinusCosThetaOverThetaSquared*G3[i] \
            + d_thetaMinusSinThetaOverThetaCubed_d_r[i]*r_hat_squared \
            + thetaMinusSinThetaOverThetaCubed*d_rHatSquared_d_r[:,:,i]
        
    return deriv

@njit
def deriv_calc_V_inv(r) -> np.ndarray:
    """Return the Jacobian tensor of the inverse of the 3x3 matrix V with respect
    to the 3 element rotation vector r. The output is a 3x3x3 tensor where the
    first two dimensions are the dimensions of the V matrix and the third
    dimension is the dimension of the rotation vector.
    """
    deriv = np.zeros((3, 3, 3))
    theta = np.linalg.norm(r)
    if theta == 0:
        return -0.5*G3
    
    # Reused terms
    r_hat_squared = skew_squared(r)
    tanThetaOver2 = np.tan(theta/2)
    oneMinusThetaOver2TanThetaOver2OverThetaSquared = (1/theta**2) * (1 - theta/(2*tanThetaOver2))
    sinSquaredThetaOver2 = np.sin(theta/2)**2

    d_theta_d_r = r / theta
    d_oneMinusThetaOver2TanThetaOver2OverThetaSquared_d_r = \
        ((theta**2/(sinSquaredThetaOver2) + 2*theta/tanThetaOver2 - 8)/(4*theta**3)) * d_theta_d_r
    d_rHatSqaured_d_r = deriv_rHatSquared_d_r(r)

    # The derivative of skew(r) wrt r is the ith SO3 generator matrix
    for i in range(3):
        deriv[:,:,i] = \
            -0.5*G3[i] \
            + d_oneMinusThetaOver2TanThetaOver2OverThetaSquared_d_r[i]*r_hat_squared \
            + oneMinusThetaOver2TanThetaOver2OverThetaSquared*d_rHatSqaured_d_r[:,:,i]
        
    return deriv

@njit
def deriv_expmap_se3(v) -> np.ndarray:
    """Derivative of the exponential map from the Lie algebra se3 to the Lie group
    SE3. The first 3 elements are the translation vector and the last 3 elements
    are the rotation vector. The derivative is a 4x4x6 tensor where the first two
    dimensions are the dimensions of the transformation matrix and the third
    dimension is the dimension of the se3 vector.
    """
    if np.linalg.norm(v) == 0:
        # return np.moveaxis(G6,0,-1)
        return np.transpose(G6, (1, 2, 0))
    
    deriv = np.zeros((4, 4, 6))
    V = calc_V(v[3:])

    # Derivative of output translation wrt input translation
    deriv[:3,3,:3] = V # checked this

    # Derivative of output R matrix wrt rotation vector
    deriv[:3, :3, 3:] = deriv_expmap(v[3:])

    # Derivative of output translation wrt rotation vector
    d_V_d_r = deriv_calc_V(v[3:])
    for i in range(3):
        deriv[:3, 3, 3+i] = d_V_d_r[:,:,i]@v[:3]

    return deriv

# @njit . Does not work with einsum
def deriv_logmap_se3(T) -> np.ndarray:
    """Derivative of the logarithmic map from the Lie group SE3 to the Lie algebra
    se3. The output derivative is a 6x(4x4) tensor where the first dimension is
    the dimension of the se3 vector and the next two dimensions are the
    dimensions of the transformation matrix. The first 3 elements of the se3
    vector are the translation vector and the last 3 elements are the rotation
    vector.
    """
    r = logmap(T[:3, :3])

    deriv = np.zeros((6, 4, 4))
    dr_dR = deriv_logmap(T[:3, :3])

    # Derivative of the translation vector wrt translation vector in T
    V_inv = calc_V_inv(r)
    deriv[:3, :3, 3] = V_inv


    d_V_inv_d_r = deriv_calc_V_inv(r)

    # Now use einsum to compute the derivative of Vinv wrt R. einsum doesn't work with jit
    d_V_inv_d_R = np.einsum('ijk,klm->ijlm',d_V_inv_d_r,dr_dR)

    # print('einsum diff: ', np.sum(np.abs(old_Vinv_d_R - d_V_inv_d_R)))

    # Set the derivative of V_inv @ t wrt R
    for R_row in range(3):
        for R_col in range(3):
            deriv[:3,R_row,R_col] = d_V_inv_d_R[:,:,R_row,R_col]@T[:3,3]

    # Derivative of the rotation vector with respect to R
    deriv[3:, :3, :3] = dr_dR

    return deriv
    


################                 TEST FUNCTIONS                 ################
def test_deriv_pexpUpdateT():
    
    np.random.seed(2)
    T = pexpmap_se3(np.concatenate((
        np.random.random(3)*10,
        np.random.random(3)*0.1
    )))
    analytical_deriv = deriv_pexpUpdateT(T)
    update_magnitude = 1e-3
    exp_updates = np.eye(6)*update_magnitude
    print('Testing deriv_pexpUpdateT...')
    for eps_idx in range(6):
        print('Eps idx {:d}:'.format(eps_idx))
        eps = exp_updates[eps_idx]
        T_updated = T @ pexpmap_se3(eps)
        numerical_deriv = (T_updated - T) / update_magnitude

        diff = analytical_deriv[:,:,eps_idx] - numerical_deriv
        rel_diff = diff / np.abs(numerical_deriv)

        print('Numerical deriv:\n',numerical_deriv)
        print('Analytical deriv:\n',analytical_deriv[:,:,eps_idx])
        print('Diff:\n',diff)
        print('Rel diff:\n',rel_diff)

        print('\n\n')

def test_deriv_calc_V():
    
    print('Testing deriv_calc_V...')
    np.random.seed(2793)
    r = np.random.random(3)*0.2
    analytical_deriv = deriv_calc_V(r)
    update_magnitude = 1e-3
    for r_idx in range(3):
        r_plus = r.copy()
        r_plus[r_idx] += update_magnitude
        r_minus = r.copy()
        r_minus[r_idx] -= update_magnitude

        V_plus = calc_V(r_plus)
        V_minus = calc_V(r_minus)

        numerical_deriv = (V_plus - V_minus) / (2*update_magnitude)

        diff = analytical_deriv[:,:,r_idx] - numerical_deriv
        rel_diff = diff / np.abs(numerical_deriv)
        print('r_idx {:d}:'.format(r_idx))
        print('Numerical deriv:\n',numerical_deriv)
        print('Analytical deriv:\n',analytical_deriv[:,:,r_idx])
        print('Diff:\n',diff)
        print('Rel diff:\n',rel_diff)
        print('\n\n')

def test_deriv_calc_V_inv():
    
    print('Testing deriv_calc_V_inv...')
    np.random.seed(2793)
    r = np.random.random(3)*0.2
    analytical_deriv = deriv_calc_V_inv(r)
    update_magnitude = 1e-3
    for r_idx in range(3):
        r_plus = r.copy()
        r_plus[r_idx] += update_magnitude
        r_minus = r.copy()
        r_minus[r_idx] -= update_magnitude

        V_inv_plus = calc_V_inv(r_plus)
        V_inv_minus = calc_V_inv(r_minus)

        numerical_deriv = (V_inv_plus - V_inv_minus) / (2*update_magnitude)

        diff = analytical_deriv[:,:,r_idx] - numerical_deriv
        rel_diff = diff / np.abs(numerical_deriv)
        print('r_idx {:d}:'.format(r_idx))
        print('Numerical deriv:\n',numerical_deriv)
        print('Analytical deriv:\n',analytical_deriv[:,:,r_idx])
        print('Diff:\n',diff)
        print('Rel diff:\n',rel_diff)
        print('\n\n')

def test_deriv_expmap_se3():
    
    np.random.seed(62)
    v = np.random.random(6)
    analytical_deriv = deriv_expmap_se3(v)
    update_magnitude = 1e-4
    for v_idx in range(6):
        v_plus = v.copy()
        v_plus[v_idx] += update_magnitude
        v_minus = v.copy()
        v_minus[v_idx] -= update_magnitude

        T_plus = expmap_se3(v_plus)
        T_minus = expmap_se3(v_minus)

        numerical_deriv = (T_plus - T_minus) / (2*update_magnitude)

        diff = analytical_deriv[:,:,v_idx] - numerical_deriv
        rel_diff = diff / np.abs(numerical_deriv)
        print('v_idx {:d}:'.format(v_idx))
        print('Numerical deriv:\n',numerical_deriv)
        print('Analytical deriv:\n',analytical_deriv[:,:,v_idx])
        print('Diff:\n',diff)
        print('Rel diff:\n',rel_diff)
        print('\n\n')

def test_deriv_logmap_se3():
    
    np.random.seed(41)
    for vector_idx in range(6):
        T = expmap_se3(np.random.random(6)*0.5)
        old_logmap = logmap_se3(T)
        analytical_deriv = deriv_logmap_se3(T)
        update_magnitude = 1e-3
        update_vector = np.zeros(6)
        update_vector[vector_idx] = update_magnitude
        update = pexpmap_se3(update_vector)
        
        T_new = T @ update
        new_logmap = logmap_se3(T_new)
        Tchange = T_new - T
        expected_delta = np.zeros((6))
        for expected_delta_idx in range(6):
            expected_delta[expected_delta_idx] = np.sum(np.array([
                analytical_deriv[expected_delta_idx,:,:] * Tchange
            ]))
        print('Update vector idx {:d}:'.format(vector_idx))
        print('expected_delta:',expected_delta)
        print('new_logmap - old_logmap:',new_logmap - old_logmap)
        print('diff:',expected_delta - (new_logmap - old_logmap))
        print('rel diff:',(expected_delta - (new_logmap - old_logmap)) / (new_logmap - old_logmap))
        print('\n')
    
def test_invT():
    
    np.random.seed(1010)
    for i in range(5):
        T = expmap_se3(np.random.random(6)*0.5)
        T_inv = inverse_T(T)
        T_inv_T = T_inv @ T
        print('T_inv_T:\n',T_inv_T)
        print('T_inv_T - I:\n',T_inv_T - I4)
        print('\n')

def test_deriv_invT():
    
    np.random.seed(23)
    for vector_idx in range(6):
        T = expmap_se3(np.random.random(6)*0.5)
        old_inv = inverse_T(T)
        analytical_deriv = deriv_invT(T)
        update_magnitude = 1e-4
        update_vector = np.zeros(6)
        update_vector[vector_idx] = update_magnitude
        update = pexpmap_se3(update_vector)

        T_new = T @ update
        new_inv = inverse_T(T_new)
        Tchange = T_new - T
        expected_delta = np.zeros((4,4))
        for expected_delta_row in range(4):
            for expected_delta_col in range(4):
                expected_delta[expected_delta_row,expected_delta_col] = np.sum(np.array([
                    analytical_deriv[expected_delta_row,expected_delta_col,:,:] * Tchange
                ]))
        print('Update vector idx {:d}:'.format(vector_idx))
        print('expected_delta:',expected_delta)
        print('new_inv - old_inv:',new_inv - old_inv)
        print('diff:',expected_delta - (new_inv - old_inv))
        print('rel diff:',(expected_delta - (new_inv - old_inv)) / (new_inv - old_inv))
        print('\n')

if __name__ == '__main__':
    # test_deriv_pexpUpdateT()
    # test_deriv_calc_V_inv()
    # test_deriv_calc_V()
    # test_deriv_expmap_se3()
    # test_deriv_logmap_se3()
    # test_invT()
    # test_deriv_invT()

    pass

