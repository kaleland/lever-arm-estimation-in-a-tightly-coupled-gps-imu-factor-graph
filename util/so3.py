## version 1.0.2

import numpy as np
from numba import njit
###############                    CONSTANTS                     ###############

# SO3 generator matrices
G3 = np.zeros((3, 3, 3)) 
G3[0] = np.array([[0, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])
G3[1] = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [-1, 0, 0]])
G3[2] = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 0]])

I3 = np.eye(3) # 3x3 identity matrix

# Constant derivatives
dA3_dA3 = np.zeros((3,3,3,3)) # Derivative of a 3x3 matrix wrt itself
for i in range(3):
    for j in range(3):
        dA3_dA3[i,j,i,j] = 1

dA3T_dA3 = np.zeros((3,3,3,3)) # Derivative of the transpose of a 3x3 matrix wrt itself
for i in range(3):
    for j in range(3):
        dA3T_dA3[i,j,j,i] = 1

###############               SO3 & SE3  FUNCTIONS               ###############
@njit
def skew(v) -> np.ndarray:
    """Skew-symmetric matrix from a vector. Also called the "hat" or "wedge"
    operator.

    :param v: 

    """
    return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

@njit
def skew_squared(v) -> np.ndarray:
    """Square of the skew-symmetric matrix from a vector. This is the same as
    skew(v) @ skew(v).

    :param v: 

    """
    return np.array([
        [-(v[1]**2 + v[2]**2), v[0]*v[1], v[0]*v[2]],
        [v[0]*v[1], -(v[0]**2 + v[2]**2), v[1]*v[2]],
        [v[0]*v[2], v[1]*v[2], -(v[0]**2 + v[1]**2)]
    ])


@njit
def vee(S:np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix to vector. This function does not check if the input
    is a skew-symmetric matrix.

    :param S:np.ndarray: 

    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

# @njit
# def vee_safe(S) -> np.ndarray:
#     '''
#     Skew-symmetric matrix to vector.
#     '''
#     assert S[0,0] == 0 and S[1,1] == 0 and S[2,2] == 0, "Diagonal of skew-symmetric matrix must be zero"
#     assert S[1,0] == -S[0,1] and S[2,0] == -S[0,2] and S[2,1] == -S[1,2], "Skew-symmetric matrix must be antisymmetric"

#     return np.array([S[2, 1], S[0, 2], S[1, 0]])


@njit
def expmap(r) -> np.ndarray:
    """Exponential map from the Lie algebra so3 to the Lie group SO3.

    :param r: 

    """
    theta = np.linalg.norm(r)
    if theta == 0:
        return I3
    r_hat = skew(r)
    return I3 \
        + (np.sin(theta)/theta) * r_hat \
        + ((1 - np.cos(theta))/theta**2) * r_hat @ r_hat

@njit
def logmap(R) -> np.ndarray:
    """Logarithmic map from the Lie group SO3 to the Lie algebra so3. This is
    [log(R)]_v, where [.]_v is the vee operator.

    :param R: 

    """
    cos_theta = (np.trace(R) - 1) / 2
    
    theta = np.arccos(min((cos_theta,1.0)))
    if theta == 0:
        return np.zeros(3)
    # TODO - account for theta == pi

    sin_theta = np.sqrt(1 - cos_theta**2)
    # if np.any(np.isnan((theta/(2*sin_theta)) * np.array(
    #     [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]))):
    #     print('Nan here...')
    return (theta/(2*sin_theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

@njit
def so3_log(R) -> np.ndarray:
    """Matrix logarithm of an SO3 matrix.

    :param R: 

    """
    return skew(logmap(R))

@njit
def left_jacobian(phi):
    """:param phi:"""
    phi_norm = np.linalg.norm(phi)
    if phi_norm < 1e-10:
        return np.eye(3)
    return I3 + (1 - np.cos(phi_norm)) / (phi_norm**2) * skew(phi) + ((phi_norm - np.sin(phi_norm)) / phi_norm**3) * skew_squared(phi)

@njit
def inverse_left_jacobian(phi):
    """:param phi:"""
    phi_norm = np.linalg.norm(phi)
    if phi_norm < 1e-10:
        return np.eye(3)
    return I3 - 0.5*skew(phi) + (1/phi_norm**2 - (1+np.cos(phi_norm))/(2*phi_norm*np.sin(phi_norm)))*skew_squared(phi)

def right_jacobian(phi):
    """:param phi:"""
    phi_norm = np.linalg.norm(phi)
    if phi_norm < 1e-10:
        return np.eye(3)
    return I3 - (1 - np.cos(phi_norm)) / (phi_norm**2 )* skew(phi) + (phi_norm - np.sin(phi_norm)) / phi_norm**3 * skew_squared(phi)

def inverse_right_jacobian(phi):
    """:param phi:"""
    phi_norm = np.linalg.norm(phi)
    if phi_norm < 1e-10:
        return np.eye(3)
    return I3 + 0.5*skew(phi) + ( 1/phi_norm**2 - (1+np.cos(phi_norm))/(2*phi_norm*np.sin(phi_norm))) * skew_squared(phi)

@njit
def calc_V(r) -> np.ndarray:
    """Calculate the V matrix for the exponential map of an SO3 matrix.

    :param r: 

    """
    r_hat = skew(r)
    theta = np.linalg.norm(r)
    if theta == 0:
        return I3
    return I3 \
        + ((1 - np.cos(theta))/theta**2) * r_hat \
        + ((theta - np.sin(theta))/theta**3) * r_hat @ r_hat


@njit
def so3_generator_matrices() -> np.ndarray:
    
    """ """
    G3 = np.zeros((3, 3, 3))
    G3[0] = np.array([[0, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]])
    G3[1] = np.array([[0, 0, 1],
                        [0, 0, 0],
                        [-1, 0, 0]])
    G3[2] = np.array([[0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 0]])
    return G3

@njit
def box_minus_right(R1:np.ndarray , R2: np.ndarray) -> np.ndarray:
    """Box-minus operation for right multiplication of SO3 matrices.
    This is the inverse of the box-plus operation.

    :param R1:np.ndarray: 
    :param R2: np.ndarray: 

    """
    return logmap(R2.T @ R1)

def lie_deriv_box_minus_right_R1(R1:np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Lie derivative of the box-minus operation with respect to R1.
    This is the derivative of the box-minus operation with respect to R1.

    :param R1:np.ndarray: 
    :param R2: np.ndarray: 

    """
    return inverse_right_jacobian(box_minus_right(R1, R2))

def lie_deriv_box_minus_right_R2(R1:np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Lie derivative of the box-minus operation with respect to R2.
    This is the derivative of the box-minus operation with respect to R2.

    :param R1:np.ndarray: 
    :param R2: np.ndarray: 

    """
    return -inverse_right_jacobian(box_minus_right(R1, R2))@R1.T @ R2




###############                   DERIVATIVES                    ###############
@njit
def deriv_rHatSquared_d_r(r):
    """Derivative of the square of the skew-symmetric matrix from a vector.
    The output is a 3x3x3 tensor where the first two dimensions are the
    dimensions of the skew-symmetric matrix and the third dimension is the
    dimension of the vector.

    :param r: 

    """
    d_rHatSquared_d_r = np.zeros((3, 3, 3))
    d_rHatSquared_d_r[:,:,0] = np.array([
        [0,     r[1],       r[2]],
        [r[1],  -2*r[0],    0],
        [r[2],  0,          -2*r[0]]
    ])
    d_rHatSquared_d_r[:,:,1] = np.array([
        [-2*r[1], r[0],     0],
        [r[0],      0,      r[2]],
        [0,         r[2],   -2*r[1]]
    ])
    d_rHatSquared_d_r[:,:,2] = np.array([
        [-2*r[2], 0,      r[0]],
        [0,       -2*r[2], r[1]],
        [r[0],    r[1],   0]
    ])
    return d_rHatSquared_d_r

@njit
def deriv_expmap(r) -> np.ndarray:
    """Derivative of the exponential map from the Lie algebra so3 to the Lie group.
    The output derivative is a (3x3)x3 tensor where the first two dimensions are
    the dimensions of the rotation matrix and the third dimension is the dimension
    of the rotation vector.

    :param r: 

    """
    theta = np.linalg.norm(r)

    # If the rotation vector is zero, the derivative is the same as the
    # SO3 generator matrices. This is a common case with optimiztion algorithms
    # that use the pseudo-exponential map to update SE3 states.
    if theta == 0:
        # return np.moveaxis(G3,0,-1)
        return np.transpose(G3, (1, 2, 0))
    
    # Initialize the derivative tensor
    deriv = np.zeros((3, 3, 3))

    # Reused terms
    r_hat = skew(r)
    r_hat_squared = r_hat @ r_hat
    d_theta_d_r = r / theta
    sinThetaOverTheta = np.sin(theta) / theta
    oneMinusCosThetaOverThetaSquared = (1 - np.cos(theta)) / theta**2
    d_sinThetaOverTheta_d_r = ((theta*np.cos(theta) - np.sin(theta)) / theta**2)*d_theta_d_r
    d_oneMinusCosThetaOverThetaSquared_d_r = ((theta*np.sin(theta) - 2*(1 - np.cos(theta))) / theta**3) * d_theta_d_r
    d_rHatSquared_d_r = deriv_rHatSquared_d_r(r)

    # Compute the derivative using product rule. The derivative of skew(r) wrt 
    # ri is the ith SO3 generator matrix.
    for i in range(3):
        deriv[:,:,i] = d_sinThetaOverTheta_d_r[i]*r_hat + sinThetaOverTheta*G3[i]\
            + d_oneMinusCosThetaOverThetaSquared_d_r[i]*r_hat_squared\
            + oneMinusCosThetaOverThetaSquared*d_rHatSquared_d_r[:,:,i]
        
    return deriv

@njit
def deriv_logmap(R) -> np.ndarray:
    """Derivative of the logarithmic map from the Lie group SO3 to the Lie algebra.
    The output derivative is a 3x(3x3) tensor where the first dimension is the
    dimension of the rotation vector and the next two dimensions are the
    dimensions of the rotation matrix.

    :param R: 

    """
    # pass

    cosTheta = min(((np.trace(R) - 1) / 2,1.0))
    sinTheta = np.sqrt(1 - cosTheta**2)

    if cosTheta > 0.999999:
        return 0.5*G3

    theta = np.arccos(cosTheta)
    a = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])\
                     * ((theta*cosTheta - sinTheta) / (4*sinTheta**3)) 
    b = theta/(2*sinTheta)

    deriv = b*G3
    deriv[0] += I3*a[0]# np.diag((a[0],a[0],a[0]))
    deriv[1] += I3*a[1] # np.diag((a[1],a[1],a[1]))
    deriv[2] +=  I3*a[2] # np.diag((a[2],a[2],a[2]))

    return deriv

@njit
def deriv_expUpdate(R) -> np.ndarray:
    """Derivative of the exponential update of an SO3 matrix with respect to an
    infestimally small exponential update. The output is a 3x3x3 tensor where
    the first two dimensions are the dimensions of the rotation matrix and the
    third dimension is the dimension of the update vector. The update is via a
    right multiplication s.t. R_new = R @ expmap(v).

    :param R: 

    """
    deriv = np.zeros((3, 3, 3))
    deriv_expmap_r = deriv_expmap(np.zeros(3))
    for i in range(3):
        deriv[:,:,i] = R @ deriv_expmap_r[:,:,i]

    return deriv




if __name__ == '__main__':
    # test_deriv_pexpUpdateT()
    # test_deriv_calc_V_inv()
    # test_deriv_calc_V()
    # test_deriv_expmap_se3()
    # test_deriv_logmap_se3()
    # test_invT()
    # test_deriv_invT()

    pass

