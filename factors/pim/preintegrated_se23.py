
from typing import Callable, List

from factors import Factor, RobustFactor
from util import so3, se23, se3
import util
import util.pim as pim
import numpy as np
from numba import njit
from dataclasses import dataclass
from functools import partial



@njit
def pim_se23_error(ups_hat: np.ndarray, state_tuple, GammaR, GammaV, GammaP, Omega, delta_t, weight: np.ndarray = np.eye(9)):
    """Calculate Upsilon using equations 86-88 from Associating Uncertainty to Extended Poses..."""
    Ri, vi, pi, Rj, vj, pj, ba, bg = state_tuple

    ups = se23._se23_from_components(
        (GammaR @ Ri).T @ Rj,
        Ri.T @ (GammaR.T @ (vj + Omega@pj - GammaV) - vi - Omega@pi),
        Ri.T @ (GammaR.T @ (pj- GammaP) - (vi + Omega@pi)*delta_t - pi)
    )

    return weight @ se23.diamond_minus_se23(ups, ups_hat)

@njit
def pim_se23_jacobians(ups_hat, state_tuple, GammaR, GammaV, GammaP, Omega, delta_t, bias_update_jacobian, weight:np.ndarray = np.eye(9)):
    """
    """
    Ri, vi, pi, Rj, vj, pj,ba,bg = state_tuple

    d_deltaR_d_Ri, d_deltaR_d_Rj = pim.deltaR_lie_jacobian_rotating_earth(Ri, Rj, GammaR)
    d_deltav_d_Ri, d_deltav_d_vi, d_deltav_d_vj, d_deltav_d_pi, d_deltav_d_pj = \
        pim.delta_v_lie_jacobian_rotating_earth(Ri, vi, vj, pi, pj, GammaR, GammaV, Omega)
    d_deltap_d_Ri, d_deltap_d_vi, d_deltap_d_pi, d_deltap_d_pj = \
        pim.delta_p_lie_jacobian_rotating_earth(Ri, vi, pi, pj, GammaR,  GammaP, Omega, delta_t)
    
    d_ups_d_update0 = np.zeros((9, 9), dtype=np.float64)
    d_ups_d_update0[:3, :3] = d_deltaR_d_Ri
    d_ups_d_update0[3:6, :3] = d_deltav_d_Ri
    d_ups_d_update0[6:9, :3] = d_deltap_d_Ri
    d_ups_d_update0[3:6, 3:6] = d_deltav_d_vi
    d_ups_d_update0[6:9, 3:6] = d_deltap_d_vi
    d_ups_d_update0[3:6, 6:9] = d_deltav_d_pi
    d_ups_d_update0[6:9, 6:9] = d_deltap_d_pi
    d_residual_d_T0_update = weight@d_ups_d_update0

    d_ups_d_update1 = np.zeros((9, 9), dtype=np.float64)
    d_ups_d_update1[:3, :3] = d_deltaR_d_Rj
    d_ups_d_update1[3:6, 3:6] = d_deltav_d_vj
    d_ups_d_update1[3:6, 6:9] = d_deltav_d_pj
    d_ups_d_update1[6:9, 6:9] = d_deltap_d_pj
    d_residual_d_T1_update = weight@d_ups_d_update1

    # d_residual_d_bias_update = weight@(-se23.adjoint_se23(se23.inv_se23(ups_hat)) @ se23.adjoint_se23(ups_hat) @ bias_update_jacobian)
    d_residual_d_bias_update = np.zeros((9,6), dtype=np.float64)
    d_residual_d_bias_update[:3,:] = (Ri.T@GammaR.T@Rj).T@(-ups_hat[:3,:3])@bias_update_jacobian[:3,:]
    d_residual_d_bias_update[3:6,:] = -bias_update_jacobian[3:6,:]
    d_residual_d_bias_update[6:9,:] = -bias_update_jacobian[6:9,:]
    d_residual_d_bias_update = weight @ d_residual_d_bias_update
    

    d_residual_d_Ri = d_residual_d_T0_update[:,:3]
    d_residual_d_vi = d_residual_d_T0_update[:,3:6]
    d_residual_d_pi = d_residual_d_T0_update[:,6:9]
    d_residual_d_Rj = d_residual_d_T1_update[:,:3]
    d_residual_d_vj = d_residual_d_T1_update[:,3:6]
    d_residual_d_pj = d_residual_d_T1_update[:,6:9]    
    d_residual_d_ba = d_residual_d_bias_update[:,3:]
    d_residual_d_bg = d_residual_d_bias_update[:,:3]
    
    return d_residual_d_Ri, d_residual_d_vi, d_residual_d_pi, d_residual_d_Rj, d_residual_d_vj, d_residual_d_pj, d_residual_d_ba, d_residual_d_bg


@dataclass
class IMUBiasValues:

    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    

@dataclass
class PIMFactor(Factor):
    """A factor representing a preintegrated measurement (PIM) in the factor graph.
    It extends the Factor class with additional attributes specific to PIMs.
    """

    ups_hat: np.ndarray
    bias_update_jacobian: np.ndarray
    GammaR: np.ndarray
    GammaV: np.ndarray
    GammaP: np.ndarray
    Omega: np.ndarray  # Skew-symmetric matrix of the angular velocity
    delta_t: float
    weight: np.ndarray
    bias_at_preintegration: IMUBiasValues
    reintegration_func: Callable[[np.ndarray,np.ndarray], pim.PIMData]
    reintegration_threshold: float = 16.0  # Threshold for determining when to reintegrate based on bias changes

    def __post_init__(self):
        
        self._precalculated_bias_information_matrix = self._compute_bias_information_matrix() # For checking for reintegration
        self.current_bias = self.bias_at_preintegration


    def _reintegrate(self, new_bias: IMUBiasValues) -> None:
        
        pim_data = self.reintegration_func(accel_bias = new_bias.accel_bias, gyro_bias = new_bias.gyro_bias)
        self.ups_hat = pim_data.UpsilonMsmt
        self.GammaR = pim_data.GammaR
        self.GammaV = pim_data.GammaV
        self.GammaP = pim_data.GammaP
        self.delta_t = pim_data.delta_t
        self.weight = util.weight_from_covariance(pim_data.error_covariance)
        self.bias_update_jacobian = np.hstack([pim_data.gyro_bias_jacobian, pim_data.accel_bias_jacobian])
        self.bias_at_preintegration = new_bias
        self.current_bias = IMUBiasValues(new_bias.accel_bias.copy(), new_bias.gyro_bias.copy())
        self._precalculated_bias_information_matrix = self._compute_bias_information_matrix()
        self.error_func = partial(pim_se23_error, ups_hat=self.ups_hat, GammaR = self.GammaR, GammaV = self.GammaV, GammaP = self.GammaP, Omega = self.Omega, delta_t = self.delta_t, weight = self.weight) 
        self.jacobian_func = partial(pim_se23_jacobians, ups_hat=self.ups_hat, GammaR = self.GammaR, GammaV = self.GammaV, GammaP = self.GammaP, Omega = self.Omega, delta_t = self.delta_t, bias_update_jacobian = self.bias_update_jacobian, weight = self.weight)    


    def update_bias(self, new_bias: IMUBiasValues) -> None:
        
        bias_delta = IMUBiasValues(new_bias.accel_bias - self.current_bias.accel_bias,
                                  new_bias.gyro_bias - self.current_bias.gyro_bias)
        # Check the bias deltas with the bias jacobians and compare to the error covariance to determine if reintegration is needed
        delta_bias_vector = np.hstack([bias_delta.gyro_bias, bias_delta.accel_bias])
        # delta_z = self.bias_update_jacobian @ delta_bias_vector
        # epsilon = delta_z.T @ (self.weight.T @ self.weight) @ delta_z
        epsilon = delta_bias_vector.T @ self._precalculated_bias_information_matrix @ delta_bias_vector
        if epsilon > self.reintegration_threshold:
            self._reintegrate(new_bias)

        else: 
            # Use the bias update jacobian to update the preintegrated measurement
            self.ups_hat = se23.diamond_plus_se23(
                self.ups_hat,
                self.bias_update_jacobian @ delta_bias_vector
            )
            self.current_bias = IMUBiasValues(new_bias.accel_bias.copy(), new_bias.gyro_bias.copy())
            self.error_func = partial(pim_se23_error, ups_hat=self.ups_hat, GammaR = self.GammaR, GammaV = self.GammaV, GammaP = self.GammaP, Omega = self.Omega, delta_t = self.delta_t, weight = self.weight)
            self.jacobian_func = partial(pim_se23_jacobians, ups_hat=self.ups_hat, GammaR = self.GammaR, GammaV = self.GammaV, GammaP = self.GammaP, Omega = self.Omega, delta_t = self.delta_t, bias_update_jacobian = self.bias_update_jacobian, weight = self.weight)


    def _compute_bias_information_matrix(self):
        
        return self.bias_update_jacobian.T @ self.weight.T @ self.weight @ self.bias_update_jacobian



@dataclass
class RobustPIMFactor(PIMFactor, RobustFactor):

    pass

def create_pim_se23_factor(
            time0: float,
            Ri_key: str,
            vi_key: str,
            pi_key: str,
            Rj_key: str,
            vj_key: str,
            pj_key: str,
            ba_key: str,
            bg_key: str,
            omega: np.ndarray,
            imu_msmts: List[pim.IMUMeasurement],
            accel_bias: np.ndarray,
            gyro_bias: np.ndarray,
            accel_white_noise: np.ndarray,
            gyro_white_noise: np.ndarray,
            gravity: np.ndarray) -> PIMFactor:
    
    bias_at_preintegration = IMUBiasValues(accel_bias, gyro_bias)
    reintegration_func: Callable[[np.ndarray,np.ndarray], pim.PIMData]= partial(pim.preintegrate_imu_one_interval,imu_measurements=imu_msmts,  accel_white_noise = accel_white_noise, gyro_white_noise = gyro_white_noise,omega = omega,gravity=gravity)
    initial_pim_data = reintegration_func(accel_bias = accel_bias, gyro_bias = gyro_bias)
    GammaR = initial_pim_data.GammaR
    GammaV = initial_pim_data.GammaV
    GammaP = initial_pim_data.GammaP
    Omega = so3.skew(omega)
    delta_t = initial_pim_data.delta_t
    weight = util.weight_from_covariance(initial_pim_data.error_covariance)
    bias_update_jacobian = np.hstack([initial_pim_data.gyro_bias_jacobian, initial_pim_data.accel_bias_jacobian])
    ups_hat = initial_pim_data.UpsilonMsmt


    return PIMFactor(
            time = time0,
            ups_hat = ups_hat,
            states = [Ri_key, vi_key, pi_key, Rj_key, vj_key, pj_key, ba_key, bg_key],
            n_rows = 9,
            jacobians_size = (9*3)*8, #9x3 for each state, 8 states
            error_func = partial(pim_se23_error, ups_hat=ups_hat, GammaR = GammaR, GammaV = GammaV, GammaP = GammaP, Omega = Omega, delta_t = delta_t, weight = weight),
            jacobian_func = partial(pim_se23_jacobians, ups_hat = ups_hat, GammaR = GammaR, GammaV = GammaV, GammaP = GammaP, Omega = Omega, delta_t = delta_t, bias_update_jacobian = bias_update_jacobian, weight = weight),
            bias_update_jacobian=bias_update_jacobian,
            GammaR = GammaR,
            GammaV = GammaV,
            GammaP = GammaP,
            Omega = Omega,
            delta_t=delta_t,
            weight = weight,
            reintegration_func = reintegration_func,
            bias_at_preintegration = bias_at_preintegration
        )

def create_robust_pim_se23_factor(
    time0: float,
    Ri_key: str,
    vi_key: str,
    pi_key: str,
    Rj_key: str,
    vj_key: str,
    pj_key: str,
    ba_key: str,
    bg_key: str,
    omega: np.ndarray,
    imu_msmts: List[pim.IMUMeasurement],
    accel_bias: np.ndarray,
    gyro_bias: np.ndarray,
    accel_white_noise: np.ndarray,
    gyro_white_noise: np.ndarray,
    gravity: np.ndarray,
    robust_weight_function)-> RobustPIMFactor:
    
    bias_at_preintegration = IMUBiasValues(accel_bias, gyro_bias)
    reintegration_func: Callable[[np.ndarray,np.ndarray], pim.PIMData]= partial(pim.preintegrate_imu_one_interval,imu_measurements=imu_msmts,  accel_white_noise = accel_white_noise, gyro_white_noise = gyro_white_noise,omega = omega,gravity=gravity)
    initial_pim_data = reintegration_func(accel_bias = accel_bias, gyro_bias = gyro_bias)
    GammaR = initial_pim_data.GammaR
    GammaV = initial_pim_data.GammaV
    GammaP = initial_pim_data.GammaP
    Omega = so3.skew(omega)
    delta_t = initial_pim_data.delta_t
    weight = util.weight_from_covariance(initial_pim_data.error_covariance)
    bias_update_jacobian = np.hstack([initial_pim_data.gyro_bias_jacobian, initial_pim_data.accel_bias_jacobian])
    ups_hat = initial_pim_data.UpsilonMsmt

    return RobustPIMFactor(
            time = time0,
            ups_hat = ups_hat,
            states = [Ri_key, vi_key, pi_key, Rj_key, vj_key, pj_key, ba_key, bg_key],
            n_rows = 9,
            jacobians_size = (9*3)*8, #9x3 for each state, 8 states
            error_func = partial(pim_se23_error, ups_hat=ups_hat, GammaR = GammaR, GammaV = GammaV, GammaP = GammaP, Omega = Omega, delta_t = delta_t, weight = weight),
            jacobian_func = partial(pim_se23_jacobians, ups_hat = ups_hat, GammaR = GammaR, GammaV = GammaV, GammaP = GammaP, Omega = Omega, delta_t = delta_t, bias_update_jacobian = bias_update_jacobian, weight = weight),
            bias_update_jacobian=bias_update_jacobian,
            GammaR = GammaR,
            GammaV = GammaV,
            GammaP = GammaP,
            Omega = Omega,
            delta_t=delta_t,
            weight = weight,
            bias_at_preintegration=bias_at_preintegration,
            reintegration_func = reintegration_func,
            robust_weight_func=robust_weight_function
        )

#### UNIT TESTS ###
def test_pim_se23_error_and_jacobian():
    
    # Setup synthetic test data
    Ri = np.eye(3)
    Rj = so3.expmap(np.random.randn(3) * 0.1)
    vi = np.random.randn(3)
    vj = np.random.randn(3)
    pi = np.random.randn(3)
    pj = np.random.randn(3)
    ba = np.random.randn(3)
    bg = np.random.randn(3)
    state_tuple = (Ri, vi, pi, Rj, vj, pj, ba, bg)

    GammaR = so3.expmap(np.random.randn(3) * 0.1)  # Random rotation perturbation
    GammaV = np.random.randn(3) * 0.1  # Random velocity perturbation
    GammaP = np.random.randn(3) * 0.1  # Random position perturbation
    Omega = so3.skew(np.random.randn(3) * 0.1)  # Random angular velocity skew matrix
    delta_t = 0.1
    weight = np.eye(9)
    R_hat = (GammaR @ Ri).T @ Rj  # Expected rotation after perturbation
    v_hat = Ri.T @ (GammaR.T @ (vj + Omega @ pj - GammaV) - vi - Omega @ pi)
    p_hat = Ri.T @ (GammaR.T @ (pj - GammaP) - (vi + Omega @ pi) * delta_t - pi)
    ups_hat = se23._se23_from_components(R_hat, v_hat, p_hat)  # Preintegrated measurement
    bias_update_jacobian = np.random.randn(9, 6)

    # Analytical Jacobians
    jac = pim_se23_jacobians(ups_hat, state_tuple, GammaR, GammaV, GammaP, Omega, delta_t, bias_update_jacobian, weight)
    analytic_jacobian = np.hstack(jac)[:,:18]  # Shape (9, 24)

    # Error function
    def error_wrapped(Ri, vi, pi, Rj, vj, pj, ba, bg):
        
        return pim_se23_error(ups_hat, (Ri, vi, pi, Rj, vj, pj, ba, bg),
                              GammaR, GammaV, GammaP, Omega, delta_t, weight)

    # Finite differencing setup
    eps = 1e-6
    numerical_jac = np.zeros((9, 18))

    def perturb_and_eval(index, direction):
        
        d = np.zeros(3)
        d[index] = eps * direction
        return d

    # Perturb Ri (right multiplication with Exp(ε))
    for i in range(3):
        d = perturb_and_eval(i, 1)
        Ri_plus = Ri @ so3.expmap(d)
        Ri_minus = Ri @ so3.expmap(-d)
        e_plus = error_wrapped(Ri_plus, vi, pi, Rj, vj, pj, ba, bg)
        e_minus = error_wrapped(Ri_minus, vi, pi, Rj, vj, pj, ba, bg)
        numerical_jac[:, i] = (e_plus - e_minus) / (2 * eps)

    # vi
    for i in range(3):
        d = perturb_and_eval(i, 1)
        e_plus = error_wrapped(Ri, vi + d, pi, Rj, vj, pj, ba, bg)
        e_minus = error_wrapped(Ri, vi - d, pi, Rj, vj, pj, ba, bg)
        numerical_jac[:, 3 + i] = (e_plus - e_minus) / (2 * eps)

    # pi
    for i in range(3):
        d = perturb_and_eval(i, 1)
        e_plus = error_wrapped(Ri, vi, pi + d, Rj, vj, pj, ba, bg)
        e_minus = error_wrapped(Ri, vi, pi - d, Rj, vj, pj, ba, bg)
        numerical_jac[:, 6 + i] = (e_plus - e_minus) / (2 * eps)

    # Perturb Rj (right multiplication with Exp(ε))
    for i in range(3):
        d = perturb_and_eval(i, 1)
        Rj_plus = Rj @ so3.expmap(d)
        Rj_minus = Rj @ so3.expmap(-d)
        e_plus = error_wrapped(Ri, vi, pi, Rj_plus, vj, pj, ba, bg)
        e_minus = error_wrapped(Ri, vi, pi, Rj_minus, vj, pj, ba, bg)
        numerical_jac[:, 9 + i] = (e_plus - e_minus) / (2 * eps)

    # vj
    for i in range(3):
        d = perturb_and_eval(i, 1)
        e_plus = error_wrapped(Ri, vi, pi, Rj, vj + d, pj, ba, bg)
        e_minus = error_wrapped(Ri, vi, pi, Rj, vj - d, pj, ba, bg)
        numerical_jac[:, 12 + i] = (e_plus - e_minus) / (2 * eps)

    # pj
    for i in range(3):
        d = perturb_and_eval(i, 1)
        e_plus = error_wrapped(Ri, vi, pi, Rj, vj, pj + d, ba, bg)
        e_minus = error_wrapped(Ri, vi, pi, Rj, vj, pj - d, ba, bg)
        numerical_jac[:, 15 + i] = (e_plus - e_minus) / (2 * eps)

    # Compare
    rel_error = np.linalg.norm(numerical_jac - analytic_jacobian) / np.linalg.norm(numerical_jac)
    print("Relative error in Jacobian:", rel_error)

    # Optional: print or visualize specific terms
    if rel_error > 1e-3:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(numerical_jac, aspect='auto', cmap='viridis')
        ax[0].set_title("Numerical Jacobian")
        ax[1].imshow(analytic_jacobian, aspect='auto', cmap='viridis')
        ax[1].set_title("Analytical Jacobian")
        plt.show()

    assert rel_error < 1e-3, "Jacobian mismatch is too high!"

# Run the test
if __name__ == "__main__":
    test_pim_se23_error_and_jacobian()
