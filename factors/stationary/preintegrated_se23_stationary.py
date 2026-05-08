import factors.pim.preintegrated_se23 as preintegrated_se23
from factors.pim.preintegrated_se23 import IMUBiasValues
from functools import partial
import numpy as np
from typing import Callable, Any, List
from dataclasses import dataclass
from util import pim, so3
import util


def state_tuple_to_expanded_preintegrated_se23(state_tuple):
    
    R, p, ba, bg = state_tuple
    stationary_v = np.zeros(3)
    return (R.copy(), stationary_v.copy(), p.copy(),R.copy(), stationary_v.copy(), p.copy(), ba.copy(), bg.copy())

def pim_se23_stationary_error(state_tuple, *args, **kwargs):
    
    expanded_state_tuple = state_tuple_to_expanded_preintegrated_se23(state_tuple)
    return preintegrated_se23.pim_se23_error(*args, state_tuple = expanded_state_tuple, **kwargs)

def pim_se23_stationary_jacobians(state_tuple, *args, **kwargs):
    
    expanded_state_tuple = state_tuple_to_expanded_preintegrated_se23(state_tuple)

    d_residual_d_Ri, _, d_residual_d_pi, d_residual_d_Rj, _, d_residual_d_pj, d_residual_d_ba, d_residual_d_bg = preintegrated_se23.pim_se23_jacobians(*args, state_tuple = expanded_state_tuple, **kwargs)
    d_residual_d_R = d_residual_d_Ri + d_residual_d_Rj
    d_residual_d_p = d_residual_d_pi + d_residual_d_pj
    return d_residual_d_R, d_residual_d_p, d_residual_d_ba, d_residual_d_bg

@dataclass
class PIMFactorStationary(preintegrated_se23.PIMFactor):
    """A PIMFactor adapted for stationary periods where velocity is zero.
    Inherits from preintegrated_se23.PIMFactor.
    """

    def update_bias(self, new_bias: IMUBiasValues) -> None:
        
        super().update_bias(new_bias)
        self.error_func = partial(pim_se23_stationary_error, ups_hat=self.ups_hat, GammaR = self.GammaR, GammaV = self.GammaV, GammaP = self.GammaP, Omega = self.Omega, delta_t = self.delta_t, weight = self.weight)
        self.jacobian_func = partial(pim_se23_stationary_jacobians, ups_hat=self.ups_hat, GammaR = self.GammaR, GammaV = self.GammaV, GammaP = self.GammaP, Omega = self.Omega, delta_t = self.delta_t, bias_update_jacobian = self.bias_update_jacobian, weight = self.weight)

    def _reintegrate(self, new_bias):
        
        super()._reintegrate(new_bias)
        self.error_func = partial(pim_se23_stationary_error, ups_hat=self.ups_hat, GammaR = self.GammaR, GammaV = self.GammaV, GammaP = self.GammaP, Omega = self.Omega, delta_t = self.delta_t, weight = self.weight)
        self.jacobian_func = partial(pim_se23_stationary_jacobians, ups_hat=self.ups_hat, GammaR = self.GammaR, GammaV = self.GammaV, GammaP = self.GammaP, Omega = self.Omega, delta_t = self.delta_t, bias_update_jacobian = self.bias_update_jacobian, weight = self.weight)

@dataclass
class RobustPIMFactorStationary(preintegrated_se23.RobustPIMFactor, PIMFactorStationary):
    """A RobustPIMFactor adapted for stationary periods where velocity is zero.
    Inherits from preintegrated_se23.RobustPIMFactor.
    """

    def update_bias(self, new_bias):
        
        PIMFactorStationary.update_bias(self, new_bias)

    def _reintegrate(self, new_bias):
        
        PIMFactorStationary._reintegrate(self, new_bias)


def create_pim_se23_stationary_factor(
        time0: float,
        R_key: str,
        p_key: str,
        ba_key: str,
        bg_key: str,
        omega: np.ndarray,
        imu_msmts: List[pim.IMUMeasurement],
        accel_bias: np.ndarray,
        gyro_bias: np.ndarray,
        accel_white_noise: np.ndarray,
        gyro_white_noise: np.ndarray,
        gravity: np.ndarray,
    ) -> PIMFactorStationary:
    
    bias_at_preintegration = IMUBiasValues(accel_bias, gyro_bias)
    reintegration_func: Callable[[np.ndarray,np.ndarray], pim.PIMData]= partial(pim.preintegrate_imu_one_interval,imu_measurements=imu_msmts,  accel_white_noise = accel_white_noise, gyro_white_noise = gyro_white_noise,omega = omega,gravity=gravity)
    initial_pim_data = reintegration_func(accel_bias = accel_bias, gyro_bias = gyro_bias)
    GammaR = initial_pim_data.GammaR
    GammaV = initial_pim_data.GammaV
    GammaP = initial_pim_data.GammaP
    Omega = so3.skew(omega)
    ups_hat = initial_pim_data.UpsilonMsmt
    delta_t = initial_pim_data.delta_t
    weight = util.weight_from_covariance(initial_pim_data.error_covariance)
    bias_update_jacobian = np.hstack([initial_pim_data.gyro_bias_jacobian, initial_pim_data.accel_bias_jacobian])



    return PIMFactorStationary(
        time = time0,
        states = [
            R_key,
            p_key,
            ba_key,
            bg_key
        ],
        n_rows = 9,
        jacobians_size=(9*3)*4, # 9x3 for each state, 4 states
        error_func = partial(pim_se23_stationary_error, ups_hat=ups_hat, GammaR = GammaR, GammaV = GammaV, GammaP = GammaP, Omega = Omega, delta_t = delta_t, weight = weight),
        jacobian_func = partial(pim_se23_stationary_jacobians, ups_hat = ups_hat, GammaR = GammaR, GammaV = GammaV, GammaP = GammaP, Omega = Omega, delta_t = delta_t, bias_update_jacobian = bias_update_jacobian, weight = weight),
        bias_update_jacobian=bias_update_jacobian,
        GammaR = GammaR,
        GammaV = GammaV,
        GammaP = GammaP,
        Omega = Omega,
        delta_t=delta_t,
        weight = weight,
        reintegration_func = reintegration_func,
        bias_at_preintegration = bias_at_preintegration,
        ups_hat=ups_hat
    )

def create_robust_pim_se23_factor(
    time0: float,
    R_key: str,
    p_key: str,
    ba_key: str,
    bg_key: str,
    omega: np.ndarray,
    imu_msmts: List[pim.IMUMeasurement],
    accel_bias: np.ndarray,
    gyro_bias: np.ndarray,
    accel_white_noise: np.ndarray,
    gyro_white_noise: np.ndarray,
    gravity: np.ndarray,
    robust_weight_function)-> RobustPIMFactorStationary:
    
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
    dt = initial_pim_data.delta_t

    return RobustPIMFactorStationary(
        time = time0,
        states = [R_key, p_key, ba_key, bg_key],
        n_rows = 9,
        jacobians_size = (9*3)*4, #9x3 for each state, 4 states
        error_func = partial(pim_se23_stationary_error, ups_hat=ups_hat, GammaR = GammaR, GammaV = GammaV, GammaP = GammaP, Omega = Omega, delta_t = delta_t, weight = weight),
        jacobian_func = partial(pim_se23_stationary_jacobians, ups_hat = ups_hat, GammaR = GammaR, GammaV = GammaV, GammaP = GammaP, Omega = Omega, delta_t = delta_t, bias_update_jacobian = bias_update_jacobian, weight = weight),
        bias_update_jacobian=bias_update_jacobian,
        GammaR = GammaR,
        GammaV = GammaV,
        GammaP = GammaP,
        Omega = Omega,
        delta_t=delta_t,
        weight = weight,
        robust_weight_func=robust_weight_function,
        reintegration_func = reintegration_func,
        bias_at_preintegration = bias_at_preintegration,
        ups_hat=ups_hat
    )
