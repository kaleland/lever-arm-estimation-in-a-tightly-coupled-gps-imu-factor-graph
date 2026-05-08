
from factors import Factor, RobustFactor
from util import so3, gnss
import numpy as np
from numba import njit
from functools import partial
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class TDCPFactor(Factor):

        prn: int  # PRN of the satellite
        weight: float # Weight for the factor

@dataclass
class RobustTDCPFactor(TDCPFactor, RobustFactor):

       pass

@dataclass
class TDCPFactorUnknownLeverArm(TDCPFactor):

       pass

@dataclass
class RobustTDCPFactorUnknownLeverArm(RobustTDCPFactor):

       pass

@njit
def tdcp_error(meas: float, state_tuple, sat_xyz0, sat_xyz1, weight = 1.0, lb = np.zeros(3)):
        """Calculate the time-differenced code phase error.
        
        Inputs:
        - meas: The measured time-differenced code phase.
        - state_tuple:
            - p0: The position of the receiver at time t0.
            - p1: The position of the receiver at time t1.
            - cb0: The clock bias of the receiver at time t0.
            - cb1: The clock bias of the receiver at time t1.
            - R0: The rotation matrix from the body frame to the ECEF frame at time t0.
            - R1: The rotation matrix from the body frame to the ECEF frame at time t1.
        - sat_xyz0: The satellite position in ECEF coordinates at time t0.
        - sat_xyz1: The satellite position in ECEF coordinates at time t1.
        - weight: The weight for the error calculation.
        - lb: The lever arm vector from the IMU to the antenna in the body frame.

        
        Returns:
        - The calculated error.

        """
        p0, p1, cb0, cb1, R0, R1 = state_tuple
        return weight * (meas - (np.linalg.norm(sat_xyz1 - (R1@lb + p1)) + cb1) + (np.linalg.norm(sat_xyz0 - (R0@lb + p0)) + cb0))
        
@njit
def tdcp_error_unknown_lever_arm(meas: float, state_tuple, sat_xyz0, sat_xyz1, weight = 1.0):
        """Calculate the time-differenced code phase error with unknown lever arm.

        Inputs:
        - meas: The measured time-differenced code phase.
        - state_tuple:
                - p0: The position of the receiver at time t0.
                - p1: The position of the receiver at time t1.
                - cb0: The clock bias of the receiver at time t0.
                - cb1: The clock bias of the receiver at time t1.
                - R0: The rotation matrix from the body frame to the ECEF frame at time t0.
                - R1: The rotation matrix from the body frame to the ECEF frame at time t1.
                - lb: The lever arm vector from the IMU to the antenna in the body frame.
        - sat_xyz0: The satellite position in ECEF coordinates at time t0.
        - sat_xyz1: The satellite position in ECEF coordinates at time t1.

        
        Returns:
        - The calculated error.

        """
        p0, p1, cb0, cb1, R0, R1, lb = state_tuple
        return weight * (meas - (np.linalg.norm(sat_xyz1 - (R1@lb + p1)) + cb1) + (np.linalg.norm(sat_xyz0 - (R0@lb + p0)) + cb0))

@njit
def tdcp_jacobians(state_tuple, sat_xyz0, sat_xyz1, weight = 1.0, lb = np.zeros(3)):
        """Calculate the Jacobians for the time-differenced code phase factor.
        
        Inputs:
        - state_tuple:
            - p0: The position of the receiver at time t0.
            - p1: The position of the receiver at time t1.
            - cb0: The clock bias of the receiver at time t0.
            - cb1: The clock bias of the receiver at time t1.
            - R0: The rotation matrix from the body frame to the ECEF frame at time t0.
            - R1: The rotation matrix from the body frame to the ECEF frame at time t1.
        - sat_xyz0: The satellite position in ECEF coordinates at time t0.
        - sat_xyz1: The satellite position in ECEF coordinates at time t1.
        - weight: The weight for the error calculation.

        - lb: The lever arm vector from the IMU to the antenna in the body frame.
        
        Returns:
        - A tuple containing the Jacobians of the error with respect to positions, clock biases, attitudes, and lever arm.

        """
        p0, p1, cb0, cb1, R0, R1 = state_tuple
        r0 = sat_xyz0 - (R0 @ lb + p0)
        r1 = sat_xyz1 - (R1 @ lb + p1)
        norm_r0 = np.linalg.norm(r0)
        norm_r1 = np.linalg.norm(r1)        
        r_hat_0 = r0/norm_r0
        r_hat_1 = r1/norm_r1
        
        de_dp0 = -(r_hat_0)*weight
        de_dp1 = (r_hat_1)*weight
        de_dcb0 = 1.0*weight
        de_dcb1 = -1.0*weight
        de_dR0 = (r_hat_0 @ R0 @ so3.skew(lb))*weight
        de_dR1 = (-r_hat_1 @ R1 @ so3.skew(lb))*weight
        # de_dlb = (-(r_hat_1 @ R0) + (r_hat_0 @ R1))*weight

        return de_dp0, de_dp1, de_dcb0, de_dcb1, de_dR0, de_dR1#, de_dlb

@njit
def tdcp_jacobians_unknown_lever_arm(state_tuple, sat_xyz0, sat_xyz1, weight = 1.0):
        """Calculate the Jacobians for the time-differenced code phase factor with unknown lever arm.

        Inputs:
        - state_tuple:
                - p0: The position of the receiver at time t0.
                - p1: The position of the receiver at time t1.
                - cb0: The clock bias of the receiver at time t0.
                - cb1: The clock bias of the receiver at time t1.
                - R0: The rotation matrix from the body frame to the ECEF frame at time t0.
                - R1: The rotation matrix from the body frame to the ECEF frame at time t1.
                - lb: The lever arm vector from the IMU to the antenna in the body frame.
        - sat_xyz0: The satellite position in ECEF coordinates at time t0.
        - sat_xyz1: The satellite position in ECEF coordinates at time t1.
        - weight: The weight for the error calculation.

        
        Returns:
        - A tuple containing the Jacobians of the error with respect to positions, clock biases, attitudes, and lever arm.

        """
        p0, p1, cb0, cb1, R0, R1, lb = state_tuple
        r0 = sat_xyz0 - (R0 @ lb + p0)
        r1 = sat_xyz1 - (R1 @ lb + p1)
        norm_r0 = np.linalg.norm(r0)
        norm_r1 = np.linalg.norm(r1)        
        r_hat_0 = r0/norm_r0
        r_hat_1 = r1/norm_r1
        
        de_dp0 = -(r_hat_0)*weight
        de_dp1 = (r_hat_1)*weight
        de_dcb0 = 1.0*weight
        de_dcb1 = -1.0*weight
        de_dR0 = (r_hat_0 @ R0 @ so3.skew(lb))*weight
        de_dR1 = (-r_hat_1 @ R1 @ so3.skew(lb))*weight
        de_dlb = ((r_hat_1 @ R1) - (r_hat_0 @ R0))*weight

        return de_dp0, de_dp1, de_dcb0, de_dcb1, de_dR0, de_dR1, de_dlb

def create_tdcp_factor(
            meas,
            time0,
            time1,
            p0_key,
            p1_key,
            cb0_key,
            cb1_key,
            R0_key,
            R1_key,
            sat_xyz0,
            sat_xyz1,
            prn: int,
            weight = 1.0,
            lb = np.zeros(3)) -> TDCPFactor:
    
    return TDCPFactor(
            time=time1,
            states = [
                p0_key,
                p1_key,
                cb0_key,
                cb1_key,
                R0_key,
                R1_key
            ],
            n_rows = 1,  # One row for the TDCP error
            jacobians_size = 3*2+2+3*2,  # 3 elements for each position and rotation, 2 for each clock bias
            error_func = partial(tdcp_error, meas = meas, sat_xyz0=sat_xyz0, sat_xyz1=sat_xyz1, weight=weight, lb=lb),
            jacobian_func = partial(tdcp_jacobians, sat_xyz0=sat_xyz0, sat_xyz1=sat_xyz1, weight=weight, lb=lb),
            prn = prn,
            weight = weight

    )


def create_robust_tdcp_factor(
        meas,
        time0,
        time1,
        p0_key,
        p1_key,
        cb0_key,
        cb1_key,
        R0_key,
        R1_key,
        sat_xyz0,
        sat_xyz1,
        prn,
        robust_weight_function: Callable[..., Any],
        weight=1.0,
        lb = np.zeros(3)
    ) -> RobustTDCPFactor:
        
        return RobustTDCPFactor(
                time=time1,
                states = [
                    p0_key,
                    p1_key,
                    cb0_key,
                    cb1_key,
                    R0_key,
                    R1_key
                ],
                n_rows = 1,  # One row for the TDCP error
                jacobians_size = 3*2+2+3*2,  # 3 elements for each position and rotation, 2 for each clock bias
                error_func = partial(tdcp_error, meas = meas, sat_xyz0=sat_xyz0, sat_xyz1=sat_xyz1, weight=weight, lb=lb),
                jacobian_func = partial(tdcp_jacobians, sat_xyz0=sat_xyz0, sat_xyz1=sat_xyz1, weight=weight, lb=lb),
                prn = prn,
                weight = weight,
                robust_weight_func = robust_weight_function
        )

def create_tdcp_factor_unknown_lever_arm(
            meas,
            time0,
            time1,
            p0_key,
            p1_key,
            cb0_key,
            cb1_key,
            R0_key,
            R1_key,
            lb_key,
            sat_xyz0,
            sat_xyz1,
            prn: int,
            weight = 1.0) -> TDCPFactorUnknownLeverArm:
    
    return TDCPFactorUnknownLeverArm(
            time=time1,
            states = [
                p0_key,
                p1_key,
                cb0_key,
                cb1_key,
                R0_key,
                R1_key,
                lb_key
            ],
            n_rows = 1,  # One row for the TDCP error
            jacobians_size = 3*2+2+3*2 + 3,  # 3 elements for each position and rotation, 2 for each clock bias, 3 for lever arm
            error_func = partial(tdcp_error_unknown_lever_arm, meas = meas, sat_xyz0=sat_xyz0, sat_xyz1=sat_xyz1, weight=weight),
            jacobian_func = partial(tdcp_jacobians_unknown_lever_arm, sat_xyz0=sat_xyz0, sat_xyz1=sat_xyz1, weight=weight),
            prn = prn,
            weight = weight

    )

def create_robust_tdcp_factor_unknown_lever_arm(
        meas,
        time0,
        time1,
        p0_key,
        p1_key,
        cb0_key,
        cb1_key,
        R0_key,
        R1_key,
        lb_key,
        sat_xyz0,
        sat_xyz1,
        prn,
        robust_weight_function: Callable[..., Any],
        weight=1.0) -> RobustTDCPFactorUnknownLeverArm:
        
        return RobustTDCPFactorUnknownLeverArm(
                time=time1,
                states = [
                    p0_key,
                    p1_key,
                    cb0_key,
                    cb1_key,
                    R0_key,
                    R1_key,
                    lb_key
                ],
                n_rows = 1,  # One row for the TDCP error
                jacobians_size = 3*2+2+3*2 + 3,  # 3 elements for each position and rotation, 2 for each clock bias, 3 for lever arm
                error_func = partial(tdcp_error_unknown_lever_arm, meas = meas, sat_xyz0=sat_xyz0, sat_xyz1=sat_xyz1, weight=weight),
                jacobian_func = partial(tdcp_jacobians_unknown_lever_arm, sat_xyz0=sat_xyz0, sat_xyz1=sat_xyz1, weight=weight),
                prn = prn,
                weight = weight,
                robust_weight_func = robust_weight_function
        )

def error_state_tdcp_error(
        tdcp_deltar_diff: float,
        state_tuple: tuple,
        H_pnom: np.ndarray,
        weight: float = 1.0,
        lb: np.ndarray = np.zeros(3)
):
        
        p0, p1, cb0, cb1, R0, R1 = state_tuple
        p_g0 = p0 + R0@lb
        p_g1 = p1 + R1@lb
        return weight * (np.dot(H_pnom, (p_g1-p_g0)) + (cb1-cb0) - (tdcp_deltar_diff))

def error_state_tdcp_error_unknown_lever_arm(
        tdcp_deltar_diff: float,
        state_tuple: tuple,
        H_pnom: np.ndarray,
        weight: float = 1.0,
):
        
        p0, p1, cb0, cb1, R0, R1, lb = state_tuple
        p_g0 = p0 + R0@lb
        p_g1 = p1 + R1@lb
        return weight * (np.dot(H_pnom, (p_g1-p_g0)) + (cb1-cb0) - (tdcp_deltar_diff))

def error_state_tdcp_jacobians(
        state_tuple: tuple,
        H_pnom: np.ndarray,
        weight: float = 1.0,
        lb: np.ndarray = np.zeros(3)
):
        
        _, _, _, _, R0, R1 = state_tuple
        de_dp0 = -H_pnom*weight
        de_dp1 = H_pnom*weight
        de_dcb0 = -1.0*weight
        de_dcb1 = 1.0*weight
        de_dR0 = (H_pnom @ R0 @ so3.skew(lb)).flatten()*weight
        de_dR1 = -(H_pnom @ R1 @ so3.skew(lb)).flatten()*weight

        return de_dp0, de_dp1, de_dcb0, de_dcb1, de_dR0, de_dR1

def error_state_tdcp_jacobians_unknown_lever_arm(
        state_tuple: tuple,
        H_pnom: np.ndarray,
        weight: float = 1.0,
):
        
        _, _, _, _, R0, R1, lb = state_tuple
        de_dp0 = -H_pnom*weight
        de_dp1 = H_pnom*weight
        de_dcb0 = -1.0*weight
        de_dcb1 = 1.0*weight
        de_dR0 = (H_pnom @ R0 @ so3.skew(lb)).flatten()*weight
        de_dR1 = -(H_pnom @ R1 @ so3.skew(lb)).flatten()*weight
        de_dlb = (H_pnom @ (R1 - R0)).flatten()*weight

        return de_dp0, de_dp1, de_dcb0, de_dcb1, de_dR0, de_dR1, de_dlb

def create_error_state_tdcp_factor(
        tdcp_meas: float,
        time0: float,
        time1: float,
        pnom: np.ndarray,
        p0_key,
        p1_key,
        cb0_key,
        cb1_key,
        R0_key,
        R1_key,
        sat_xyz0: np.ndarray,
        sat_xyz1: np.ndarray,
        prn: int,
        weight: float = 1.0,
        lb: np.ndarray = np.zeros(3)
) -> TDCPFactor:
        
        # Assume satellite clock error is already removed from carrier phase measurements
        r_x0 = np.linalg.norm(sat_xyz0 - pnom)
        r_x1 = np.linalg.norm(sat_xyz1 - pnom) 
        deltar = r_x1 - r_x0
        H_pnom, _ = gnss.calc_H_values(sat_xyz0, pnom)
        tdcp_deltar_diff = tdcp_meas-deltar
        return TDCPFactor(
                time = time1,
                states = [
                        p0_key,
                        p1_key,
                        cb0_key,
                        cb1_key,
                        R0_key,
                        R1_key
                ],
                n_rows = 1,  # One row for the TDCP error
                jacobians_size = 3*2+2+3*2 + 3,  # 3 elements for each position and rotation, 2 for each clock bias, 3 for lever arm
                error_func = partial(error_state_tdcp_error, tdcp_deltar_diff=tdcp_deltar_diff, H_pnom=H_pnom, weight=weight, lb=lb),
                jacobian_func = partial(error_state_tdcp_jacobians, H_pnom=H_pnom, weight=weight, lb=lb),
                prn = prn,
                weight = weight
        )

def create_robust_error_state_tdcp_factor(
        tdcp_meas: float,
        time0: float,
        time1: float,
        pnom: np.ndarray,
        p0_key,
        p1_key,
        cb0_key,
        cb1_key,
        R0_key,
        R1_key,
        sat_xyz0: np.ndarray,
        sat_xyz1: np.ndarray,
        prn: int,
        robust_weight_function: Callable[..., Any],
        weight: float = 1.0,
        lb: np.ndarray = np.zeros(3)
) -> RobustTDCPFactor:
        
        r_x0 = np.linalg.norm(sat_xyz0 - pnom)
        r_x1 = np.linalg.norm(sat_xyz1 - pnom) 
        deltar = r_x1 - r_x0
        H_pnom, _ = gnss.calc_H_values(sat_xyz0, pnom)
        tdcp_deltar_diff = tdcp_meas-deltar
        return RobustTDCPFactor(
                time = time1,
                states = [
                        p0_key,
                        p1_key,
                        cb0_key,
                        cb1_key,
                        R0_key,
                        R1_key,
                ],
                n_rows = 1,  # One row for the TDCP error
                jacobians_size = 3*2+2+3*2,  # 3 elements for each position and rotation, 2 for each clock bias
                error_func = partial(error_state_tdcp_error, tdcp_deltar_diff=tdcp_deltar_diff, H_pnom=H_pnom, weight=weight, lb=lb),
                jacobian_func = partial(error_state_tdcp_jacobians, H_pnom=H_pnom, weight=weight, lb=lb),
                prn = prn,
                weight = weight,
                robust_weight_func = robust_weight_function
        )


def create_robust_error_state_tdcp_factor_unknown_lever_arm(
        tdcp_meas: float,
        time0: float,
        time1: float,
        pnom: np.ndarray,
        p0_key,
        p1_key,
        cb0_key,
        cb1_key,
        R0_key,
        R1_key,
        lb_key,
        sat_xyz0: np.ndarray,
        sat_xyz1: np.ndarray,
        prn: int,
        robust_weight_function: Callable[..., Any],
        weight: float = 1.0,
) -> RobustTDCPFactorUnknownLeverArm:
        
        r_x0 = np.linalg.norm(sat_xyz0 - pnom)
        r_x1 = np.linalg.norm(sat_xyz1 - pnom) 
        deltar = r_x1 - r_x0
        H_pnom, _ = gnss.calc_H_values(sat_xyz0, pnom)
        tdcp_deltar_diff = tdcp_meas-deltar
        
        return RobustTDCPFactorUnknownLeverArm(
                time = time1,
                states = [
                        p0_key,
                        p1_key,
                        cb0_key,
                        cb1_key,
                        R0_key,
                        R1_key,
                        lb_key
                ],
                n_rows = 1,  # One row for the TDCP error
                jacobians_size = 3*2+2+3*2 + 3,  # 3 elements for each position and rotation, 2 for each clock bias, 3 for lever arm
                error_func = partial(error_state_tdcp_error_unknown_lever_arm, tdcp_deltar_diff=tdcp_deltar_diff, H_pnom=H_pnom, weight=weight),
                jacobian_func = partial(error_state_tdcp_jacobians_unknown_lever_arm, H_pnom=H_pnom, weight=weight),
                prn = prn,
                weight = weight,
                robust_weight_func = robust_weight_function
        )

