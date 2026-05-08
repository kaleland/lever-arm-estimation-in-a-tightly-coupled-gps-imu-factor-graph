from factors import Factor, RobustFactor
from util import so3, gnss
import numpy as np
from numba import njit
from functools import partial
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class PseudorangeFactor(Factor):

        prn: int  # PRN of the satellite
        weight: float # Weight for the factor

@dataclass
class RobustPseudorangeFactor(PseudorangeFactor, RobustFactor):

        pass

@dataclass
class PseudorangeFactorUnknownLeverArm(PseudorangeFactor):

       pass

@dataclass
class RobustPseudorangeFactorUnknownLeverArm(RobustPseudorangeFactor):

       pass

@njit
def pseudorange_error(meas: float, state_tuple, sat_xyz, weight = 1.0, lb = np.zeros(3)):
        """Calculate the pseudorange error.
        
        Inputs:
        - meas: The measured pseudorange.
        - state_tuple:
            - p: The position of the receiver.
            - cb: The clock bias of the receiver.
            - R: The rotation matrix from the body frame to the ECEF frame.
        - sat_xyz: The satellite position in ECEF coordinates.
        - weight: The weight for the error calculation.
        - lb: The lever arm vector from the receiver to the antenna in the body frame.

        
        Returns:
        - The calculated error.

        """
        p, cb, R = state_tuple
        return weight * (meas - (np.linalg.norm(sat_xyz - (R@lb + p)) + cb))

@njit
def pseudorange_error_unknown_lever_arm(meas: float, state_tuple, sat_xyz, weight = 1.0):
        """Calculate the pseudorange error with unknown lever arm.

        Inputs:
        - meas: The measured pseudorange.
        - state_tuple:
                - p: The position of the receiver.
                - cb: The clock bias of the receiver.
                - R: The rotation matrix from the body frame to the ECEF frame.
                - lb: The lever arm vector from the receiver to the antenna in the body frame.
        - sat_xyz: The satellite position in ECEF coordinates.
        - weight: The weight for the error calculation.


        Returns:
        - The calculated error.

        """
        p, cb, R, lb = state_tuple
        return weight * (meas - (np.linalg.norm(sat_xyz - (R@lb + p)) + cb))

@njit
def pseudorange_jacobians(state_tuple, sat_xyz, weight = 1.0, lb = np.zeros(3)):
        """Calculate the Jacobians for the pseudorange factor.
        
        Inputs:
        - state_tuple:
            - p: The position of the receiver.
            - cb: The clock bias of the receiver.
            - R: The rotation matrix from the body frame to the ECEF frame.   
        - sat_xyz: The satellite position in ECEF coordinates.
        - lb: The lever arm vector from the receiver to the antenna in the body frame.
        
        Returns:
        - A tuple containing the Jacobians of the errorwith respect to position clock bias, attitude, and lever arm.

        """
        # Unpack the state tuple
        p, _, R = state_tuple
        # Calculate the residual
        r = sat_xyz - (R @ lb + p)
        norm_r = np.linalg.norm(r)
        r_hat = r/norm_r
        de_dp = (r_hat)*weight
        de_dcb = (-1.0)*weight
        de_dR = (- (r_hat) @ R @ so3.skew(lb))*weight
        #de_dlb = ((r_hat) @ R)*weight
        return de_dp, de_dcb, de_dR#, de_dlb

@njit
def pseudorange_jacobians_unknown_lever_arm(state_tuple, sat_xyz, weight = 1.0):
        """Calculate the Jacobians for the pseudorange factor with unknown lever arm.

        Inputs:
        - state_tuple:
                - p: The position of the receiver.
                - cb: The clock bias of the receiver.
                - R: The rotation matrix from the body frame to the ECEF frame.
                - lb: The lever arm vector from the receiver to the antenna in the body frame.
        - sat_xyz: The satellite position in ECEF coordinates.
        
        Returns:
        - A tuple containing the Jacobians of the error with respect to position, clock bias, attitude, and lever arm.

        """
        # Unpack the state tuple
        p, _, R, lb = state_tuple
        # Calculate the residual
        r = sat_xyz - (R @ lb + p)
        norm_r = np.linalg.norm(r)
        r_hat = r/norm_r
        de_dp = (r_hat)*weight
        de_dcb = (-1.0)*weight
        de_dR = (- (r_hat) @ R @ so3.skew(lb))*weight
        de_dlb = (r_hat @ R)*weight
        return de_dp, de_dcb, de_dR, de_dlb

def create_pseudorange_factor(
        meas,
        time,
        p_key,
        cb_key,
        R_key,
        sat_xyz,
        prn: int,
        lb = np.zeros(3),
        weight = 1.0) -> PseudorangeFactor:
    return PseudorangeFactor(
            time = time, 
            states = [p_key, cb_key, R_key],
            n_rows = 1,  # One row for the pseudorange error
            jacobians_size = 3 + 1 + 3,  # 3 for position, 1 for clock bias, 3 for rotation matrix
            error_func = partial(pseudorange_error, meas = meas, sat_xyz=sat_xyz, weight=weight, lb=lb),
            jacobian_func = partial(pseudorange_jacobians, sat_xyz=sat_xyz, weight=weight, lb=lb),
            prn = prn,
            weight = weight          
    )

def create_robust_pseudorange_factor(
        meas,
        time,
        p_key,
        cb_key,
        R_key,
        sat_xyz,
        prn: int,
        robust_weight_function: Callable[..., Any],
        lb = np.zeros(3),
        weight = 1.0
) -> RobustPseudorangeFactor:
        return RobustPseudorangeFactor(
                time=time,  
                states=[p_key, cb_key, R_key],
                n_rows=1,  # One row for the pseudorange error
                jacobians_size=3 + 1 + 3,  # 3 for position, 1 for clock bias, 3 for rotation matrix
                error_func = partial(pseudorange_error, meas = meas, sat_xyz=sat_xyz, weight=weight, lb=lb),
                jacobian_func = partial(pseudorange_jacobians, sat_xyz=sat_xyz, weight=weight, lb=lb),
                prn=prn,
                weight=weight,
                robust_weight_func=robust_weight_function,
        )


def create_pseudorange_factor_unknown_lever_arm(
        meas,
        time,
        p_key,
        cb_key,
        R_key,
        lb_key,
        sat_xyz,
        prn: int,
        weight = 1.0) -> PseudorangeFactorUnknownLeverArm:
    return PseudorangeFactorUnknownLeverArm(
            time = time, 
            states = [p_key, cb_key, R_key, lb_key],
            n_rows = 1,  # One row for the pseudorange error
            jacobians_size = 3 + 1 + 3 + 3,  # 3 for position, 1 for clock bias, 3 for rotation matrix, 3 for lever arm
            error_func = partial(pseudorange_error_unknown_lever_arm, meas = meas, sat_xyz=sat_xyz, weight=weight),
            jacobian_func = partial(pseudorange_jacobians_unknown_lever_arm, sat_xyz=sat_xyz, weight=weight),
            prn = prn,
            weight = weight          
    )

def create_robust_pseudorange_factor_unknown_lever_arm(
        meas,
        time,
        p_key,
        cb_key,
        R_key,
        lb_key,
        sat_xyz,
        prn: int,
        robust_weight_function: Callable[..., Any],
        weight = 1.0
) -> RobustPseudorangeFactorUnknownLeverArm:
    return RobustPseudorangeFactorUnknownLeverArm(
            time = time,
            states = [p_key, cb_key, R_key, lb_key],
            n_rows = 1,  # One row for the pseudorange error
            jacobians_size = 3 + 1 + 3 + 3,  # 3 for position, 1 for clock bias, 3 for rotation matrix, 3 for lever arm
            error_func = partial(pseudorange_error_unknown_lever_arm, meas = meas, sat_xyz=sat_xyz, weight=weight),
            jacobian_func = partial(pseudorange_jacobians_unknown_lever_arm, sat_xyz=sat_xyz, weight=weight),
            prn = prn,
            weight = weight,
            robust_weight_func = robust_weight_function
    )


def error_state_pseudorange_error(
        pseudorange_range_diff: float,
        state_tuple: tuple[np.ndarray, float, np.ndarray],
        H_pnom: np.ndarray,
        pnom: np.ndarray,
        weight: float = 1.0,
        lb: np.ndarray = np.zeros(3)
) -> float:
       p, cb, R = state_tuple
#        delta_p = (p+R@lb) - pnom
       return weight*( np.dot(H_pnom, (p+R@lb) - pnom) + cb - pseudorange_range_diff)

def error_state_pseudorange_error_unknown_lever_arm(
        pseudorange_range_diff: float,
        state_tuple: tuple[np.ndarray, float, np.ndarray, np.ndarray],
        H_pnom: np.ndarray,
        pnom: np.ndarray,
        weight: float = 1.0
) -> float:
       p, cb, R, lb = state_tuple
       return weight*( np.dot(H_pnom, (p+R@lb) - pnom) + cb - pseudorange_range_diff)

def error_state_pseudorange_jacobians(
        state_tuple: tuple[np.ndarray, float, np.ndarray],
        H_pnom: np.ndarray,
        weight: float = 1.0,
        lb: np.ndarray = np.zeros(3)
) -> tuple[np.ndarray, float, np.ndarray]:
       _, _, R = state_tuple
       de_dp = H_pnom*weight
       de_dcb = weight
       de_dR = -( H_pnom.reshape(1,3) @ R @ so3.skew(lb)).flatten()*weight
       return de_dp, de_dcb, de_dR

def error_state_pseudorange_jacobians_unknown_lever_arm(
        state_tuple: tuple[np.ndarray, float, np.ndarray, np.ndarray],
        H_pnom: np.ndarray,
        weight: float = 1.0
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
       _, _, R, lb = state_tuple
       de_dp = H_pnom*weight
       de_dcb = weight
       de_dR = -( H_pnom.reshape(1,3) @ R @ so3.skew(lb)).flatten()*weight
       de_dlb = (H_pnom.reshape(1,3) @ R).flatten()*weight
       return de_dp, de_dcb, de_dR, de_dlb

def create_error_state_pseudorange_factor(
        pseudorange_meas: float,
        time: float,
        pnom: np.ndarray,
        p_key,
        cb_key,
        R_key,
        sat_xyz: np.ndarray,
        prn: int,
        lb: np.ndarray = np.zeros(3),
        weight: float = 1.0
) -> PseudorangeFactor:
        # Calculate the expected pseudorange at the linearization point
        r_x0 = np.linalg.norm(sat_xyz - pnom)
        H_pnom, _ = gnss.calc_H_values(sat_xyz, pnom)
        pseudorange_range_diff = pseudorange_meas-r_x0

        return PseudorangeFactor(
                time = time,
                states = [p_key, cb_key, R_key],
                n_rows = 1,  # One row for the pseudorange error
                jacobians_size = 3 + 1 + 3,  # 3 for position, 1 for clock bias, 3 for rotation matrix
                error_func = partial(error_state_pseudorange_error, pseudorange_range_diff = pseudorange_range_diff, H_pnom=H_pnom, pnom=pnom, weight=weight, lb=lb),
                jacobian_func = partial(error_state_pseudorange_jacobians, H_pnom=H_pnom, weight=weight, lb=lb),
                prn = prn,
                weight = weight
        )

def create_robust_error_state_pseudorange_factor(
        pseudorange_meas: float,
        time: float,
        pnom: np.ndarray,
        p_key,
        cb_key,
        R_key,
        sat_xyz: np.ndarray,
        prn: int,
        robust_weight_function: Callable[..., Any],
        lb: np.ndarray = np.zeros(3),
        weight: float = 1.0
) -> RobustPseudorangeFactor:
        # Calculate the expected pseudorange at the linearization point
        r_x0 = np.linalg.norm(sat_xyz - pnom)
        H_pnom, _ = gnss.calc_H_values(sat_xyz, pnom)
        pseudorange_range_diff = pseudorange_meas-r_x0

        return RobustPseudorangeFactor(
                time = time,
                states = [p_key, cb_key, R_key],
                n_rows = 1,  # One row for the pseudorange error
                jacobians_size = 3 + 1 + 3,  # 3 for position, 1 for clock bias, 3 for rotation matrix
                error_func = partial(error_state_pseudorange_error, pseudorange_range_diff = pseudorange_range_diff, H_pnom=H_pnom, pnom=pnom, weight=weight, lb=lb),
                jacobian_func = partial(error_state_pseudorange_jacobians, H_pnom=H_pnom, weight=weight, lb=lb),
                prn = prn,
                weight = weight,
                robust_weight_func = robust_weight_function
        )
               
def create_robust_error_state_pseudorange_factor_unknown_lever_arm(
        pseudorange_meas: float,
        time: float,
        pnom: np.ndarray,
        p_key,
        cb_key,
        R_key,
        lb_key,
        sat_xyz: np.ndarray,
        prn: int,
        robust_weight_function: Callable[..., Any],
        weight: float = 1.0
) -> RobustPseudorangeFactorUnknownLeverArm:
        # Calculate the expected pseudorange at the linearization point
        r_x0 = np.linalg.norm(sat_xyz - pnom)
        H_pnom, _ = gnss.calc_H_values(sat_xyz, pnom)
        pseudorange_range_diff = pseudorange_meas-r_x0

        return RobustPseudorangeFactorUnknownLeverArm(
                time = time,
                states = [p_key, cb_key, R_key, lb_key],
                n_rows = 1,  # One row for the pseudorange error
                jacobians_size = 3 + 1 + 3 + 3,  # 3 for position, 1 for clock bias, 3 for rotation matrix, 3 for lever arm
                error_func = partial(error_state_pseudorange_error_unknown_lever_arm, pseudorange_range_diff = pseudorange_range_diff, H_pnom=H_pnom, pnom=pnom, weight=weight),
                jacobian_func = partial(error_state_pseudorange_jacobians_unknown_lever_arm, H_pnom=H_pnom, weight=weight),
                prn = prn,
                weight = weight,
                robust_weight_func = robust_weight_function
        )
