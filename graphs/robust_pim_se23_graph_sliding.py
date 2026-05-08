from typing import List

from graphs import RobustFactorGraph, PIMSE23GraphSliding
from factors import clock_transition
from factors.gnss import tdcp, pseudorange
from factors.pim import imu_bias_transition, preintegrated_se23
from factors.stationary import tdcp_stationary, preintegrated_se23_stationary
from util import pim, rcf, so3
from util.constants import wei
import numpy as np

class RobustPIMSE23GraphSliding(RobustFactorGraph, PIMSE23GraphSliding):
    """A robust PIM SE(2,3) graph for sliding window optimization.
    Inherits from RobustFactorGraph and PIMSE23GraphSliding.
    """
    
    def __init__(
            self, 
            Tew, 
            lb, 
            pr_weight_function = rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight), 
            tdcp_weight_function = rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight), 
            pim_weight_function = rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight), 
            clock_transition_weight_function = rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight),
            imu_bias_transition_weight_function = rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight)
            ):
        """Initialize."""
        PIMSE23GraphSliding.__init__(self, Tew=Tew, lb=lb)
        self.pr_weight_function = pr_weight_function
        self.tdcp_weight_function = tdcp_weight_function
        self.pim_weight_function = pim_weight_function
        self.clock_transition_weight_function = clock_transition_weight_function
        self.imu_bias_transition_weight_function = imu_bias_transition_weight_function

    def add_error_state_tdcp_factor(
            self,
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
            pnom: np.ndarray,
            weight = 1.0,
            lb = np.zeros(3)
    ):
        """Add a robust error-state time-differenced code phase (TDCP) factor to the graph."""
        if p0_key == p1_key or R0_key == R1_key:
            raise NotImplementedError("Stationary error-state TDCP factors not implemented.")
        
        self.add_factor(tdcp.create_robust_error_state_tdcp_factor(
            tdcp_meas=meas,
            time0=time0,
            time1=time1,
            p0_key=p0_key,
            p1_key=p1_key,
            cb0_key=cb0_key,
            cb1_key=cb1_key,
            R0_key=R0_key,
            R1_key=R1_key,
            sat_xyz0=sat_xyz0,
            sat_xyz1=sat_xyz1,
            prn = prn,
            pnom = pnom,
            weight=weight,
            lb=lb,
            robust_weight_function=self.tdcp_weight_function
        ))

    def add_tdcp_factor(
            self,
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
            lb = np.zeros(3)):
        """Add a robust time-differenced code phase (TDCP) factor to the graph."""
        if p0_key == p1_key and R0_key == R1_key:
            self.add_factor(tdcp_stationary.create_robust_tdcp_stationary_factor(
                meas=meas,
                time0=time0,
                time1=time1,
                p_key=p0_key,
                cb0_key=cb0_key,
                cb1_key=cb1_key,
                R_key=R0_key,
                sat_xyz0=sat_xyz0,
                sat_xyz1=sat_xyz1,
                prn=prn,
                robust_weight_function=self.tdcp_weight_function,
                weight=weight,
                lb=lb
            ))
        elif p0_key != p1_key and R0_key != R1_key:
            self.add_factor(tdcp.create_robust_tdcp_factor(
                meas=meas,
                time0=time0,
                time1=time1,
                p0_key=p0_key,
                p1_key=p1_key,
                cb0_key=cb0_key,
                cb1_key=cb1_key,
                R0_key=R0_key,
                R1_key=R1_key,
                sat_xyz0=sat_xyz0,
                sat_xyz1=sat_xyz1,
                prn = prn,
                robust_weight_function=self.tdcp_weight_function,
                weight=weight,
                lb=lb
            ))  
        elif p0_key != p1_key and R0_key == R1_key and np.linalg.norm(self.lb)<1e-3:
            self.add_factor(tdcp.create_robust_tdcp_factor(
                meas=meas,
                time0=time0,
                time1=time1,
                p0_key=p0_key,
                p1_key=p1_key,
                cb0_key=cb0_key,
                cb1_key=cb1_key,
                R0_key=R0_key,
                R1_key=R1_key,
                sat_xyz0=sat_xyz0,
                sat_xyz1=sat_xyz1,
                prn = prn,
                robust_weight_function=self.tdcp_weight_function,
                weight=weight,
                lb=lb
            )) 
        else:
            raise ValueError("Invalid combination of position and rotation keys for TDCP factor.")


    def add_error_state_pseudorange_factor(
            self,
            meas,
            time,
            p_key,
            cb_key,
            R_key,
            sat_xyz,
            prn: int,
            pnom: np.ndarray,
            lb = np.zeros(3),
            weight = 1.0):
        """Add a robust error-state pseudorange factor to the graph."""
        self.add_factor(pseudorange.create_robust_error_state_pseudorange_factor(
            pseudorange_meas=meas,
            time=time,
            p_key=p_key,
            cb_key=cb_key,
            R_key=R_key,
            sat_xyz=sat_xyz,
            prn=prn,
            pnom=pnom,
            lb=lb,
            weight=weight,
            robust_weight_function=self.pr_weight_function
        ))

    def add_pseudorange_factor(
            self,
            meas,
            time,
            p_key,
            cb_key,
            R_key,
            sat_xyz,
            prn: int,
            lb = np.zeros(3),
            weight = 1.0):
        """Add a robust pseudorange factor to the graph."""
        self.add_factor(pseudorange.create_robust_pseudorange_factor(
            meas=meas,
            time=time,
            p_key=p_key,
            cb_key=cb_key,
            R_key=R_key,
            sat_xyz=sat_xyz,
            prn = prn,
            robust_weight_function=self.pr_weight_function,
            lb=lb,
            weight=weight
        ))

    def add_pim_factor(
            self,
            time0: float,
            Ri_key: str,
            vi_key: str,
            pi_key: str,
            Rj_key: str,
            vj_key: str,
            pj_key: str,
            ba_key: str,
            bg_key: str,
            imu_msmts: List[pim.IMUMeasurement],
            accel_white_noise: np.ndarray,
            gyro_white_noise: np.ndarray,
            gravity: np.ndarray,
    ):
        
        omega = self.Rew.T@wei
        accel_bias = self.state_dict[ba_key].value
        gyro_bias = self.state_dict[bg_key].value


        if pi_key == pj_key and vi_key == vj_key and Ri_key == Rj_key:
            self.add_factor(preintegrated_se23_stationary.create_robust_pim_se23_factor(
                time0 = time0,
                R_key = Ri_key,
                p_key = pi_key,
                ba_key = ba_key,
                bg_key = bg_key,
                omega = omega,
                imu_msmts=imu_msmts,
                accel_bias = accel_bias,
                gyro_bias = gyro_bias,
                accel_white_noise = accel_white_noise,
                gyro_white_noise = gyro_white_noise,
                gravity = gravity,
                robust_weight_function = self.pim_weight_function
            ))

        elif pi_key != pj_key and vi_key != vj_key and Ri_key != Rj_key:
            self.add_factor(preintegrated_se23.create_robust_pim_se23_factor(
                time0 = time0,
                Ri_key = Ri_key,
                vi_key = vi_key,
                pi_key = pi_key,
                Rj_key = Rj_key,
                vj_key = vj_key,
                pj_key = pj_key,
                ba_key = ba_key,
                bg_key = bg_key,
                omega = omega,
                imu_msmts=imu_msmts,
                accel_bias = accel_bias,
                gyro_bias = gyro_bias,
                accel_white_noise=accel_white_noise,
                gyro_white_noise=gyro_white_noise,
                gravity=gravity,
                robust_weight_function = self.pim_weight_function
            ))

        else:
            raise ValueError("Invalid combination of position, velocity, and attitude keys for PIM factor.")

    def add_clock_transition_factor(self, bias_time_0, clock_bias_0_key, clock_drift_0_key, clock_bias_1_key, clock_drift_1_key, delta_t, weight=1.0):
        """Add a clock transition factor to the graph."""
        self.add_factor(clock_transition.create_robust_clock_transition_factor(
            bias_time_0=bias_time_0,
            bias_time_1=bias_time_0 + delta_t,
            clock_bias_0_key=clock_bias_0_key,
            clock_drift_0_key=clock_drift_0_key,
            clock_bias_1_key=clock_bias_1_key,
            clock_drift_1_key=clock_drift_1_key,
            delta_t=delta_t,
            robust_weight_function=self.clock_transition_weight_function,
            weight=weight
        ))

    def add_imu_bias_transition_factor(self, bias_time_0, imu_bias_0_key, imu_bias_1_key, dt, tau, weight=np.eye(3)):
        """Add an IMU bias transition factor to the graph."""
        # Calculate bias_time_1 from bias_time_0 + dt for backward compatibility
        bias_time_1 = bias_time_0 + dt
        self.add_factor(imu_bias_transition.create_robust_bias_transition_factor(
            bias_time_0=bias_time_0,
            bias_time_1=bias_time_1,
            imu_bias_0_key=imu_bias_0_key,
            imu_bias_1_key=imu_bias_1_key,
            dt=dt,
            tau=tau,
            weight=weight,
            robust_weight_function=self.imu_bias_transition_weight_function
        ))
        