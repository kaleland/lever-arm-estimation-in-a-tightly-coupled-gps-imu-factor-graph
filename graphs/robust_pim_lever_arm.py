from graphs import RobustFactorGraph, RobustPIMSE23GraphSliding
from factors import marginalization, prior, Factor
from factors.gnss import tdcp, pseudorange, loosely_coupled
from factors.pim import imu_bias_transition, preintegrated_se23
from factors.stationary import tdcp_stationary, preintegrated_se23_stationary
from util import pim, rcf, so3
from util.constants import wei
import numpy as np

class RobustPIMLeverArmGraph(RobustPIMSE23GraphSliding):
    """A robust PIM SE(2,3) graph for sliding window optimization.
    Inherits from RobustFactorGraph and PIMSE23GraphSliding.
    """
    
    def __init__(self, Tew, pr_weight_function, tdcp_weight_function, pim_weight_function = rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight), clock_transition_weight_function = rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight), imu_bias_transition_weight_function = rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight), lc_gps_weight_function = rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight)):
        """Initialize."""
        RobustPIMSE23GraphSliding.__init__(self, Tew=Tew, lb=None, pr_weight_function=pr_weight_function, tdcp_weight_function=tdcp_weight_function, pim_weight_function=pim_weight_function, clock_transition_weight_function=clock_transition_weight_function, imu_bias_transition_weight_function=imu_bias_transition_weight_function)
        self.lc_gps_weight_function = lc_gps_weight_function


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
            lb_key,
            sat_xyz0,
            sat_xyz1,
            prn: int,
            pnom: np.ndarray,
            weight = 1.0,
    ):
        """Add a robust error-state time-differenced code phase (TDCP) factor to the graph."""
        if p0_key == p1_key or R0_key == R1_key:
            raise NotImplementedError("Stationary error-state TDCP factors not implemented.")
        
        self.add_factor(tdcp.create_robust_error_state_tdcp_factor_unknown_lever_arm(
            tdcp_meas=meas,
            time0=time0,
            time1=time1,
            p0_key=p0_key,
            p1_key=p1_key,
            cb0_key=cb0_key,
            cb1_key=cb1_key,
            R0_key=R0_key,
            R1_key=R1_key,
            lb_key=lb_key,
            sat_xyz0=sat_xyz0,
            sat_xyz1=sat_xyz1,
            prn = prn,
            pnom = pnom,
            weight=weight,
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
            lb_key,
            sat_xyz0,
            sat_xyz1,
            prn: int,
            weight = 1.0):
        """Add a robust time-differenced code phase (TDCP) factor to the graph."""
        if p0_key == p1_key and R0_key == R1_key:
            self.add_factor(tdcp_stationary.create_robust_tdcp_stationary_unknown_lever_arm_factor(
                meas=meas,
                time0=time0,
                time1=time1,
                p_key=p0_key,
                cb0_key=cb0_key,
                cb1_key=cb1_key,
                R_key=R0_key,
                lb_key=lb_key,
                sat_xyz0=sat_xyz0,
                sat_xyz1=sat_xyz1,
                prn=prn,
                robust_weight_function=self.tdcp_weight_function,
                weight=weight
            ))
        elif p0_key != p1_key and R0_key != R1_key:
            self.add_factor(tdcp.create_robust_tdcp_factor_unknown_lever_arm(
                meas=meas,
                time0=time0,
                time1=time1,
                p0_key=p0_key,
                p1_key=p1_key,
                cb0_key=cb0_key,
                cb1_key=cb1_key,
                R0_key=R0_key,
                R1_key=R1_key,
                lb_key=lb_key,
                sat_xyz0=sat_xyz0,
                sat_xyz1=sat_xyz1,
                prn = prn,
                robust_weight_function=self.tdcp_weight_function,
                weight=weight
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
            lb_key,
            sat_xyz,
            prn: int,
            pnom: np.ndarray,
            weight = 1.0):
        """Add a robust error-state pseudorange factor to the graph."""
        self.add_factor(pseudorange.create_robust_error_state_pseudorange_factor_unknown_lever_arm(
            pseudorange_meas=meas,
            time=time,
            p_key=p_key,
            cb_key=cb_key,
            R_key=R_key,
            lb_key=lb_key,
            sat_xyz=sat_xyz,
            prn=prn,
            pnom=pnom,
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
            lb_key,
            sat_xyz,
            prn: int,
            weight = 1.0):
        """Add a robust pseudorange factor to the graph."""
        self.add_factor(pseudorange.create_robust_pseudorange_factor_unknown_lever_arm(
            meas=meas,
            time=time,
            p_key=p_key,
            cb_key=cb_key,
            R_key=R_key,
            lb_key=lb_key,
            sat_xyz=sat_xyz,
            prn = prn,
            robust_weight_function=self.pr_weight_function,
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
            imu_msmts: list[pim.IMUMeasurement],
            accel_white_noise: np.ndarray,
            gyro_white_noise: np.ndarray,
            gravity: np.ndarray,
    ):
        
        RobustPIMSE23GraphSliding.add_pim_factor(
            self = self,
            time0=time0,
            Ri_key=Ri_key,
            vi_key=vi_key,
            pi_key=pi_key,
            Rj_key=Rj_key,
            vj_key=vj_key,
            pj_key=pj_key,
            ba_key=ba_key,
            bg_key=bg_key,
            imu_msmts=imu_msmts,
            accel_white_noise=accel_white_noise,
            gyro_white_noise=gyro_white_noise,
            gravity=gravity
        )
        
    def add_loosely_coupled_gnss_factor(
            self,
            p_meas,
            v_meas,
            wb_raw,
            time,
            pw_key,
            vw_key,
            R_key,
            bg_key,
            lb_key,
            weight = np.eye(6)
    ):
        """Add a robust loosely coupled GNSS factor to the graph."""
        self.add_factor(loosely_coupled.create_robust_loosely_coupled_gnss_factor(
            p_meas=p_meas,
            v_meas=v_meas,
            wb_raw=wb_raw,
            time=time,
            pw_key=pw_key,
            vw_key=vw_key,
            R_key=R_key,
            bg_key=bg_key,
            lb_key=lb_key,
            robust_weight_func=self.lc_gps_weight_function,
            weight=weight
        ))
