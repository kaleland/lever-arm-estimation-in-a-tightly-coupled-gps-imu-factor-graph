from graphs import FactorGraph
from factors import marginalization, prior, Factor, clock_transition
from factors.gnss import tdcp, pseudorange
from factors.pim import preintegrated_se23, imu_bias_transition
from factors.pim.preintegrated_se23 import PIMFactor, IMUBiasValues
from factors.stationary import preintegrated_se23_stationary
from factors.stationary.preintegrated_se23_stationary import PIMFactorStationary
from factors.gnss.tdcp import RobustTDCPFactor
from factors.gnss.pseudorange import RobustPseudorangeFactor
from factors.pim.imu_bias_transition import IMUBiasTransitionFactor
from util.constants import I3, wei
from util import sparsify_dense_matrix, householder_qr, prefixed_counter
from util import so3, se23
from states import AccelBiasState, GyroBiasState

import numpy as np
from typing import List
from functools import partial
import copy
from itertools import count


class PIMSE23GraphSliding(FactorGraph):
    """A factor graph using extended pose PIM, pseudorange measurements, and
    TDCP measurements to estimate position, velocity, attitude, clock bias,
    and IMU biases over a sliding window of time.
    """

    def __init__(self, Tew, lb):
        """Initialize."""
        super().__init__()

        self.Rew = Tew[:3,:3]
        self.pew = Tew[:3,3]
        self.lb = lb

        self.pim_factor_idxs = List[int]  # List of indices of PIM factors in the factors list


    # Optimization                 
    def update_states(self, delta: np.ndarray):
        """Update the states in the graph with the given delta vector.
        The delta vector is expected to be in the same order as the state keys 
        in the state_index_dict.
        """
        # Store a copy of the current state_dict before updating
        self.old_state_dict = copy.deepcopy(self.state_dict)
        self.old_pim_factors = copy.deepcopy([factor for factor in self.factors if isinstance(factor, PIMFactor)])

        # Apply the delta to the state_dict
        state_index_dict = self.gen_state_index_dict()
        for key, index in state_index_dict.items():
            state_dim = self.state_dict[key].delta_dimensions

            # Update the state with the corresponding delta
            if state_dim == 1:
                self.state_dict[key].update(delta[index])
            else:
                self.state_dict[key].update(delta[index:index + state_dim])

                if isinstance(self.state_dict[key], AccelBiasState) or isinstance(self.state_dict[key], GyroBiasState):
                    # Find the index of the PIM in self.pim_msmts_and_bias_keys
                    matching_factors = [factor for factor in self.factors if isinstance(factor, PIMFactor) and key in factor.states]
                    for factor in matching_factors:
                        # Update the PIM measurement with the delta
                        factor_accel_bias_key = factor.states[-2]
                        factor_gyro_bias_key = factor.states[-1]
                        new_bias = IMUBiasValues(
                            accel_bias=self.state_dict[factor_accel_bias_key].value,
                            gyro_bias=self.state_dict[factor_gyro_bias_key].value
                        )

                        factor.update_bias(new_bias=new_bias)  
                        stop_here = True
    def revert_states(self):
        """Revert the states in the graph to the previous state_dict.
        This is useful if the optimization step needs to be rolled back.
        """
        self.state_dict = copy.deepcopy(self.old_state_dict)   
        pim_factors = [factor for factor in self.factors if isinstance(factor, PIMFactor)]
        for factor_idx, pim_factor in enumerate(pim_factors):
            if isinstance(pim_factor, PIMFactor):
                old_factor = self.old_pim_factors[factor_idx]

                # Revert the PIM factors to their previous state
                pim_factor.ups_hat = old_factor.ups_hat
                pim_factor.error_func = old_factor.error_func
                pim_factor.jacobian_func = old_factor.jacobian_func

    def add_clock_transition_factor(self, bias_time_0, clock_bias_0_key, clock_drift_0_key, clock_bias_1_key, clock_drift_1_key, delta_t, weight=1.0):
        """Add a clock transition factor to the graph."""
        self.add_factor(clock_transition.create_clock_transition_factor(
            bias_time_0=bias_time_0,
            bias_time_1=bias_time_0 + delta_t,
            clock_bias_0_key=clock_bias_0_key,
            clock_drift_0_key=clock_drift_0_key,
            clock_bias_1_key=clock_bias_1_key,
            clock_drift_1_key=clock_drift_1_key,
            delta_t=delta_t,
            weight=weight
        ))

    def add_imu_bias_transition_factor(self, bias_time_0, imu_bias_0_key, imu_bias_1_key, dt, tau, weight=I3):
        """Add an IMU bias transition factor to the graph."""
        # Calculate bias_time_1 from bias_time_0 + dt for backward compatibility
        bias_time_1 = bias_time_0 + dt
        self.add_factor(imu_bias_transition.create_bias_transition_factor(
            bias_time_0=bias_time_0,
            bias_time_1=bias_time_1,
            imu_bias_0_key=imu_bias_0_key,
            imu_bias_1_key=imu_bias_1_key,
            dt=dt,
            tau=tau,
            weight=weight
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
        """Add a time-differenced code phase (TDCP) factor to the graph."""
        self.add_factor(tdcp.create_tdcp_factor(
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
            weight=weight,
            lb=lb
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
        """Add a pseudorange factor to the graph."""
        self.add_factor(pseudorange.create_pseudorange_factor(
            meas=meas,
            time=time,
            p_key=p_key,
            cb_key=cb_key,
            R_key=R_key,
            sat_xyz=sat_xyz,
            prn = prn,
            lb=lb,
            weight=weight
        ))

    def add_pim_factor(
            self,
            time0: float,
            ups_hat: np.ndarray,
            Ri_key: str,
            vi_key: str,
            pi_key: str,
            Rj_key: str,
            vj_key: str,
            pj_key: str,
            ba_key: str,
            bg_key: str,
            bias_update_jacobian: np.ndarray,
            GammaR: np.ndarray,
            GammaV: np.ndarray,
            GammaP: np.ndarray,
            delta_t: float,
            weight: np.ndarray
    ):
        
        Omega = so3.skew(self.Rew.T@wei)

        self.add_factor(preintegrated_se23.create_pim_se23_factor(
            time0 = time0,
            ups_hat = ups_hat,
            Ri_key = Ri_key,
            vi_key = vi_key,
            pi_key = pi_key,
            Rj_key = Rj_key,
            vj_key = vj_key,
            pj_key = pj_key,
            ba_key = ba_key,
            bg_key = bg_key,
            bias_update_jacobian = bias_update_jacobian,
            GammaR = GammaR,
            GammaV = GammaV,
            GammaP = GammaP,
            delta_t = delta_t,
            weight = weight,
            Omega = Omega
        ))

