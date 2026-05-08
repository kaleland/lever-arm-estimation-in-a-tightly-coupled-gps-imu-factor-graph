import numpy as np
from util import gnss, so3, imu, constants
from factors.gnss import pseudorange, tdcp
from dataclasses import dataclass

@dataclass
class PseudorangeMeasurement:

    satellite_position: np.ndarray  # (3,)
    pseudorange: float
    covariance: float

@dataclass
class TDCPMeasurement:

    satellite_position0: np.ndarray  # (3,)
    satellite_position1: np.ndarray  # (3,)
    tdcp: float
    covariance: float

@dataclass
class IMUMeasurement:

    accel: np.ndarray  # (3,)
    gyro: np.ndarray  # (3,)


def ekf_imu_propagate(
    pw, vw, wRb, b_a, b_g,
    acc_meas, gyro_meas, dt,
    sigma_acc, sigma_gyro,
    bias_acc_std, bias_gyro_std,
    tau_acc, tau_gyro,
    eTw
):
    """acc_meas, gyro_meas: arrays of shape (N,3)
    dt: scalar timestep.
    """
    eRw = eTw[0:3, 0:3]
    pe = eTw[0:3, 3]
    def pw_to_pe(pw):
        
        return eRw@pw + pe

    N = acc_meas.shape[0]

    Phi = np.eye(15)
    Qd  = np.zeros((15, 15))

    phi_ba = np.exp(-dt / tau_acc)
    phi_bg = np.exp(-dt / tau_gyro)

    omega_wi = eRw.T@constants.wei
    Omega_wi = so3.skew(omega_wi)

    for k in range(N):
        # ---- Nominal bias propagation ----
        b_a = phi_ba * b_a
        b_g = phi_bg * b_g

        # Bias-corrected measurements
        f_b = acc_meas[k] - b_a

        # Gravity and Rotation of the Earth at estimated PVA
        ge = imu.grav_somigliana_pe(pw_to_pe(pw))
        g = eRw.T @ ge
        

        omega_eb_b = (
            gyro_meas[k]
            - b_g
            - wRb.T @ omega_wi
        )

        # ---- Nominal state propagation ----

        # Attitude Propagation
        wRb = wRb @ so3.expmap(omega_eb_b * dt) 

        # Acceleration in world frame
        a_w = (
            wRb @ f_b
            - 2 * Omega_wi @ vw
            + g
        )

        # Position and Velocity Propagation
        pw = pw + vw * dt + 0.5 * a_w * dt**2
        vw = vw + a_w * dt

        # ---- Error-state dynamics ----
        F = np.zeros((15, 15))

        # Position
        F[0:3, 3:6] = np.eye(3)

        # Velocity
        F[3:6, 3:6] = -2*Omega_wi
        F[3:6, 6:9] = -wRb @ so3.skew(f_b)
        F[3:6, 9:12] = -wRb

        # Attitude
        F[6:9, 6:9] = -so3.skew(omega_eb_b)
        F[6:9, 12:15] = -np.eye(3)

        # FOGM Bias dynamics
        F[9:12, 9:12]   = -(1.0 / tau_acc) * np.eye(3)
        F[12:15, 12:15] = -(1.0 / tau_gyro) * np.eye(3)        

        # Discrete transition
        Phi_k = np.eye(15) + F * dt

        # ---- Discrete noise ----
        Qk = np.zeros((15, 15))
        Qk[3:6, 3:6] = sigma_acc**2 * dt * np.eye(3)
        Qk[6:9, 6:9] = sigma_gyro**2 * dt * np.eye(3)

        Qk[9:12, 9:12]   = bias_acc_std**2 * np.eye(3)
        Qk[12:15, 12:15] = bias_gyro_std**2 * np.eye(3)


        # Accumulate
        Qd = Phi_k @ Qd @ Phi_k.T + Qk
        Phi = Phi_k @ Phi

    # # Final covariance propagation
    # P = Phi @ P @ Phi.T + Qd

    return pw, vw, wRb, b_a, b_g, Phi, Qd

class ExtendedKalmanFilterGPSIMU:
    """An Extended Kalman Filter (EKF) implementation for navigation with GPS
    pseudorange and TDCP measurements and preintegrated IMU data (delta-R/v/p).
    The IMU data are used for propagation, while GPS measurements are used for
    updates. The state vector is:
    x = [p_w, v_w, w_r_b, b_a, b_g, c_b, c_d, l_b]
        - p_w: position in local tangent world frame (3,), index 0-2
        - v_w: velocity in local tangent world frame (3,), index 3-5
        - w_r_b: Log(w_R_b) rotation from body to world frame (3,), index 6-8
        - b_a: accelerometer bias (3,), index 9-11
        - b_g: gyroscope bias (3,), index 12-14
        - c_b: clock bias (1,), index 15
        - c_d: clock drift (1,), index 16
        - l_b: lever arm (3,), index 17-19.
    """

    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, imu_config: dict, gnss_config: dict, eTw: np.ndarray, unknown_lever_arm: bool = True):
        """Initialize."""
        self.state = initial_state  # State vector
        self.covariance = initial_covariance  # State covariance matrix
        self._old_state: np.ndarray | None = None
        self._old_covariance: np.ndarray | None = None
        self.accel_tau = imu_config['accel_tau']
        self.gyro_tau = imu_config['gyro_tau']
        self.sigma_accel = imu_config['sigma_accel']
        self.sigma_gyro = imu_config['sigma_gyro']
        self.bias_accel_std = imu_config['bias_accel_std']
        self.bias_gyro_std = imu_config['bias_gyro_std']

        self.clock_noise_model = gnss_config['clock_noise_model']
        self.eTw = eTw
        self.imu_dt = imu_config['sampling_period']
        self.unknown_lever_arm = unknown_lever_arm
        if not unknown_lever_arm:
            self.lb = gnss_config['lever_arm']  # Known lever arm
        if unknown_lever_arm:
            assert self.state.shape[0] == 20, "State vector size must be 20 when lever arm is unknown."
            assert self.covariance.shape == (20, 20), "Covariance matrix size must be 20x20 when lever arm is unknown."
        else:
            assert self.state.shape[0] == 17, "State vector size must be 17 when lever arm is known."
            assert self.covariance.shape == (17, 17), "Covariance matrix size must be 17x17 when lever arm is known."
    
    def predict(self, imu_msmts: list[IMUMeasurement], predict_period:float, stationary: bool = False):
        
        if self.unknown_lever_arm:
            pw0, vw0, wRb0, ba0, bg0, cb0, cd0, lb0 = self.state_values_tuple()
        else:
            pw0, vw0, wRb0, ba0, bg0, cb0, cd0 = self.state_values_tuple()
            lb0 = np.zeros(3)
        self._old_state = self.state.copy()
        self._old_covariance = self.covariance.copy()

        F_full = np.eye(self.state.shape[0])
        Q_full = np.zeros((self.state.shape[0], self.state.shape[0]))

        # Perform PVA prediction using preintegrated IMU measurements
        accel_msmts = np.array([m.accel for m in imu_msmts])
        gyro_msmts = np.array([m.gyro for m in imu_msmts])
        pw1, vw1, wRb1, ba1, bg1, F_pva_ba_bg, Q_pva_ba_bg = ekf_imu_propagate(
            pw0, vw0, wRb0, ba0, bg0,
            accel_msmts, gyro_msmts, self.imu_dt,
            self.sigma_accel, self.sigma_gyro,
            self.bias_accel_std, self.bias_gyro_std,
            self.accel_tau, self.gyro_tau,
            self.eTw  
        )

        # Propagate Clock Bias States and Lever Arm
        cb1 = cb0 + cd0*predict_period  # Simple linear model for clock bias
        cd1 = cd0  # Clock drift follows a random walk
        if self.unknown_lever_arm:
            lb1 = lb0  # Lever arm is constant

        # Fill in F_full and Q_full for clock bias, clock drift, and lever arm
        F_full[15, 16] = predict_period  # dc_b/dc_d
        F_full[15, 15] = 1.0  # dc_b/dc_b
        F_full[16, 16] = 1.0  # dc_d/dc_d
        if self.unknown_lever_arm:
            F_full[17:20, 17:20] = np.eye(3)  # dl_b/dl_b

        Q_full[15:17, 15:17] = gnss.clock_transition_variance(self.clock_noise_model, predict_period)
        
        if self.unknown_lever_arm:
            Q_full[17:20, 17:20] = np.zeros((3,3))  # No lever arm propagation noise

        if stationary == False:
            # Replace state vector with propagated values
            if self.unknown_lever_arm:
                self.state = self.state_tuple_to_vector((pw1, vw1, wRb1, ba1, bg1, cb1, cd1, lb1))
            else:
                self.state = self.state_tuple_to_vector((pw1, vw1, wRb1, ba1, bg1, cb1, cd1))
            
            # Propagate covariance

            F_full[0:15, 0:15] = F_pva_ba_bg
            Q_full[0:15, 0:15] = Q_pva_ba_bg
            
        else:
            # Replace state vector with propagated values
            if self.unknown_lever_arm:
                self.state = self.state_tuple_to_vector((pw0, vw0, wRb0, ba1, bg1, cb1, cd1, lb0))
            else:
                self.state = self.state_tuple_to_vector((pw0, vw0, wRb0, ba1, bg1, cb1, cd1))
            
            # Propagate covariance
            F_full[0:10, 0:10] = np.eye(10)
            Q_full[0:10, 0:10] = np.zeros((10,10))
            F_full[10:15, 10:15] = F_pva_ba_bg[10:15, 10:15]
            Q_full[10:15, 10:15] = Q_pva_ba_bg[10:15, 10:15]

        P = self.covariance
        P_new = F_full @ P @ F_full.T + Q_full
        self.covariance = P_new

    def update_gps(self, pseudorange_measurements: list[PseudorangeMeasurement], tdcp_measurements: list[TDCPMeasurement], stationary: bool = False):
        
        # Extract Current State
        if self.unknown_lever_arm:
            pw, vw, wRb, ba, bg, cb, cd, lb = self.state_values_tuple()
        else:
            pw, vw, wRb, ba, bg, cb, cd = self.state_values_tuple()

        # Evaluate Pseudorange Residuals and Jacobians
        n_pseudo = len(pseudorange_measurements)
        pseudorange_errors = np.zeros((n_pseudo,))
        pseudorange_variances = np.zeros((n_pseudo,))
        pseudorange_jacobians = np.zeros((n_pseudo, len(self.state)))
        if self.unknown_lever_arm:
            pr_states = (pw, cb, wRb, lb)
        else:
            pr_states = (pw, cb, wRb, self.lb)

        for pr_idx, pr_msmt in enumerate(pseudorange_measurements):
            # Errors to build y
            pr_error = pseudorange.pseudorange_error_unknown_lever_arm(
                meas = pr_msmt.pseudorange,
                state_tuple = pr_states,
                sat_xyz = pr_msmt.satellite_position,
                weight = 1.0 # Measurement covariances are used later
            )
            pseudorange_errors[pr_idx] = pr_error

            # Jacobians to build H
            dr_dp, dr_dcb, dr_dR, dr_dl = pseudorange.pseudorange_jacobians_unknown_lever_arm(
                state_tuple = pr_states,
                sat_xyz = pr_msmt.satellite_position,
                weight = 1.0 # Measurement covariances are used later
            )
            H_row = np.zeros((len(self.state),))
            H_row[0:3] = -dr_dp
            H_row[6:9] = -dr_dR
            H_row[15] = -dr_dcb
            if self.unknown_lever_arm:
                H_row[17:20] = -dr_dl
            pseudorange_jacobians[pr_idx, :] = H_row
            
            # Variances to build R
            pseudorange_variances[pr_idx] = pr_msmt.covariance

        # Evaluate TDCP Residuals and Jacobians
        n_tdcp = len(tdcp_measurements)
        if self.unknown_lever_arm:
            pw_old, _, wRb_old, _, _, cb_old, _, _ = self.state_values_tuple_from_vector(self._old_state)
        else:
            pw_old, _, wRb_old, _, _, cb_old, _ = self.state_values_tuple_from_vector(self._old_state)
        tdcp_errors = np.zeros((n_tdcp,))
        tdcp_variances = np.zeros((n_tdcp,))
        tdcp_jacobians = np.zeros((n_tdcp, len(self.state)))
        if self.unknown_lever_arm:
            tdcp_states = (pw_old, pw, cb_old, cb, wRb_old, wRb, lb)
        else:
            tdcp_states = (pw_old, pw, cb_old, cb, wRb_old, wRb, self.lb)

        for tdcp_idx, tdcp_msmt in enumerate(tdcp_measurements):
            # Errors to build y
            tdcp_error = tdcp.tdcp_error_unknown_lever_arm(
                meas = tdcp_msmt.tdcp,
                state_tuple = tdcp_states,
                sat_xyz0 = tdcp_msmt.satellite_position0,
                sat_xyz1 = tdcp_msmt.satellite_position1,
                weight = 1.0 # Measurement covariances are used later
            )
            tdcp_errors[tdcp_idx] = tdcp_error

            # Jacobians to build H
            dr_dp0, dr_dp1, dr_dcb0, dr_dcb1, dr_dR0, dr_dR1, dr_dl = tdcp.tdcp_jacobians_unknown_lever_arm(
                state_tuple = tdcp_states,
                sat_xyz0 = tdcp_msmt.satellite_position0,
                sat_xyz1 = tdcp_msmt.satellite_position1,
                weight = 1.0 # Measurement covariances are used later
            )
            H_row = np.zeros((len(self.state),))
            H_row[0:3] = -dr_dp1  # Current position
            H_row[6:9] = -dr_dR1    # Current orientation
            H_row[15] = -dr_dcb1   # Current clock bias
            if self.unknown_lever_arm:
                H_row[17:20] = -dr_dl   # Lever arm
            tdcp_jacobians[tdcp_idx, :] = H_row

            # Variances to build R
            tdcp_variances[tdcp_idx] = tdcp_msmt.covariance
        
        # Stationary Update
        if stationary:
            stationary_error = vw - np.zeros(3)
            stationary_jacobians = np.zeros((3, len(self.state)))
            stationary_jacobians[:, 3:6] = np.eye(3)
            stationary_update_covariance = 1e-5 * np.eye(3)
        else:
            stationary_error = None
            stationary_jacobians = None
            stationary_update_covariance = None


        # Update State and Covariance
        if stationary:
            y = np.concatenate([pseudorange_errors, tdcp_errors, stationary_error])
            H = np.vstack([pseudorange_jacobians, tdcp_jacobians, stationary_jacobians])
            R = np.diag(np.concatenate([pseudorange_variances, tdcp_variances, stationary_update_covariance.diagonal()]))
        else:
            y = np.concatenate([pseudorange_errors, tdcp_errors])
            H = np.vstack([pseudorange_jacobians, tdcp_jacobians])
            R = np.diag(np.concatenate([pseudorange_variances, tdcp_variances]))
            
        P = self.covariance
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        delta = K @ y
        self.update_state(delta)
        new_covariance = (np.eye(len(self.state)) - K @ H) @ P 
        self.covariance = new_covariance
        
    def state_values_tuple(self):
        
        return self.state_values_tuple_from_vector(self.state)
    
    def state_values_tuple_from_vector(self, state_vector: np.ndarray):
        
        p_w = state_vector[0:3]
        v_w = state_vector[3:6]
        w_R_b = so3.expmap(state_vector[6:9])
        b_a = state_vector[9:12]
        b_g = state_vector[12:15]
        c_b = state_vector[15]
        c_d = state_vector[16]
        if self.unknown_lever_arm:
            l_b = state_vector[17:20]
            return (p_w, v_w, w_R_b, b_a, b_g, c_b, c_d, l_b)
        else:
            return (p_w, v_w, w_R_b, b_a, b_g, c_b, c_d)
    
    def state_covariances_tuple(self):
        
        return self.state_covariances_tuple_from_matrix(self.covariance)
    
    def state_covariances_tuple_from_matrix(self, covariance_matrix: np.ndarray):
        
        p_w_cov = covariance_matrix[0:3, 0:3]
        v_w_cov = covariance_matrix[3:6, 3:6]
        w_R_b_cov = covariance_matrix[6:9, 6:9]
        b_a_cov = covariance_matrix[9:12, 9:12]
        b_g_cov = covariance_matrix[12:15, 12:15]
        c_b_cov = covariance_matrix[15, 15]
        c_d_cov = covariance_matrix[16, 16]
        if self.unknown_lever_arm:
            l_b_cov = covariance_matrix[17:20, 17:20]
            return (p_w_cov, v_w_cov, w_R_b_cov, b_a_cov, b_g_cov, c_b_cov, c_d_cov, l_b_cov)
        else:
            return (p_w_cov, v_w_cov, w_R_b_cov, b_a_cov, b_g_cov, c_b_cov, c_d_cov)
    
    def state_tuple_to_vector(self, state_tuple)-> np.ndarray:
        
        if self.unknown_lever_arm:
            p_w, v_w, w_R_b, b_a, b_g, c_b, c_d, l_b = state_tuple
            w_r_b = so3.logmap(w_R_b)
            return np.concatenate([p_w, v_w, w_r_b, b_a, b_g, np.array([c_b]), np.array([c_d]), l_b])
        else:
            p_w, v_w, w_R_b, b_a, b_g, c_b, c_d = state_tuple
            w_r_b = so3.logmap(w_R_b)
            return np.concatenate([p_w, v_w, w_r_b, b_a, b_g, np.array([c_b]), np.array([c_d])])

    
    def update_state(self, delta: np.ndarray):
        
        new_state = np.zeros_like(self.state)

        # Arthmetic update for position, velocity, biases, clock, lever arm
        new_state[0:6] = self.state[0:6] + delta[0:6]  # Position and Velocity
        new_state[9:] = self.state[9:] + delta[9:]  # Biases, clock, lever arm (optional)

        # Lie Group update for SO(3) rotation
        current_rot = so3.expmap(self.state[6:9])
        delta_rot = so3.expmap(delta[6:9])
        new_state[6:9] = so3.logmap(current_rot @ delta_rot)  # Rotation

        self.state = new_state

    @staticmethod
    def state_vector_from_values(
        p_w: np.ndarray,
        v_w: np.ndarray,
        w_R_b: np.ndarray,
        b_a: np.ndarray,
        b_g: np.ndarray,
        c_b: float,
        c_d: float,
        l_b: np.ndarray | None = None
    ) -> np.ndarray:
        
        w_r_b = so3.logmap(w_R_b)
        if l_b is not None:
            return np.concatenate([p_w, v_w, w_r_b, b_a, b_g, np.array([c_b]), np.array([c_d]), l_b])
        else:
            return np.concatenate([p_w, v_w, w_r_b, b_a, b_g, np.array([c_b]), np.array([c_d])])


class ExtendedKalmanFilterGPS:
    """An Extended Kalman Filter (EKF) implementation for navigation with GPS
    pseudorange and TDCP measurements and preintegrated IMU data (delta-R/v/p).
    The IMU data are used for propagation, while GPS measurements are used for
    updates. The state vector is:
    x = [p_w, v_w, c_b, c_d]
        - p_w: position in local tangent world frame (3,), index 0-2
        - v_w: velocity in local tangent world frame (3,), index 3-5
        - c_b: clock bias (1,), index 6
        - c_d: clock drift (1,), index 7.
    """

    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, gnss_config: dict, eTw: np.ndarray):
        """Initialize."""
        self.state = initial_state  # State vector
        self.covariance = initial_covariance  # State covariance matrix
        self._old_state: np.ndarray | None = None
        self._old_covariance: np.ndarray | None = None
        self.clock_noise_model = gnss_config['clock_noise_model']
        self.eTw = eTw
        assert self.state.shape[0] == 8
        assert self.covariance.shape == (8, 8)

    def generate_Q_and_F_for_position_velocity(self, predict_period: float, q: float = 0.5):
        """Generates F and Q for state vector [px, py, pz, vx, vy, vz].
        dt: time step in seconds
        q: process noise spectral density (approx max_accel^2).
        """
        # 1. State Transition Matrix (F)
        # x_new = x + vx*dt, etc.
        F = np.eye(6)
        for i in range(3):
            F[i, i + 3] = predict_period
            
        # 2. Process Noise Matrix (Q)
        # Discrete-time white noise acceleration model
        Q = np.zeros((6, 6))
        
        # Pre-calculate Q-block values for a single axis
        q11 = (q * predict_period**3) / 3  # variance in position
        q12 = (q * predict_period**2) / 2  # covariance position-velocity
        q22 = q * predict_period           # variance in velocity
        
        for i in range(3):
            # Map 2x2 axis blocks into the 6x6 matrix
            Q[i, i] = q11          # Position variance (0,0), (1,1), (2,2)
            Q[i, i + 3] = q12      # Pos-Vel covariance (0,3), (1,4), (2,5)
            Q[i + 3, i] = q12      # Vel-Pos covariance (3,0), (4,1), (5,2)
            Q[i + 3, i + 3] = q22  # Velocity variance (3,3), (4,4), (5,5)
            
        return F, Q

    def predict(self, predict_period:float, q: float = 5.0,stationary: bool = False):
        
        pw0, vw0, cb0, cd0= self.state_values_tuple()
        q = 1e-6 if stationary else q

        self._old_state = self.state.copy()
        self._old_covariance = self.covariance.copy()

        F_full = np.eye(self.state.shape[0])
        Q_full = np.zeros((self.state.shape[0], self.state.shape[0]))


        # Propagate Clock Bias States and Lever Arm
        cb1 = cb0 + cd0*predict_period  # Simple linear model for clock bias
        cd1 = cd0  # Clock drift follows a random walk

        # Propagate position and velocity with simple dynamics model
        pw1 = pw0 + vw0*predict_period if not stationary else pw0
        vw1 = vw0 if not stationary else np.zeros(3)

        # Fill in F_full and Q_full for position and velocity
        F_pv, Q_pv = self.generate_Q_and_F_for_position_velocity(predict_period = predict_period, q = q)
        F_full[0:6, 0:6] = F_pv
        Q_full[0:6, 0:6] = Q_pv

        # Fill in F_full and Q_full for clock bias, clock drift
        F_full[6,7] = predict_period
        Q_full[6:8,6:8 ] = gnss.clock_transition_variance(self.clock_noise_model, predict_period)


        # Replace state vector with propagated values
        self.state = self.state_tuple_to_vector((pw1, vw1, cb1, cd1))
            


        P = self.covariance
        P_new = F_full @ P @ F_full.T + Q_full
        self.covariance = P_new

    def update_gps(self, pseudorange_measurements: list[PseudorangeMeasurement], tdcp_measurements: list[TDCPMeasurement], stationary: bool = False):
        
        # Extract Current State
        pw, vw, cb, cd = self.state_values_tuple()


        # Evaluate Pseudorange Residuals and Jacobians
        n_pseudo = len(pseudorange_measurements)
        pseudorange_errors = np.zeros((n_pseudo,))
        pseudorange_variances = np.zeros((n_pseudo,))
        pseudorange_jacobians = np.zeros((n_pseudo, len(self.state)))

        pr_states = (pw, cb, np.eye(3), np.zeros(3)) # Position,Clock Bias, Attitude, Lever Arm

        for pr_idx, pr_msmt in enumerate(pseudorange_measurements):
            # Errors to build y
            pr_error = pseudorange.pseudorange_error_unknown_lever_arm(
                meas = pr_msmt.pseudorange,
                state_tuple = pr_states,
                sat_xyz = pr_msmt.satellite_position,
                weight = 1.0 # Measurement covariances are used later
            )
            pseudorange_errors[pr_idx] = pr_error

            # Jacobians to build H
            dr_dp, dr_dcb, _, _ = pseudorange.pseudorange_jacobians_unknown_lever_arm(
                state_tuple = pr_states,
                sat_xyz = pr_msmt.satellite_position,
                weight = 1.0 # Measurement covariances are used later
            )
            H_row = np.zeros((len(self.state),))
            H_row[0:3] = -dr_dp
            H_row[6] = -dr_dcb
            pseudorange_jacobians[pr_idx, :] = H_row
            
            # Variances to build R
            pseudorange_variances[pr_idx] = pr_msmt.covariance

        # Evaluate TDCP Residuals and Jacobians
        n_tdcp = len(tdcp_measurements)
        pw_old, _, cb_old, _ = self.state_values_tuple_from_vector(self._old_state)
        tdcp_errors = np.zeros((n_tdcp,))
        tdcp_variances = np.zeros((n_tdcp,))
        tdcp_jacobians = np.zeros((n_tdcp, len(self.state)))
        tdcp_states = (pw_old, pw, cb_old, cb, np.eye(3), np.eye(3), np.zeros(3))

        for tdcp_idx, tdcp_msmt in enumerate(tdcp_measurements):
            # Errors to build y
            tdcp_error = tdcp.tdcp_error_unknown_lever_arm(
                meas = tdcp_msmt.tdcp,
                state_tuple = tdcp_states,
                sat_xyz0 = tdcp_msmt.satellite_position0,
                sat_xyz1 = tdcp_msmt.satellite_position1,
                weight = 1.0 # Measurement covariances are used later
            )
            tdcp_errors[tdcp_idx] = tdcp_error

            # Jacobians to build H
            dr_dp0, dr_dp1, dr_dcb0, dr_dcb1, _, _, _ = tdcp.tdcp_jacobians_unknown_lever_arm(
                state_tuple = tdcp_states,
                sat_xyz0 = tdcp_msmt.satellite_position0,
                sat_xyz1 = tdcp_msmt.satellite_position1,
                weight = 1.0 # Measurement covariances are used later
            )
            H_row = np.zeros((len(self.state),))
            H_row[0:3] = -dr_dp1  # Current position
            H_row[6] = -dr_dcb1   # Current clock bias
            tdcp_jacobians[tdcp_idx, :] = H_row

            # Variances to build R
            tdcp_variances[tdcp_idx] = tdcp_msmt.covariance
        
        # Stationary Update
        if stationary:
            stationary_error = vw - np.zeros(3)
            stationary_jacobians = np.zeros((3, len(self.state)))
            stationary_jacobians[:, 3:6] = np.eye(3)
            stationary_update_covariance = 1e-5 * np.eye(3)
        else:
            stationary_error = None
            stationary_jacobians = None
            stationary_update_covariance = None


        # Update State and Covariance
        if stationary:
            y = np.concatenate([pseudorange_errors, tdcp_errors, stationary_error])
            H = np.vstack([pseudorange_jacobians, tdcp_jacobians, stationary_jacobians])
            R = np.diag(np.concatenate([pseudorange_variances, tdcp_variances, stationary_update_covariance.diagonal()]))
        else:
            y = np.concatenate([pseudorange_errors, tdcp_errors])
            H = np.vstack([pseudorange_jacobians, tdcp_jacobians])
            R = np.diag(np.concatenate([pseudorange_variances, tdcp_variances]))
            
        P = self.covariance
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        delta = K @ y
        self.update_state(delta)
        new_covariance = (np.eye(len(self.state)) - K @ H) @ P 
        self.covariance = new_covariance
        
    def state_values_tuple(self):
        
        return self.state_values_tuple_from_vector(self.state)
    
    def state_values_tuple_from_vector(self, state_vector: np.ndarray):
        
        p_w = state_vector[0:3]
        v_w = state_vector[3:6]
        c_b = state_vector[6]
        c_d = state_vector[7]

        return (p_w, v_w, c_b, c_d)
    
    def state_covariances_tuple(self):
        
        return self.state_covariances_tuple_from_matrix(self.covariance)
    
    def state_covariances_tuple_from_matrix(self, covariance_matrix: np.ndarray):
        
        p_w_cov = covariance_matrix[0:3, 0:3]
        v_w_cov = covariance_matrix[3:6, 3:6]
        c_b_cov = covariance_matrix[6, 6]
        c_d_cov = covariance_matrix[7, 7]
        return (p_w_cov, v_w_cov, c_b_cov, c_d_cov)
    
    def state_tuple_to_vector(self, state_tuple)-> np.ndarray:
        
        p_w, v_w, c_b, c_d = state_tuple
        return np.concatenate([p_w, v_w, np.array([c_b]), np.array([c_d])])

    
    def update_state(self, delta: np.ndarray):
        
        # Arthmetic update for position, velocity, clock, drift
        self.state = self.state + delta

    @staticmethod
    def state_vector_from_values(
        p_w: np.ndarray,
        v_w: np.ndarray,
        c_b: float,
        c_d: float,
    ) -> np.ndarray:
        
        return np.concatenate([p_w, v_w, np.array([c_b]), np.array([c_d])])    


class ExtendedKalmanFilterGPSAugmentedState:
    """An Extended Kalman Filter (EKF) implementation for navigation with GPS
    pseudorange and TDCP measurements and preintegrated IMU data (delta-R/v/p).
    The IMU data are used for propagation, while GPS measurements are used for
    updates. The state vector is:
    x = [p_w, v_w, c_b, c_d]
        - p_w: position in local tangent world frame (3,), index 0-2
        - v_w: velocity in local tangent world frame (3,), index 3-5
        - c_b: clock bias (1,), index 6
        - c_d: clock drift (1,), index 7
        - p_w_old: position in local tangent world frame (3,), index 8-10
        - c_b_old: clock bias (1,), index 11.
    """

    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, gnss_config: dict, eTw: np.ndarray):
        """Initialize."""
        self.state = initial_state  # State vector
        self.covariance = initial_covariance  # State covariance matrix
        self._old_state: np.ndarray | None = None
        self._old_covariance: np.ndarray | None = None
        self.clock_noise_model = gnss_config['clock_noise_model']
        self.eTw = eTw
        assert self.state.shape[0] == 12
        assert self.covariance.shape == (12, 12)

    def generate_Q_and_F_for_position_velocity(self, predict_period: float, q: float = 0.5):
        """Generates F and Q for augmented state:
        [px_k, py_k, pz_k, vx_k, vy_k, vz_k, px_{k-1}, py_{k-1}, pz_{k-1}].
        """
        # 1. State Transition Matrix F (9x9)
        F = np.eye(9)
        
        # Propagate current position using velocity (same as before)
        for i in range(3):
            F[i, i + 3] = dt   # px_k = px_{k-1_aug} + vx * dt  (but via the blocks below)
        
        # The cloned previous position (indices 6:9) comes from the *old* current position.
        # In the predict step, after the previous update, the current becomes the new "previous".
        # So we set:
        #   cloned_pos = previous_current_pos   → this is handled by the block structure
        
        # Full F:
        # Top 6x6: original propagation for current states
        # Bottom-right 3x3: identity for cloned position (it stays fixed until next augmentation)
        # Cross terms: the cloned position at step k is the current position from step k-1
        
        # More explicitly (standard stochastic cloning form):
        Phi = np.eye(6)                     # original 6x6 transition without noise
        for i in range(3):
            Phi[i, i + 3] = dt
        
        # Augmented F:
        F = np.block([
            [Phi, np.zeros((6, 3))],        # current states propagate normally
            [np.eye(3), np.zeros((3, 3))]   # cloned previous pos = old current pos (shift)
            # Wait — actually the shift happens in how you augment after update
        ])
        
        # Correct common implementation:
        # After update at k-1, you augment by appending the *current* estimate of position as the cloned block.
        # Then for predict to k:
        F = np.block([
            [Phi, np.zeros((6, 3))],   # new current pos/vel from dynamics
            [np.zeros((3, 6)), np.eye(3)]  # cloned pos stays the same (it's the old one)
        ])
        
        # 2. Process Noise Matrix Q (only on current states)
        Q = np.zeros((9, 9))
        
        # Same as your original for the current 6 states (indices 0:6)
        q11 = (q * dt**3) / 3
        q12 = (q * dt**2) / 2
        q22 = q * dt
        
        for i in range(3):
            Q[i, i]         = q11
            Q[i, i + 3]     = q12
            Q[i + 3, i]     = q12
            Q[i + 3, i + 3] = q22
        
        # Cloned position (indices 6:9) gets NO process noise
        # Cross blocks between current and cloned remain zero in Q

        return F, Q

    def predict(self, predict_period:float, q: float = 5.0,stationary: bool = False):
        
        pw0, vw0, cb0, cd0= self.state_values_tuple()
        q = 1e-6 if stationary else q

        self._old_state = self.state.copy()
        self._old_covariance = self.covariance.copy()

        F_full = np.eye(self.state.shape[0])
        Q_full = np.zeros((self.state.shape[0], self.state.shape[0]))


        # Propagate Clock Bias States and Lever Arm
        cb1 = cb0 + cd0*predict_period  # Simple linear model for clock bias
        cd1 = cd0  # Clock drift follows a random walk
        cb_old = cb0

        # Propagate position and velocity with simple dynamics model
        pw1 = pw0 + vw0*predict_period if not stationary else pw0
        pw_old = pw0
        vw1 = vw0 if not stationary else np.zeros(3)

        # Fill in F_full and Q_full for position and velocity
        F_pvp, Q_pvp = self.generate_Q_and_F_for_position_velocity(predict_period = predict_period, q = q)
        F_full[[0,1,2,3,4,5,8,10], [0,1,2,3,4,5,8,10]] = F_pvp
        Q_full[[0,1,2,3,4,5,8,10], [0,1,2,3,4,5,8,10]] = Q_pvp

        # Fill in F_full and Q_full for clock bias, clock drift
        F_full[6,7] = predict_period
        Q_full[6:8,6:8 ] = gnss.clock_transition_variance(self.clock_noise_model, predict_period)


        # Replace state vector with propagated values
        self.state = self.state_tuple_to_vector((pw1, vw1, cb1, cd1, pw_old, cb_old))
            


        P = self.covariance
        P_new = F_full @ P @ F_full.T + Q_full
        self.covariance = P_new

    def update_gps(self, pseudorange_measurements: list[PseudorangeMeasurement], tdcp_measurements: list[TDCPMeasurement], stationary: bool = False):
        
        # Extract Current State
        pw, vw, cb, cd, pw_old, cb_old = self.state_values_tuple()


        # Evaluate Pseudorange Residuals and Jacobians
        n_pseudo = len(pseudorange_measurements)
        pseudorange_errors = np.zeros((n_pseudo,))
        pseudorange_variances = np.zeros((n_pseudo,))
        pseudorange_jacobians = np.zeros((n_pseudo, len(self.state)))

        pr_states = (pw, cb, np.eye(3), np.zeros(3)) # Position,Clock Bias, Attitude, Lever Arm

        for pr_idx, pr_msmt in enumerate(pseudorange_measurements):
            # Errors to build y
            pr_error = pseudorange.pseudorange_error_unknown_lever_arm(
                meas = pr_msmt.pseudorange,
                state_tuple = pr_states,
                sat_xyz = pr_msmt.satellite_position,
                weight = 1.0 # Measurement covariances are used later
            )
            pseudorange_errors[pr_idx] = pr_error

            # Jacobians to build H
            dr_dp, dr_dcb, _, _ = pseudorange.pseudorange_jacobians_unknown_lever_arm(
                state_tuple = pr_states,
                sat_xyz = pr_msmt.satellite_position,
                weight = 1.0 # Measurement covariances are used later
            )
            H_row = np.zeros((len(self.state),))
            H_row[0:3] = -dr_dp
            H_row[6] = -dr_dcb
            pseudorange_jacobians[pr_idx, :] = H_row
            
            # Variances to build R
            pseudorange_variances[pr_idx] = pr_msmt.covariance

        # Evaluate TDCP Residuals and Jacobians
        n_tdcp = len(tdcp_measurements)
        tdcp_errors = np.zeros((n_tdcp,))
        tdcp_variances = np.zeros((n_tdcp,))
        tdcp_jacobians = np.zeros((n_tdcp, len(self.state)))
        tdcp_states = (pw_old, pw, cb_old, cb, np.eye(3), np.eye(3), np.zeros(3))

        for tdcp_idx, tdcp_msmt in enumerate(tdcp_measurements):
            # Errors to build y
            tdcp_error = tdcp.tdcp_error_unknown_lever_arm(
                meas = tdcp_msmt.tdcp,
                state_tuple = tdcp_states,
                sat_xyz0 = tdcp_msmt.satellite_position0,
                sat_xyz1 = tdcp_msmt.satellite_position1,
                weight = 1.0 # Measurement covariances are used later
            )
            tdcp_errors[tdcp_idx] = tdcp_error

            # Jacobians to build H
            dr_dp0, dr_dp1, dr_dcb0, dr_dcb1, _, _, _ = tdcp.tdcp_jacobians_unknown_lever_arm(
                state_tuple = tdcp_states,
                sat_xyz0 = tdcp_msmt.satellite_position0,
                sat_xyz1 = tdcp_msmt.satellite_position1,
                weight = 1.0 # Measurement covariances are used later
            )
            H_row = np.zeros((len(self.state),))
            H_row[0:3] = -dr_dp1  # Current position
            H_row[6] = -dr_dcb1   # Current clock bias
            H_row[8:11] = -dr_dp0  # Old position
            H_row[11] = -dr_dcb0   # Old clock bias
            tdcp_jacobians[tdcp_idx, :] = H_row

            # Variances to build R
            tdcp_variances[tdcp_idx] = tdcp_msmt.covariance
        
        # Stationary Update
        if stationary:
            stationary_error = vw - np.zeros(3)
            stationary_jacobians = np.zeros((3, len(self.state)))
            stationary_jacobians[:, 3:6] = np.eye(3)
            stationary_update_covariance = 1e-5 * np.eye(3)
        else:
            stationary_error = None
            stationary_jacobians = None
            stationary_update_covariance = None


        # Update State and Covariance
        if stationary:
            y = np.concatenate([pseudorange_errors, tdcp_errors, stationary_error])
            H = np.vstack([pseudorange_jacobians, tdcp_jacobians, stationary_jacobians])
            R = np.diag(np.concatenate([pseudorange_variances, tdcp_variances, stationary_update_covariance.diagonal()]))
        else:
            y = np.concatenate([pseudorange_errors, tdcp_errors])
            H = np.vstack([pseudorange_jacobians, tdcp_jacobians])
            R = np.diag(np.concatenate([pseudorange_variances, tdcp_variances]))
            
        P = self.covariance
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        delta = K @ y
        self.update_state(delta)
        new_covariance = (np.eye(len(self.state)) - K @ H) @ P 
        self.covariance = new_covariance
        
    def state_values_tuple(self):
        
        return self.state_values_tuple_from_vector(self.state)
    
    def state_values_tuple_from_vector(self, state_vector: np.ndarray):
        
        p_w = state_vector[0:3]
        v_w = state_vector[3:6]
        c_b = state_vector[6]
        c_d = state_vector[7]
        p_w_old = state_vector[8:11]
        c_b_old = state_vector[11]

        return (p_w, v_w, c_b, c_d, p_w_old, c_b_old)
    
    def state_covariances_tuple(self):
        
        return self.state_covariances_tuple_from_matrix(self.covariance)
    
    def state_covariances_tuple_from_matrix(self, covariance_matrix: np.ndarray):
        
        p_w_cov = covariance_matrix[0:3, 0:3]
        v_w_cov = covariance_matrix[3:6, 3:6]
        c_b_cov = covariance_matrix[6, 6]
        c_d_cov = covariance_matrix[7, 7]   
        p_w_old_cov = covariance_matrix[8:11, 8:11]
        c_b_old_cov = covariance_matrix[11, 11]
        return (p_w_cov, v_w_cov, c_b_cov, c_d_cov, p_w_old_cov, c_b_old_cov)
    
    def state_tuple_to_vector(self, state_tuple)-> np.ndarray:
        
        p_w, v_w, c_b, c_d, p_w_old, c_b_old = state_tuple
        return np.concatenate([p_w, v_w, np.array([c_b]), np.array([c_d]), p_w_old, np.array([c_b_old])])

    
    def update_state(self, delta: np.ndarray):
        
        # Arthmetic update for position, velocity, clock, drift
        self.state = self.state + delta

    @staticmethod
    def state_vector_from_values(
        p_w: np.ndarray,
        v_w: np.ndarray,
        c_b: float,
        c_d: float,
        p_w_old: np.ndarray,
        c_b_old: float,
    ) -> np.ndarray:
        
        return np.concatenate([p_w, v_w, np.array([c_b]), np.array([c_d]), p_w_old, np.array([c_b_old])])   