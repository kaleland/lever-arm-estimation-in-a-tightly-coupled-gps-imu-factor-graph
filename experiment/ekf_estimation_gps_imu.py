from util.measurement_deque import TimedMeasurementQueue
from graphs.ekf import *
from util.constants import DEG_TO_RADIANS, RADIANS_TO_DEG, wei, I3
from util import config, gnss, rcf, weight_from_covariance, imu, so3

import numpy as np
import r3f
from util.data_management import import_gps_data
import os
import time
import numpy as np
from tqdm import tqdm
import yaml


def ekf_estimation_gps_imu(
        sim_subdir='lasso/navigation_grade',
        tdcp_cov_scale=1.0,
        pseudorange_cov_scale=1.0,
        accel_cov_scale=1.0,
        gyro_cov_scale=1.0,
        save=True,
        save_covariances=False,
        true_lever_arm: np.ndarray = np.zeros(3),
        lever_arm_init: np.ndarray = np.zeros(3),
        lever_arm_prior_var: None | float = None,
        output_dir: str = 'results',
        gps_data_file: str = None,
        unknown_lever_arm: bool = True,
        init_rotation_error: np.ndarray = np.eye(3),
        initial_position_error: np.ndarray = np.zeros(3),
        perfect_start: bool = False,
        true_clock_bias_m: float = 0.0,
        true_clock_drift_m_s: float = 0.0):
    """Performs EKF-based navigation state estimation using GPS pseudorange and IMU
    data with a potentially unknown lever arm.

    Inputs:
        sim_subdir: Subdirectory within 'sim' containing the simulation data.
        tdcp_cov_scale: Scaling factor for TDCP measurement covariance.
        pseudorange_cov_scale: Scaling factor for pseudorange measurement covariance.
        accel_cov_scale: Scaling factor for accelerometer noise covariance.
        gyro_cov_scale: Scaling factor for gyroscope noise covariance.
        save: Whether to save the results.
        save_covariances: Whether to save state covariances.
        true_lever_arm: True lever arm value for error calculation.
        lever_arm_init: Initial guess for the lever arm.
        lever_arm_prior_var: Prior variance for the lever arm estimate.
        output_dir: Directory to save the results.
        gps_data_file: Optional GPS data file path.
        unknown_lever_arm: Whether to estimate the lever arm or keep it fixed.
        init_rotation_error: Initial rotation error to apply to the attitude.
        initial_position_error: Initial position error to apply to the position state.

    Returns:
        dict containing estimation results and errors.

    """
    imu_deque = TimedMeasurementQueue()
    pr_deque = TimedMeasurementQueue()
    tdcp_deque = TimedMeasurementQueue(sort_idx=1)

    # Load data and config
    data_dir = './sim/' + sim_subdir + '/'

    with open(data_dir + 'config.yaml', 'r') as f:
        sim_config = yaml.safe_load(f)
        
    # Config
    imu_sensor = sim_config['sensor']['imu']
    imu_noise_model = imu.create_imu_noise_model(config.sim['imu'][imu_sensor])
    clock_sensor = sim_config['sensor']['clock']
    clock_noise_model = config.sim['gps']['clock'][clock_sensor]
    stationary_period = sim_config.get('stationary_period_s', None)

    # IMU
    imu_npz = np.load(data_dir + 'imu.npz')
    imu_data = imu_npz['imu_data']
    true_bias_times = imu_data[:,0]
    true_accel_biases = imu_npz['accel_bias']
    true_gyro_biases = imu_npz['gyro_bias']

    # GPS
    if gps_data_file:
         gps_npz = np.load(gps_data_file)
    else:
         gps_npz = np.load(data_dir + 'gps.npz')
    gps_raw_data = gps_npz['gps']


    # Truth
    truth_npz = np.load(data_dir + 'truth.npz')
    truth_times = truth_npz['times']
    truth_pw = truth_npz['pw']
    truth_eTw = truth_npz['eTw']
    eRw = truth_eTw[:3,:3]
    Omega = so3.skew(eRw.T@wei)
    truth_vw = truth_npz['vw']
    truth_rpy = truth_npz['rpy']
    truth_attitude = np.array([r3f.rpy_to_dcm(rpy).T for rpy in truth_rpy])

    # Initializations
    start_time = 0.0
    init_p = truth_pw[0] + initial_position_error
    init_R = truth_attitude[0] @ init_rotation_error

    accel_bias_init = true_accel_biases[0]
    # Initialize bias variances using the Bias Stability (steady-state sigma),
    # not the process noise covariance (which scales with dt).
    accel_bias_stability_mg = config.sim['imu'][imu_sensor]['accelerometer_bias_stability_mg']
    accel_bias_stability_ms2 = accel_bias_stability_mg * 1e-3 * 9.80665
    accel_bias_init_var = accel_bias_stability_ms2**2

    gyro_bias_init = true_gyro_biases[0]
    gyro_bias_stability_deg_hr = config.sim['imu'][imu_sensor]['gyro_bias_stability_deg_hr']
    gyro_bias_stability_rad_s = gyro_bias_stability_deg_hr * DEG_TO_RADIANS / 3600.0
    gyro_bias_init_var = gyro_bias_stability_rad_s**2

    # Initialize measurement deques
    imu_data[:, 0] = imu_data[:, 0] - start_time  # Normalize time to start_time
    imu_deque.extend(imu_data)
    gps_data = import_gps_data(
        gps_raw_data,
        Tew=truth_eTw,
        graph_start_time_gnss=int(start_time),
        base_phase_sigma=config.sim['gps']['adr']['white_noise_std'],
        base_pr_sigma=config.sim['gps']['pseudorange']['total_sigma'])
    pr_deque.extend(gps_data['pr_msmts'])
    tdcp_deque.extend(gps_data['tdcp_msmts'])
    t_gps = gps_data['t_gnss']

    # Define all state times
    all_times = t_gps.copy()

    # Get truth data on hand
    def get_truth_at_t(t_query, truth_times, truth_vals):
        # find nearest truth index
        indices = np.searchsorted(truth_times, t_query)
        indices = np.clip(indices, 0, len(truth_times)-1)
        # Check if previous index is closer
        prev_indices = np.clip(indices-1, 0, len(truth_times)-1)
        
        dist = np.abs(truth_times[indices] - t_query)
        dist_prev = np.abs(truth_times[prev_indices] - t_query)
        
        use_prev = dist_prev < dist
        indices[use_prev] = prev_indices[use_prev]
        
        return truth_vals[indices]
    
    # Create the EKF
    initial_state = ExtendedKalmanFilterGPSIMU.state_vector_from_values(
        p_w = init_p,
        v_w = np.zeros(3) if stationary_period else truth_vw[0],
        w_R_b = init_R,
        b_a = accel_bias_init,
        b_g = gyro_bias_init,
        c_b = true_clock_bias_m if perfect_start else 0.0,
        c_d = true_clock_drift_m_s if perfect_start else 0.0,
        l_b = lever_arm_init if unknown_lever_arm else None,
    )

    initial_covariance = np.zeros((20,20)) if unknown_lever_arm else np.zeros((17,17))
    initial_covariance[:3,:3] = I3*config.sim['prior_params']['prior_pos_var']
    initial_covariance[3:6,3:6] = I3*config.sim['prior_params']['prior_vel_var']
    initial_covariance[6:9,6:9] = I3*config.sim['prior_params']['prior_att_var']
    if perfect_start:
        initial_covariance[9:12,9:12] = I3*1e-15
        initial_covariance[12:15,12:15] = I3*1e-15
        initial_covariance[15,15] = 1e-6
        initial_covariance[16,16] = 1e-8
    else:
        initial_covariance[9:12,9:12] = I3*accel_bias_init_var
        initial_covariance[12:15,12:15] = I3*gyro_bias_init_var
        initial_covariance[15,15] = config.sim['prior_params']['prior_clock_bias_var']
        initial_covariance[16,16] = config.sim['prior_params']['prior_clock_drift_var']
        
    if unknown_lever_arm:
        initial_covariance[17:20,17:20] = I3*lever_arm_prior_var

    imu_dt = config.sim['imu'][imu_sensor]['sampling_period']
    imu_config = {
        'accel_tau': config.sim['imu'][imu_sensor]['time_constant_accel_bias_s'],
        'gyro_tau': config.sim['imu'][imu_sensor]['time_constant_gyro_bias_s'],
        'sigma_accel':  imu_noise_model['accel_white_noise_sigma']* np.sqrt(accel_cov_scale),
        'sigma_gyro': imu_noise_model['gyro_white_noise_sigma']* np.sqrt(gyro_cov_scale),
        'bias_accel_std': imu_noise_model['accel_bias_driving_std_func'](imu_dt),
        'bias_gyro_std': imu_noise_model['gyro_bias_driving_std_func'](imu_dt),
        'sampling_period': imu_dt
    }

    gnss_config = {
        'clock_noise_model': clock_noise_model,
        'lever_arm': lever_arm_init if lever_arm_init is not None else true_lever_arm
    }

    ekf = ExtendedKalmanFilterGPSIMU(
        initial_state = initial_state,
        initial_covariance = initial_covariance,
        imu_config = imu_config,
        gnss_config = gnss_config,
        eTw = truth_eTw,
        unknown_lever_arm = unknown_lever_arm,
    )

    # Prepare to filter
    current_time = all_times[0]
    position_estimates = TimedMeasurementQueue()
    position_covariances = TimedMeasurementQueue()
    velocity_estimates = TimedMeasurementQueue()
    velocity_covariances = TimedMeasurementQueue()
    rpy_estimates = TimedMeasurementQueue()
    accel_bias_estimates = TimedMeasurementQueue()
    gyro_bias_estimates = TimedMeasurementQueue()
    clock_bias_estimates = TimedMeasurementQueue()
    clock_drift_estimates = TimedMeasurementQueue()
    lever_arm_estimates = TimedMeasurementQueue() if unknown_lever_arm else None
    lever_arm_covariances = TimedMeasurementQueue() if unknown_lever_arm else None

    def save_estimates_and_covariances():
        position_estimates.extend([
            np.concatenate(([current_time], ekf.state[0:3]))
        ])
        position_covariances.extend([
            ekf.covariance[0:3,0:3]
        ])
        velocity_estimates.extend([
            np.concatenate(([current_time], ekf.state[3:6]))
        ])
        velocity_covariances.extend([
            ekf.covariance[3:6,3:6]
        ])
        rpy_estimates.extend([
            np.concatenate(([current_time], r3f.dcm_to_rpy(so3.expmap(ekf.state[6:9]).T)))
        ])
        accel_bias_estimates.extend([
            np.concatenate(([current_time], ekf.state[9:12]))
        ])
        gyro_bias_estimates.extend([
            np.concatenate(([current_time], ekf.state[12:15]))
        ])
        clock_bias_estimates.extend([
            np.concatenate(([current_time], [ekf.state[15]]))
        ])
        clock_drift_estimates.extend([
            np.concatenate(([current_time], [ekf.state[16]]))
        ])
        if unknown_lever_arm:
            lever_arm_estimates.extend([
                np.concatenate(([current_time], ekf.state[17:20]))
            ])
            lever_arm_covariances.extend([
                ekf.covariance[17:20,17:20]
            ])

    save_estimates_and_covariances()


    p_error = None
    v_error = None
    att_error = None
    def compare_current_estimate_to_truth():
        # For debugging: compare current estimate to truth
        est_p = ekf.state[0:3]
        est_v = ekf.state[3:6]
        est_R = so3.expmap(ekf.state[6:9])
        true_p = get_truth_at_t(np.array([current_time]), truth_times, truth_pw)[0]
        true_v = get_truth_at_t(np.array([current_time]), truth_times, truth_vw)[0]
        true_rpy = get_truth_at_t(np.array([current_time]), truth_times, truth_rpy)[0]
        true_R = r3f.rpy_to_dcm(true_rpy).T

        p_error = true_p - est_p
        v_error = true_v - est_v
        att_error = so3.box_minus_right(est_R, true_R)
        return p_error, v_error, att_error

    p_error, v_error, att_error = compare_current_estimate_to_truth()

        
    # Main Filtering Loop
    filter_opt_times = []
    for gps_time in tqdm(all_times[1:]):
        iter_start_time = time.time()
        last_time = current_time
        current_time = gps_time

        if stationary_period is not None and current_time <= stationary_period:
            currently_stationary = True
        else:
            currently_stationary = False

        ### Predict to current time using IMU data
        new_imu_msmts = imu_deque.popleft_through(current_time)
        imu_msmts_list = [IMUMeasurement(accel = msmt[1:4],
                                        gyro = msmt[4:7],) for msmt in new_imu_msmts]
        ekf.predict(imu_msmts_list, predict_period=(current_time - last_time), stationary=currently_stationary)

        ### Compare current estimate to truth
        p_error, v_error, att_error = compare_current_estimate_to_truth()

        ### Update
        # Pseudorange measurements
        new_pr_msmts = pr_deque.popleft_through(current_time)
        pr_msmts_list = [PseudorangeMeasurement(
            satellite_position = msmt[4:7],
            pseudorange = msmt[1],
            covariance = (msmt[2]**2)*pseudorange_cov_scale,
        ) for msmt in new_pr_msmts]
        
        # TDCP measurements
        new_tdcp_msmts = tdcp_deque.popleft_through(current_time)
        tdcp_msmts_list = [TDCPMeasurement(
            satellite_position0 = msmt[5:8],
            satellite_position1 = msmt[8:11],
            tdcp = msmt[2],
            covariance = (msmt[3]**2)*tdcp_cov_scale,
        ) for msmt in new_tdcp_msmts]

        ekf.update_gps(pseudorange_measurements=pr_msmts_list, tdcp_measurements=tdcp_msmts_list, stationary = currently_stationary)

        p_error, v_error, att_error = compare_current_estimate_to_truth()

        ### Save estimates and covariances
        save_estimates_and_covariances()
        iter_end_time = time.time()
        filter_opt_times.append(iter_end_time - iter_start_time)
    print(f"EKF estimation took {np.sum(filter_opt_times):.2f} seconds total.")
    print(f"Average time per iteration: {np.mean(filter_opt_times):.4f} seconds.")

    #### Save results
    ## Evaluate errors
    position_estimates = position_estimates.pop_all()
    position_covariances = position_covariances.pop_all() if save_covariances else None
    velocity_estimates = velocity_estimates.pop_all()
    velocity_covariances = velocity_covariances.pop_all() if save_covariances else None
    rpy_estimates = rpy_estimates.pop_all()
    attitude_estimates = np.array([r3f.rpy_to_dcm(rpy_estimates[i][1:]).T for i in range(len(rpy_estimates))])
    accel_bias_estimates = accel_bias_estimates.pop_all()
    gyro_bias_estimates = gyro_bias_estimates.pop_all()
    clock_bias_estimates = clock_bias_estimates.pop_all()
    clock_drift_estimates = clock_drift_estimates.pop_all()
    
    if unknown_lever_arm:
        lever_arm_estimates = lever_arm_estimates.pop_all()
        lever_arm_covariances = lever_arm_covariances.pop_all() if save_covariances else None
    else:
        lever_arm_estimates = None
        lever_arm_covariances = None

    # Find truth values corresponding to estimate times
    est_times = position_estimates[:,0]
    matched_truth_pos = get_truth_at_t(est_times, truth_times, truth_pw)
    matched_truth_vel = get_truth_at_t(est_times, truth_times, truth_vw)
    matched_truth_att = np.array([r3f.rpy_to_dcm(rpy).T for rpy in get_truth_at_t(est_times, truth_times, truth_rpy)])
    matched_truth_rpy = get_truth_at_t(est_times, truth_times, truth_rpy)
    
    # Calculate errors as true - estimate
    pw_errors = matched_truth_pos - position_estimates[:, 1:]
    vw_errors = matched_truth_vel - velocity_estimates[:, 1:]
    attitude_errors = np.array([so3.box_minus_right(attitude_estimates[i], matched_truth_att[i]) for i in range(len(est_times))])
    rpy_errors = matched_truth_rpy - rpy_estimates[:, 1:]

    # Lever arm errors over time
    lever_arm_errors = None
    final_lever_arm_estimate = None
    final_lever_arm_error = None
    
    if unknown_lever_arm:
        lever_arm_errors = np.array([true_lever_arm - lever_arm_estimates[i][1:] for i in range(len(lever_arm_estimates))])

        # Ensure covariances are numpy array if present
        if lever_arm_covariances is not None and not isinstance(lever_arm_covariances, np.ndarray):
            lever_arm_covariances = np.array(lever_arm_covariances)

        # Final lever arm estimate and error
        final_lever_arm_estimate = lever_arm_estimates[-1][1:]
        final_lever_arm_error = true_lever_arm - final_lever_arm_estimate

        print(f"True lever arm: {true_lever_arm}")
        print(f"Final lever arm estimate: {final_lever_arm_estimate}")
        print(f"Final lever arm error: {final_lever_arm_error}")
        print(f"Final lever arm error norm: {np.linalg.norm(final_lever_arm_error):.4f} m")

    if save:
        if unknown_lever_arm:
             prior_suffix = '_prior' if lever_arm_prior_var is not None else '_no_prior'
             lever_arm_str = f'ekf_unknown_lever_arm{prior_suffix}_true_{true_lever_arm[0]:.2f}_{true_lever_arm[1]:.2f}_{true_lever_arm[2]:.2f}'
        else:
             lever_arm_str = f'ekf_known_lever_arm_true_{true_lever_arm[0]:.2f}_{true_lever_arm[1]:.2f}_{true_lever_arm[2]:.2f}'
        save_dir = os.path.join(output_dir, sim_subdir, lever_arm_str)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_dict = {
            'position_estimates': position_estimates[:, 1:],
            'position_covariances': position_covariances if save_covariances else None,
            'velocity_estimates': velocity_estimates[:, 1:],
            'velocity_covariances': velocity_covariances if save_covariances else None,
            'rpy_estimates': rpy_estimates[:, 1:],
            'attitude_estimates': attitude_estimates,
            'accel_bias_estimates': accel_bias_estimates[:, 1:],
            'gyro_bias_estimates': gyro_bias_estimates[:, 1:],
            'clock_bias_estimates': clock_bias_estimates[:, 1],
            'clock_drift_estimates': clock_drift_estimates[:, 1],
            'pw_errors': pw_errors,
            'vw_errors': vw_errors,
            'attitude_errors': attitude_errors,
            'rpy_errors': rpy_errors,
            'true_positions': matched_truth_pos,
            'true_velocities': matched_truth_vel,
            'true_attitudes': matched_truth_att,
            'true_rpy': matched_truth_rpy,
            'times': est_times, 
            'eTw': truth_eTw,
            'true_lever_arm': true_lever_arm,
        }
        
        if unknown_lever_arm:
             save_dict.update({
                'lever_arm_estimates': lever_arm_estimates,
                'lever_arm_errors': lever_arm_errors,
                'lever_arm_covariances': lever_arm_covariances if save_covariances else None,
                'final_lever_arm_estimate': final_lever_arm_estimate,
                'final_lever_arm_error': final_lever_arm_error,
             })

        np.savez(os.path.join(save_dir, 'results.npz'), **save_dict)

    return {
        'estimate_times': est_times,
        'pw_estimates': position_estimates,
        'vw_estimates': velocity_estimates,
        'rpy_estimates': rpy_estimates,
        'attitude_estimates': attitude_estimates,
        'accel_bias_estimates': accel_bias_estimates,
        'gyro_bias_estimates': gyro_bias_estimates,
        'clock_bias_estimates': clock_bias_estimates,
        'clock_drift_estimates': clock_drift_estimates,
        'pw_errors': pw_errors,
        'vw_errors': vw_errors,
        'attitude_errors': attitude_errors,
        'rpy_errors': rpy_errors,
        'lever_arm_estimates': lever_arm_estimates,
        'lever_arm_errors': lever_arm_errors,
        'lever_arm_covariances': lever_arm_covariances,
        'final_lever_arm_estimate': final_lever_arm_estimate,
        'final_lever_arm_error': final_lever_arm_error,
        'save_path': os.path.join(save_dir, 'results.npz') if save else None
    }

if __name__ == "__main__":
    sim_base_dir = os.path.join('sim', 'lasso')
    grade = "tactical_grade"
    true_lever_arm = np.array([1.0, 0.0, 0.0])
    
    sim_subdir = f"lasso/{grade}"
    sim_dir = os.path.join(sim_base_dir, grade, 'lever_arm_simulations')
    gps_file = os.path.join(sim_dir, "gps_lever_arm_1.00_0.00_0.00.npz")

    lever_arm_prior_var = np.eye(3)
    init_lever_arm = true_lever_arm + np.random.multivariate_normal(mean=np.zeros(3), cov=lever_arm_prior_var)

    ekf_estimation_gps_imu(
        sim_subdir,
        save = True,
        true_lever_arm=true_lever_arm,
        lever_arm_init = init_lever_arm,
        lever_arm_prior_var=lever_arm_prior_var,
        unknown_lever_arm = True,
        gps_data_file=gps_file
    )
    print(f"Completed EKF-GPS-IMU estimation for dataset: {sim_dir}")

    
