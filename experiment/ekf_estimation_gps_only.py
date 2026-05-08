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


def ekf_estimation_gps_only(
        sim_subdir='lasso/navigation_grade',
        tdcp_cov_scale=1.0,
        pseudorange_cov_scale=1.0,
        save=True,
        save_covariances=False,
        output_dir: str = 'results',
        gps_data_file: str = None,
        initial_position_error: np.ndarray = np.zeros(3),
        perfect_start: bool = False,
        true_clock_bias_m: float = 0.0,
        true_clock_drift_m_s: float = 0.0,
        true_lever_arm_body = np.zeros(3)):
    """Performs EKF-based navigation state estimation using GPS pseudorange and TDCP Data.

    Inputs:
        sim_subdir: Subdirectory within 'sim' containing the simulation data.
        tdcp_cov_scale: Scaling factor for TDCP measurement covariance.
        pseudorange_cov_scale: Scaling factor for pseudorange measurement covariance.
        save: Whether to save the results.
        save_covariances: Whether to save state covariances.
        output_dir: Directory to save the results.
        gps_data_file: Optional GPS data file path.
        initial_position_error: Initial position error to apply to the position state.

    Returns:
        dict containing estimation results and errors.

    """
    imu_deque = TimedMeasurementQueue()
    pr_deque = TimedMeasurementQueue()
    tdcp_deque = TimedMeasurementQueue(sort_idx=1)

    # Load data and config
    data_dir = './sim/' + sim_subdir + '/'
    if 'lasso' not in sim_subdir:
        raise ValueError("This function is currently configured to work with 'lasso' dataset structure. Please ensure sim_subdir is correct.")
    
    with open(data_dir + 'config.yaml', 'r') as f:
        sim_config = yaml.safe_load(f)
        
    # Config
    imu_sensor = sim_config['sensor']['imu']
    clock_sensor = sim_config['sensor']['clock']
    clock_noise_model = config.sim['gps']['clock'][clock_sensor]
    stationary_period = sim_config.get('stationary_period_s', None)

    # GPS
    if gps_data_file:
         gps_npz = np.load(gps_data_file)
    else:
         gps_npz = np.load(data_dir + 'gps.npz')
    gps_raw_data = gps_npz['gps']


    # Truth
    truth_npz = np.load(data_dir + 'truth.npz')
    truth_times = truth_npz['times']
    _truth_pw = truth_npz['pw']
    truth_eTw = truth_npz['eTw']
    _truth_vw = truth_npz['vw']
    truth_rpy = truth_npz['rpy']
    truth_attitude = np.array([r3f.rpy_to_dcm(rpy).T for rpy in truth_rpy])
    # Load true t_pvafr file
    if imu_sensor == 'tactical_grade':
        t_pvafr_path = os.path.join('sim', 'sim_trajectory_lasso_100Hz.npz')
    elif imu_sensor == 'navigation_grade':
        t_pvafr_path = os.path.join('sim', 'sim_trajectory_lasso_250Hz.npz')
    else:
        raise ValueError(f"Unsupported IMU sensor type: {imu_sensor}")
    tpva_fr = np.load(t_pvafr_path)['tpva_fr']
    gyro_data_rad = tpva_fr[:, -3:]
    times = tpva_fr[:, 0]
    gyro_at_truth_times = np.zeros_like(gyro_data_rad)
    for i in range(3):
        gyro_at_truth_times[:, i] = np.interp(truth_times, times, gyro_data_rad[:, i])
    truth_vw_gps = np.array([_truth_vw[idx] + truth_attitude[idx]@(np.cross(gyro_at_truth_times[idx], true_lever_arm_body)) for idx in range(len(truth_times))])
    truth_pw_gps = np.array([_truth_pw[idx] + truth_attitude[idx]@true_lever_arm_body for idx in range(len(truth_times))])

    # Initializations
    start_time = 0.0
    init_p = truth_pw_gps[0] + initial_position_error




    # Initialize measurement deques
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
    initial_state = ExtendedKalmanFilterGPS.state_vector_from_values(
        p_w = init_p,
        v_w = np.zeros(3) if stationary_period else truth_vw_gps[0],
        c_b = true_clock_bias_m if perfect_start else 0.0,
        c_d = true_clock_drift_m_s if perfect_start else 0.0,
    )

    initial_covariance = np.zeros((8,8))
    initial_covariance[:3,:3] = I3*config.sim['prior_params']['prior_pos_var']
    initial_covariance[3:6,3:6] = I3*config.sim['prior_params']['prior_vel_var']
    if perfect_start:
        initial_covariance[6,6] = 1e-6
        initial_covariance[7,7] = 1e-8
    else:
        initial_covariance[6,6] = config.sim['prior_params']['prior_clock_bias_var']
        initial_covariance[7,7] = config.sim['prior_params']['prior_clock_drift_var']


    gnss_config = {
        'clock_noise_model': clock_noise_model,
    }

    ekf = ExtendedKalmanFilterGPS(
        initial_state = initial_state,
        initial_covariance = initial_covariance,
        gnss_config = gnss_config,
        eTw = truth_eTw,
    )

    # Prepare to filter
    current_time = all_times[0]
    position_estimates = TimedMeasurementQueue()
    position_covariances = TimedMeasurementQueue()
    velocity_estimates = TimedMeasurementQueue()
    velocity_covariances = TimedMeasurementQueue()
    clock_bias_estimates = TimedMeasurementQueue()
    clock_drift_estimates = TimedMeasurementQueue()

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
        clock_bias_estimates.extend([
            np.concatenate(([current_time], [ekf.state[6]]))
        ])
        clock_drift_estimates.extend([
            np.concatenate(([current_time], [ekf.state[7]]))
        ])


    save_estimates_and_covariances()


    p_error = None
    v_error = None
    att_error = None
    def compare_current_estimate_to_truth():
        # For debugging: compare current estimate to truth
        est_p = ekf.state[0:3]
        est_v = ekf.state[3:6]
        true_p = get_truth_at_t(np.array([current_time]), truth_times, truth_pw_gps)[0]
        true_v = get_truth_at_t(np.array([current_time]), truth_times, truth_vw_gps)[0]

        p_error = true_p - est_p
        v_error = true_v - est_v
        return p_error, v_error

    p_error, v_error = compare_current_estimate_to_truth()

        
    # Main Filtering Loop
    filter_opt_times = []
    for gps_time in tqdm(all_times[1:]):
        iter_start_time = time.time()
        last_time = current_time
        current_time = gps_time

        if stationary_period is not None and current_time < stationary_period:
            currently_stationary = True
        else:
            currently_stationary = False

        ### Predict to current time using IMU data
        ekf.predict(predict_period=(current_time - last_time), stationary=currently_stationary)

        ### Compare current estimate to truth
        p_error, v_error = compare_current_estimate_to_truth()

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

        p_error, v_error = compare_current_estimate_to_truth()

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
    clock_bias_estimates = clock_bias_estimates.pop_all()
    clock_drift_estimates = clock_drift_estimates.pop_all()
    

    # Find truth values corresponding to estimate times
    est_times = position_estimates[:,0]
    matched_truth_pos = get_truth_at_t(est_times, truth_times, truth_pw_gps)
    matched_truth_vel = get_truth_at_t(est_times, truth_times, truth_vw_gps)
    
    # Calculate errors as true - estimate
    pw_errors = matched_truth_pos - position_estimates[:, 1:]
    vw_errors = matched_truth_vel - velocity_estimates[:, 1:]
    vw_estimate_times = est_times + 0.25
    vw_estimates_interpolated = np.zeros_like(velocity_estimates[:, 1:])
    for i in range(3):
        vw_estimates_interpolated[:, i] = np.interp(vw_estimate_times, est_times, velocity_estimates[:, i+1])
    vw_errors_interpolated = matched_truth_vel - vw_estimates_interpolated

    # # Plot true vs estimated positions and velocities over time
    # # Position first - one plot with a subplot for each axis
    # from matplotlib import pyplot as plt
    # fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # for i in range(3):
    #     axes[i].plot(est_times, matched_truth_pos[:, i], label='True Position')
    #     axes[i].plot(est_times, position_estimates[:, i+1], label='Estimated Position')
    #     axes[i].set_ylabel(f'Position {i+1} (m)')
    #     axes[i].legend()
    # axes[-1].set_xlabel('Time (s)')
    # plt.suptitle('True vs Estimated Positions')

    # # Velocity second - one plot with a subplot for each axis
    # fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # for i in range(3):
    #     axes[i].plot(est_times, matched_truth_vel[:, i], label='True Velocity')
    #     axes[i].plot(est_times, vw_estimates_interpolated[:, i], label='Estimated Velocity Interpolated')
    #     axes[i].plot(vw_estimate_times, velocity_estimates[:, i+1], label='Estimated Velocity')
    #     axes[i].set_ylabel(f'Velocity {i+1} (m/s)')
    #     axes[i].legend()
    # axes[-1].set_xlabel('Time (s)')
    # plt.suptitle('True vs Estimated Velocities')

    # # Now plot position and velocity errors over time
    # fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # for i in range(3):
    #     axes[i].plot(est_times, pw_errors[:, i], label='Position Error')
    #     axes[i].set_ylabel(f'Position Error {i+1} (m)')
    #     axes[i].legend()
    # axes[-1].set_xlabel('Time (s)')
    # plt.suptitle('Position Errors')

    # fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # for i in range(3):
    #     axes[i].plot(est_times, vw_errors[:, i], label='Velocity Error')
    #     axes[i].plot(vw_estimate_times, vw_errors_interpolated[:, i], label='Velocity Error Interpolated')
    #     axes[i].set_ylabel(f'Velocity Error {i+1} (m/s)')
    #     axes[i].legend()
    # axes[-1].set_xlabel('Time (s)')
    # plt.suptitle('Velocity Errors')
    # plt.show()

    if save:
     
        save_str = f'ekf_gps_only'
        save_dir = os.path.join(output_dir, sim_subdir, save_str)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_dict = {
            'position_estimates': position_estimates[:, 1:],
            'position_covariances': position_covariances if save_covariances else None,
            'velocity_estimates': vw_estimates_interpolated,
            'velocity_covariances': velocity_covariances if save_covariances else None,
            'clock_bias_estimates': clock_bias_estimates[:, 1],
            'clock_drift_estimates': clock_drift_estimates[:, 1],
            'pw_errors': pw_errors,
            'vw_errors': vw_errors_interpolated,
            'true_positions': matched_truth_pos,
            'true_velocities': matched_truth_vel,
            'times': est_times, 
            'eTw': truth_eTw,
        }


        np.savez(os.path.join(save_dir, 'results.npz'), **save_dict)

    velocity_error_covariances = np.var(vw_errors_interpolated, axis=0)
    # velocity_2D_error_covariance = np.var(np.hstack((vw_errors_interpolated[:, 0], vw_errors_interpolated[:, 1])), axis=0)
    # velocity_vertical_error_covariance = np.var(vw_errors_interpolated[:, 2], axis=0)
    velocirt_2D_error_rmse = np.sqrt(np.mean(np.hstack((vw_errors_interpolated[:, 0], vw_errors_interpolated[:, 1]))**2))
    velocity_vertical_error_rmse = np.sqrt(np.mean(vw_errors_interpolated[:, 2]**2))
    velocity_2D_error_covariance = velocirt_2D_error_rmse**2
    velocity_vertical_error_covariance = velocity_vertical_error_rmse**2
    position_error_covariances = np.var(pw_errors, axis=0)
    position_2D_error_rmse = np.sqrt(np.mean(np.hstack((pw_errors[:, 0], pw_errors[:, 1]))**2))
    position_vertical_error_rmse = np.sqrt(np.mean(pw_errors[:, 2]**2))
    return {
        'estimate_times': est_times,
        'pw_estimates': position_estimates,
        'position_covariances': np.array([np.diag([position_2D_error_rmse**2, position_2D_error_rmse**2, position_vertical_error_rmse**2]) for _ in range(len(est_times))]) if save_covariances else None,
        'velocity_covariances': np.array([np.diag([velocity_2D_error_covariance, velocity_2D_error_covariance, velocity_vertical_error_covariance]) for _ in range(len(est_times))]) if save_covariances else None,
        'vw_estimates': np.hstack((est_times.reshape(-1, 1), vw_estimates_interpolated)),
        'clock_bias_estimates': clock_bias_estimates,
        'clock_drift_estimates': clock_drift_estimates,
        'pw_errors': pw_errors,
        'vw_errors': vw_errors_interpolated,
        'save_path': os.path.join(save_dir, 'results.npz') if save else None
    }

if __name__ == "__main__":
    datasets = [
        'lasso/tactical_grade',
    ]

    for dataset in datasets:
        # ekf_estimation_gps_imu(
        #     dataset,
        #     save = True,
        #     true_lever_arm=np.zeros(3),
        #     lever_arm_init = np.zeros(3),
        #     lever_arm_prior_var=1.0**2,
        #     unknown_lever_arm = False,
        #     init_rotation_error = np.eye(3)
        # )

        ekf_estimation_gps_only(
            dataset,
            save = True,
        )
        print(f"Completed EKF-GPS-only estimation for dataset: {dataset}")

    
