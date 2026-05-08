from util.measurement_deque import TimedMeasurementQueue
from graphs import RobustPIMLeverArmGraph, RobustPIMSE23GraphSliding
from util.constants import wei, I3
from util import config, gnss, pim, rcf, weight_from_covariance, imu, so3
from optimize import GN_opt_damped
from states import AccelBiasState, GyroBiasState, R3State, FloatState, SO3State

import numpy as np
import r3f
from util.data_management import import_gps_data
import os
import time
import numpy as np
from tqdm import tqdm
import yaml


def sliding_optimization_gps_imu(
        sim_subdir='lasso/navigation_grade',
        window_dur=30.0,
        pr_weight_function=rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight),
        tdcp_weight_function=rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight),
        pim_weight_function=rcf.gen_partial_robust_weight_function(rcf.l2_sqrt_weight),
        tdcp_cov_scale=1.0,
        pseudorange_cov_scale=1.0,
        accel_cov_scale=1.0,
        gyro_cov_scale=1.0,
        save=True,
        save_covariances=False,
        true_lever_arm: np.ndarray = np.zeros(3),
        lever_arm_init: np.ndarray | None = np.zeros(3),
        lever_arm_prior_var: None | float = None,
        output_dir: str = 'results',
        pim_interval: float = None,
        gps_data_file: str = None,
        unknown_lever_arm: bool = True,
        init_rotation_error: np.ndarray = np.eye(3),
        initial_position_error: np.ndarray = np.zeros(3)):
    """Perform sliding window optimization with unknown lever arm estimation.

    This function estimates the lever arm between the IMU and GNSS antenna
    along with the vehicle pose, velocity, and biases using a factor graph
    approach with a sliding window.

    Inputs:
        sim_subdir: Subdirectory within 'sim' containing the simulation data.
        window_dur: Duration of the sliding window in seconds.
        pr_weight_function: Robust weight function for pseudorange factors.
        tdcp_weight_function: Robust weight function for TDCP factors.
        pim_weight_function: Robust weight function for PIM factors.
        tdcp_cov_scale: Scale factor for TDCP covariance.
        pseudorange_cov_scale: Scale factor for pseudorange covariance.
        accel_cov_scale: Scale factor for accelerometer noise covariance.
        gyro_cov_scale: Scale factor for gyroscope noise covariance.
        save: Whether to save the results.
        save_covariances: Whether to save state covariances.
        true_lever_arm: True lever arm value for error calculation.
        lever_arm_init: Initial guess for the lever arm.
        lever_arm_prior_var: Prior variance for the lever arm (m^2).
        output_dir: Directory to save results.
        pim_interval: Interval for PIM factors.
        gps_data_file: Optional path to a specific GPS .npz file to use.
        unknown_lever_arm: Whether to estimate the lever arm or keep it fixed.
        init_rotation_error: Initial rotation error to apply to the attitude.
        initial_position_error: Initial position error to apply to the position state and prior.

    Returns:
        Dictionary containing estimation errors.

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
    accel_bias_init_var = config.sim['imu'][imu_sensor]['accelerometer_bias_cov']

    gyro_bias_init = true_gyro_biases[0]
    gyro_bias_init_var = config.sim['imu'][imu_sensor]['gyro_bias_cov']

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
    if pim_interval is not None and pim_interval > 0:
        start_t = t_gps[0]
        end_t = t_gps[-1]
        interp_times = np.arange(start_t, end_t, pim_interval)
        all_times = np.unique(np.concatenate((all_times, interp_times)))
        all_times = np.sort(all_times)
    
    # helper for keys
    def time_to_key(t, prefix):
        
        if stationary_period is not None and t <= stationary_period and prefix in ['p','R','v']:
            return f'{prefix}{stationary_period:.5f}'
        
        return f'{prefix}{t:.5f}'
        
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
    
    
    # Compute initial truth values for the first window later.


    # Create the graph
    if unknown_lever_arm:
        graph = RobustPIMLeverArmGraph(
            Tew=truth_eTw,
            pr_weight_function=pr_weight_function,
            tdcp_weight_function=tdcp_weight_function,
            pim_weight_function=pim_weight_function
        )
        # Add the lever arm state (static, not per-timestep)
        graph.add_state(
            key='lb',
            state=R3State(
                time=0.0,  # Static state, time is arbitrary
                value=lever_arm_init.copy() if lever_arm_init is not None else np.zeros(3)
            )
        )
    else:
        graph = RobustPIMSE23GraphSliding(
            Tew=truth_eTw,
            lb=lever_arm_init.copy() if lever_arm_init is not None else np.zeros(3),
            pr_weight_function=pr_weight_function,
            tdcp_weight_function=tdcp_weight_function,
            pim_weight_function=pim_weight_function
        )

    # Prepare the first window
    stop_time_init = t_gps[0] + window_dur
    t_state_window = all_times[(all_times >= 0) & (all_times <= stop_time_init)]
    
    pr_window = pr_deque.popleft_through(stop_time_init)
    tdcp_window = tdcp_deque.popleft_through(stop_time_init)
    imu_window = imu_deque.popleft_through(stop_time_init)

    time_states_dict = {}

    # Get truth for init window
    window_truth_pos = get_truth_at_t(t_state_window, truth_times, truth_pw)
    window_truth_att = np.array([r3f.rpy_to_dcm(rpy).T for rpy in get_truth_at_t(t_state_window, truth_times, truth_rpy)])
    window_truth_vel = get_truth_at_t(t_state_window, truth_times, truth_vw)

    ## Add states for the first window

    # Add stationary states and prior factors

    graph.add_state(
        key = time_to_key(0.0, 'p'),
        state = R3State(
            time = 0.0,
            value = init_p
        )
    )
    graph.add_prior_r3_factor(
        time = stationary_period if stationary_period is not None else 0.0,
        prior_value = init_p,
        state_key = time_to_key(0.0, 'p'),
        weight = weight_from_covariance(config.sim['prior_params']['prior_pos_var']*I3)
    )
    graph.add_state(
        key = time_to_key(0.0, 'R'),
        state = SO3State(
            time = 0.0,
            value = init_R
        )
    )
    graph.add_prior_so3_factor(
        time = stationary_period if stationary_period is not None else 0.0,
        prior_value = init_R,
        state_key = time_to_key(0.0, 'R'),
        weight = weight_from_covariance(config.sim['prior_params']['prior_att_var']*I3)
    )

    graph.add_state(
        key = time_to_key(0.0, 'v'),
        state = R3State(
            time = stationary_period if stationary_period is not None else 0.0,
            value = np.zeros(3) if stationary_period is not None else truth_vw[0]
        )
    )
    graph.add_prior_r3_factor(
        time = stationary_period if stationary_period is not None else 0.0,
        prior_value = np.zeros(3) if stationary_period is not None else truth_vw[0],
        state_key = time_to_key(0.0, 'v'),
        weight = weight_from_covariance(config.sim['prior_params']['prior_vel_var']*I3)
    )

    # Add remaining states in the window
    for idx, time_i in enumerate(t_state_window):
        in_stationary = False if stationary_period is None else (time_i <= stationary_period)
        if not in_stationary and time_i != 0.0:   
            graph.add_state(
                key = time_to_key(time_i, 'p'),
                state = R3State(
                    time = time_i,
                    value = window_truth_pos[idx],
                )
            )
            graph.add_state(
                key = time_to_key(time_i, 'R'),
                state = SO3State(
                    time = time_i,
                    value = window_truth_att[idx],
                )
            )

            graph.add_state(
                key = time_to_key(time_i, 'v'),
                state = R3State(
                    time = time_i,
                    value = window_truth_vel[idx]
                )
            )

        graph.add_state(
            key = time_to_key(time_i, 'a'),
            state = AccelBiasState(
                time = time_i,
                value = get_truth_at_t(np.array([time_i]), true_bias_times, true_accel_biases)[0]
            )
        )

        graph.add_state(
            key = time_to_key(time_i, 'g'),
            state = GyroBiasState(
                time = time_i,
                value = get_truth_at_t(np.array([time_i]), true_bias_times, true_gyro_biases)[0]
            )
        )        

        graph.add_state(
            key = time_to_key(time_i, 'c'),
            state = FloatState(
                time = time_i,
                value = 0.0
            )
        )
        # Clock Drift State
        graph.add_state(
            key = time_to_key(time_i, 'd'),
            state = FloatState(
                time = time_i,
                value = 0.0
            )
        )

        time_states_dict[time_i] = [
            time_to_key(time_i, 'p'),
            time_to_key(time_i, 'R'),
            time_to_key(time_i, 'c'),
            time_to_key(time_i, 'd'),
            time_to_key(time_i, 'v'),
            time_to_key(time_i, 'a'),
            time_to_key(time_i, 'g'),
        ]

    

    # Add pseudorange factors with unknown lever arm
    for pr_msmt in pr_window:
        t_msmt = pr_msmt[0]
        if unknown_lever_arm:
             graph.add_pseudorange_factor(
                meas=pr_msmt[1],
                time=t_msmt,
                # pnom=true_positions[0],
                # p_key=f'p{int(pr_msmt[0])}',
                p_key = time_to_key(t_msmt, 'p'),
                cb_key= time_to_key(t_msmt, 'c'),
                R_key = time_to_key(t_msmt, 'R'),
                sat_xyz=pr_msmt[4:7],
                lb_key='lb',
                weight=1 / (pr_msmt[2] * np.sqrt(pseudorange_cov_scale)),
                prn=pr_msmt[3]
            )
        else:
             graph.add_pseudorange_factor(
                meas=pr_msmt[1],
                time=t_msmt,
                # pnom=true_positions[0],
                # p_key=f'p{int(pr_msmt[0])}',
                p_key = time_to_key(t_msmt, 'p'),
                cb_key= time_to_key(t_msmt, 'c'),
                R_key = time_to_key(t_msmt, 'R'),
                sat_xyz=pr_msmt[4:7],
                lb=lever_arm_init.copy() if lever_arm_init is not None else np.zeros(3),
                weight=1 / (pr_msmt[2] * np.sqrt(pseudorange_cov_scale)),
                prn=pr_msmt[3]
            )

    # Add TDCP factors with unknown lever arm
    for tdcp_msmt in tdcp_window:
        if unknown_lever_arm:
             graph.add_tdcp_factor(
                meas=tdcp_msmt[2],
                # pnom=true_positions[0],
                time0=tdcp_msmt[0],
                time1=tdcp_msmt[1],
                p0_key=time_to_key(tdcp_msmt[0], 'p'),
                cb0_key=time_to_key(tdcp_msmt[0], 'c'),
                R0_key=time_to_key(tdcp_msmt[0], 'R'),
                sat_xyz0=tdcp_msmt[5:8],
                p1_key=time_to_key(tdcp_msmt[1], 'p'),
                cb1_key=time_to_key(tdcp_msmt[1], 'c'),
                R1_key=time_to_key(tdcp_msmt[1], 'R'),
                sat_xyz1=tdcp_msmt[8:11],
                weight=1 / (tdcp_msmt[3] * np.sqrt(tdcp_cov_scale)),
                lb_key='lb',
                prn=tdcp_msmt[4]
            )
        else:
             graph.add_tdcp_factor(
                meas=tdcp_msmt[2],
                # pnom=true_positions[0],
                time0=tdcp_msmt[0],
                time1=tdcp_msmt[1],
                p0_key=time_to_key(tdcp_msmt[0], 'p'),
                cb0_key=time_to_key(tdcp_msmt[0], 'c'),
                R0_key=time_to_key(tdcp_msmt[0], 'R'),
                sat_xyz0=tdcp_msmt[5:8],
                p1_key=time_to_key(tdcp_msmt[1], 'p'),
                cb1_key=time_to_key(tdcp_msmt[1], 'c'),
                R1_key=time_to_key(tdcp_msmt[1], 'R'),
                sat_xyz1=tdcp_msmt[8:11],
                weight=1 / (tdcp_msmt[3] * np.sqrt(tdcp_cov_scale)),
                lb=lever_arm_init if lever_arm_init is not None else np.zeros(3),
                prn=tdcp_msmt[4]
            )

    # Add bias transition and PIM factors for the whole window
    gravities =  np.array([truth_eTw[:3,:3].T@imu.grav_somigliana_pe(truth_eTw[:3,:3]@pos + truth_eTw[:3,3]) for pos in window_truth_pos])
    imu_msmts_by_state_times: dict[tuple[float,float],list[pim.IMUMeasurement]]= {}
        
    # Look through all the state times and get the corresponding IMU measurements for each interval between state times
    for idx in range(len(t_state_window)-1):
        time_i = t_state_window[idx]
        time_j = t_state_window[idx+1]
        imu_data_interval  = imu_window[(imu_window[:,0] >= time_i) & (imu_window[:,0] <= time_j)]
        imu_msmts_interval = [pim.IMUMeasurement(
            timestamp=msmt[0],
            accel=msmt[1:4],
            gyro=msmt[4:7]
        ) for msmt in imu_data_interval]
        imu_msmts_by_state_times[(time_i, time_j)] = imu_msmts_interval

    for idx in range(len(t_state_window)-1):
        time_i = t_state_window[idx]
        time_j = t_state_window[idx+1]
        
        dt_val = time_j - time_i
        
        graph.add_clock_transition_factor(
            bias_time_0 = time_i,
            clock_bias_0_key=time_to_key(time_i, 'c'),
            clock_bias_1_key=time_to_key(time_j, 'c'),
            delta_t = dt_val,
            clock_drift_0_key=time_to_key(time_i, 'd'),
            clock_drift_1_key=time_to_key(time_j, 'd'),
            weight = weight_from_covariance(gnss.clock_transition_variance(clock_noise_model, dt_val))
        )

        graph.add_imu_bias_transition_factor(
            bias_time_0 = time_i,
            imu_bias_0_key = time_to_key(time_i, 'a'),
            imu_bias_1_key = time_to_key(time_j, 'a'),
            dt = dt_val,
            weight = weight_from_covariance(I3*(imu_noise_model['accel_bias_driving_std_func'](dt_val)**2)),
            tau = config.sim['imu'][imu_sensor]['time_constant_accel_bias_s']
        )

        graph.add_imu_bias_transition_factor(
            bias_time_0 = time_i,
            imu_bias_0_key = time_to_key(time_i, 'g'),
            imu_bias_1_key = time_to_key(time_j, 'g'),
            dt = dt_val,
            weight = weight_from_covariance(I3*(imu_noise_model['gyro_bias_driving_std_func'](dt_val)**2)),
            tau = config.sim['imu'][imu_sensor]['time_constant_gyro_bias_s']
        )
        
        graph.add_pim_factor(
            time0 = time_i,
            Ri_key = time_to_key(time_i, 'R'),
            vi_key = time_to_key(time_i, 'v'),
            pi_key = time_to_key(time_i, 'p'),
            Rj_key = time_to_key(time_j, 'R'),
            vj_key = time_to_key(time_j, 'v'),
            pj_key = time_to_key(time_j, 'p'),
            ba_key = time_to_key(time_i, 'a'),
            bg_key = time_to_key(time_i, 'g'),
            imu_msmts = imu_msmts_by_state_times[(time_i, time_j)],
            accel_white_noise = I3*(imu_noise_model['accel_white_noise_sigma']**2)*accel_cov_scale,
            gyro_white_noise = I3*(imu_noise_model['gyro_white_noise_sigma']**2)*gyro_cov_scale,
            gravity = gravities[idx]
        )

   
    # Add prior factors - IMU biases
    graph.add_prior_r3_factor(
        time = t_state_window[0],
        prior_value = accel_bias_init,
        state_key = time_to_key(t_state_window[0], 'a'),
        weight = weight_from_covariance(accel_bias_init_var)
    )

    graph.add_prior_r3_factor(
        time = t_state_window[0],
        prior_value = gyro_bias_init,
        state_key = time_to_key(t_state_window[0], 'g'),
        weight = weight_from_covariance(gyro_bias_init_var)
    )

    # Add prior factor - Clock bias
    graph.add_prior_scalar_factor(
        time = t_state_window[0],
        prior_value = 0.0,
        state_key = time_to_key(t_state_window[0], 'c'),
        weight = weight_from_covariance(config.sim['prior_params']['prior_clock_bias_var'])
    )

    # Add lever arm prior factor
    if unknown_lever_arm and lever_arm_prior_var is not None and lever_arm_init is not None:
        graph.add_prior_r3_factor(
            time=0.0,
            prior_value=lever_arm_init,
            state_key='lb',
            weight=weight_from_covariance(np.atleast_2d(lever_arm_prior_var) if isinstance(lever_arm_prior_var, np.ndarray) else lever_arm_prior_var * I3)
        )

    # def error_from_factor(factor: Factor):
    #     return factor.error_func(state_tuple = graph.state_tuple_from_keys(factor.states))
    # def rcf_error_from_robust_factor(robust_factor: RobustFactor):
    #     error_from_factor_val = error_from_factor(robust_factor)
    #     return robust_factor.robust_weight_func(error_from_factor_val)* error_from_factor_val
    # factors = graph.factors
    # pim_factors = [fac for fac in factors if isinstance(fac, PIMFactor)]
    # pim_errors = np.array([pim_fac.error_func(state_tuple = graph.state_tuple_from_keys(pim_fac.states)) for pim_fac in pim_factors])

    # pr_factors = [factor for factor in factors if isinstance(factor, RobustPseudorangeFactor)]
    # pr_errors = np.array([pr_fac.error_func(state_tuple = graph.state_tuple_from_keys(pr_fac.states)) for pr_fac in pr_factors])

    # tdcp_factors = [factor for factor in factors if isinstance(factor, RobustTDCPFactor)]
    # tdcp_errors = np.array([tdcp_fac.error_func(state_tuple = graph.state_tuple_from_keys(tdcp_fac.states)) for tdcp_fac in tdcp_factors])

    # from matplotlib import pyplot as plt
    # plt.figure()
    # for i in range(9):
    #     plt.plot(pim_errors[:,i], label=f'pim_err_{i}')
    # plt.legend()
    # plt.figure()
    # plt.plot(pr_errors, label='pr_errors')
    # plt.legend()
    # plt.figure()
    # plt.plot(tdcp_errors, label='tdcp_errors')
    # plt.legend()



    # Optimize the first window
    GN_opt_damped(
        graph,
        min_step_factor_power_2=3,
        rel_decrease_tol=1e-6,
        max_iter=10,
        print_iterations=False
    )
    
    # Store old estimates, marginalize
    position_estimates = TimedMeasurementQueue()
    position_covariances = TimedMeasurementQueue()
    velocity_estimates = TimedMeasurementQueue()
    velocity_covariances = TimedMeasurementQueue()
    rpy_estimates = TimedMeasurementQueue()
    accel_bias_estimates = TimedMeasurementQueue()
    gyro_bias_estimates = TimedMeasurementQueue()
    clock_bias_estimates = TimedMeasurementQueue()
    clock_drift_estimates = TimedMeasurementQueue()
    lever_arm_estimates = TimedMeasurementQueue()
    lever_arm_covariances = TimedMeasurementQueue()

    def do_marginalize_step():
        
        # Drop oldest state
        time_to_drop = t_state_window[0]
        position_estimates.extend([
            np.concatenate(([float(time_to_drop)], graph.state_dict[time_to_key(time_to_drop, 'p')].value))
        ])
        if save_covariances:
            position_covariances.extend([
                graph.marginal_covariance_of_states([time_to_key(time_to_drop, 'p')])
            ])
        velocity_estimates.extend([
            np.concatenate(([float(time_to_drop)], graph.state_dict[time_to_key(time_to_drop, 'v')].value))
        ])
        if save_covariances:
            velocity_covariances.extend([
                graph.marginal_covariance_of_states([time_to_key(time_to_drop, 'v')])
            ])
        rpy_estimates.extend([
            np.concatenate(([float(time_to_drop)], r3f.dcm_to_rpy(graph.state_dict[time_to_key(time_to_drop, 'R')].value.T)))
        ])
        accel_bias_estimates.extend([
            np.concatenate(([float(time_to_drop)], graph.state_dict[time_to_key(time_to_drop, 'a')].value))
        ])
        gyro_bias_estimates.extend([
            np.concatenate(([float(time_to_drop)], graph.state_dict[time_to_key(time_to_drop, 'g')].value))
        ])
        clock_bias_estimates.extend([
            [float(time_to_drop), graph.state_dict[time_to_key(time_to_drop, 'c')].value]
        ])
        clock_drift_estimates.extend([
            [float(time_to_drop), graph.state_dict[time_to_key(time_to_drop, 'd')].value]
        ])
        if unknown_lever_arm:
            lever_arm_estimates.extend([
                np.concatenate(([float(time_to_drop)], graph.state_dict['lb'].value))
            ])
            if save_covariances:
                lever_arm_covariances.extend([
                    graph.marginal_covariance_of_states(['lb'])
                ])
        
        states_to_drop = time_states_dict[time_to_drop]
        in_stationary = False if stationary_period is None else (time_to_drop <= stationary_period)
        if in_stationary:
            # Keep stationary states
            states_to_drop = [s for s in states_to_drop if s not in [
                time_to_key(stationary_period, 'p'),
                time_to_key(stationary_period, 'R'),
                time_to_key(stationary_period, 'v')]]
        time_states_dict.pop(time_to_drop, None)
    
        if len(t_state_window) > 1:
             next_t = t_state_window[1]
             graph.marginalize_states(states_to_drop, marginalization_time=next_t)
        
        # We need to remove from t_state_window in the outer scope
        # Return true if done
        return time_to_drop

    do_marginalize_step()
    t_state_window = t_state_window[1:]

    sliding_window_opt_times = []
    
    # Progressively slide the window
    remaining_gps = t_gps[t_gps > t_state_window[-1]]
    last_imu_msmt = imu_window[-1]

    for next_gps in tqdm(remaining_gps, desc="Sliding Window Progress"):
        iter_start_time = time.time()

        # Identify all new states to add (up to next_gps)
        last_state_time = t_state_window[-1]
        
        new_times = all_times[(all_times > last_state_time) & (all_times <= next_gps)]
        
        if len(new_times) == 0:
            continue
            
        new_pr_msmts = pr_deque.popleft_through(next_gps)
        new_tdcp_msmts = tdcp_deque.popleft_through(next_gps)
        new_imu_msmts = np.vstack((last_imu_msmt.reshape(1,-1),imu_deque.popleft_through(next_gps) ))
        last_imu_msmt = new_imu_msmts[-1]
        
        chain_times = np.concatenate(([last_state_time], new_times))
        
        last_accel_bias = graph.state_dict[time_to_key(last_state_time, 'a')].value
        last_gyro_bias = graph.state_dict[time_to_key(last_state_time, 'g')].value
        
        # Gravity
        chain_truth_pos = get_truth_at_t(chain_times[:-1], truth_times, truth_pw)
        chain_gravities = np.array([eRw.T@imu.grav_somigliana_pe(eRw@pos + truth_eTw[:3,3]) for pos in chain_truth_pos])

        # Prepare for PIM
        imu_msmts_by_state_times: dict[tuple[float,float],list[pim.IMUMeasurement]]= {}
        pim_times = np.hstack([last_state_time,new_times])
        for idx in range(len(pim_times)-1):
            time_i = pim_times[idx]
            time_j = pim_times[idx+1]
            imu_data_interval  = new_imu_msmts[(new_imu_msmts[:,0] >= time_i) & (new_imu_msmts[:,0] <= time_j)]
            imu_msmts_interval = [pim.IMUMeasurement(
                timestamp=msmt[0],
                accel=msmt[1:4],
                gyro=msmt[4:7]
            ) for msmt in imu_data_interval]
            imu_msmts_by_state_times[(time_i, time_j)] = imu_msmts_interval

        for i, t_curr in enumerate(new_times):
            t_prev = chain_times[i] 
            dt = t_curr - t_prev
            
            # Predict
            prev_pos = graph.state_dict[time_to_key(t_prev, 'p')].value
            prev_vel = graph.state_dict[time_to_key(t_prev, 'v')].value
            prev_att = graph.state_dict[time_to_key(t_prev, 'R')].value
            prev_accel_bias = graph.state_dict[time_to_key(t_prev, 'a')].value
            prev_gyro_bias = graph.state_dict[time_to_key(t_prev, 'g')].value
            prev_clock_bias = graph.state_dict[time_to_key(t_prev, 'c')].value
            prev_clock_drift = graph.state_dict[time_to_key(t_prev, 'd')].value


            new_accel_bias = np.exp(-dt/config.sim['imu'][imu_sensor]['time_constant_accel_bias_s'])*(prev_accel_bias)
            new_gyro_bias = np.exp(-dt/config.sim['imu'][imu_sensor]['time_constant_gyro_bias_s'])*(prev_gyro_bias)
            pim_data_interval: pim.PIMData = pim.preintegrate_imu_one_interval(
                imu_measurements = imu_msmts_by_state_times[(t_prev, t_curr)],
                accel_bias = new_accel_bias,
                gyro_bias = new_gyro_bias,
                gravity = chain_gravities[i],
                omega = eRw.T@wei,
                accel_white_noise=I3*(imu_noise_model['accel_white_noise_sigma']**2)*accel_cov_scale,
                gyro_white_noise=I3*(imu_noise_model['gyro_white_noise_sigma']**2)*gyro_cov_scale,
            )

            accel_msmts_interval = np.array([msmt.accel for msmt in imu_msmts_by_state_times[(t_prev, t_curr)]])
            gyro_msmts_interval = np.array([msmt.gyro for msmt in imu_msmts_by_state_times[(t_prev, t_curr)]])

            ups_hat_curr = pim_data_interval.UpsilonMsmt
            ups_hat_curr_R = ups_hat_curr[:3,:3]
            ups_hat_curr_v = ups_hat_curr[:3,3]
            ups_hat_curr_p = ups_hat_curr[:3,4]
            gammaR_curr = pim_data_interval.GammaR
            gammaV_curr = pim_data_interval.GammaV
            gammaP_curr = pim_data_interval.GammaP
            new_att = gammaR_curr@prev_att @ ups_hat_curr_R
            new_pos = gammaR_curr@(prev_att @ ups_hat_curr_p + (prev_vel+Omega@prev_pos)*dt+prev_pos)+ gammaP_curr 
            new_vel = gammaR_curr@(prev_att @ ups_hat_curr_v+prev_vel+Omega@prev_pos)-Omega@new_pos+gammaV_curr

            new_clock_bias = prev_clock_bias + prev_clock_drift*dt

            in_stationary = False if stationary_period is None else (t_curr <= stationary_period)
            if not in_stationary:
                graph.add_state(key = time_to_key(t_curr, 'p'), state = R3State(time = t_curr, value = new_pos))
                graph.add_state(key = time_to_key(t_curr, 'v'), state = R3State(time = t_curr, value = new_vel))
                graph.add_state(key = time_to_key(t_curr, 'R'), state = SO3State(time = t_curr, value = new_att))
            graph.add_state(key = time_to_key(t_curr, 'c'), state = FloatState(time = t_curr, value = new_clock_bias))
            graph.add_state(key = time_to_key(t_curr, 'd'), state = FloatState(time = t_curr, value = prev_clock_drift))
            graph.add_state(key = time_to_key(t_curr, 'a'), state = AccelBiasState(time = t_curr, value = new_accel_bias))
            graph.add_state(key = time_to_key(t_curr, 'g'), state = GyroBiasState(time = t_curr, value = new_gyro_bias))
            
            time_states_dict[t_curr] = [time_to_key(t_curr, k) for k in ['p','R','c','v','a','g','d']]
            
            graph.add_clock_transition_factor(
                bias_time_0 = t_prev,
                clock_bias_0_key=time_to_key(t_prev, 'c'),
                clock_bias_1_key=time_to_key(t_curr, 'c'),
                delta_t = dt,
                clock_drift_0_key=time_to_key(t_prev, 'd'),
                clock_drift_1_key=time_to_key(t_curr, 'd'),
                weight = weight_from_covariance(gnss.clock_transition_variance(clock_noise_model, dt))
            )

            
            graph.add_imu_bias_transition_factor(
                bias_time_0 = t_prev,
                imu_bias_0_key = time_to_key(t_prev, 'a'),
                imu_bias_1_key = time_to_key(t_curr, 'a'),
                dt = dt,
                weight = weight_from_covariance(I3*(imu_noise_model['accel_bias_driving_std_func'](dt)**2)),
                tau = config.sim['imu'][imu_sensor]['time_constant_accel_bias_s']
            )

            graph.add_imu_bias_transition_factor(
                bias_time_0 = t_prev,
                imu_bias_0_key = time_to_key(t_prev, 'g'),
                imu_bias_1_key = time_to_key(t_curr, 'g'),
                dt = dt,
                weight = weight_from_covariance(I3*(imu_noise_model['gyro_bias_driving_std_func'](dt)**2)),
                tau = config.sim['imu'][imu_sensor]['time_constant_gyro_bias_s']
            )
            
            graph.add_pim_factor(
                time0 = t_prev,
                Ri_key = time_to_key(t_prev, 'R'),
                vi_key = time_to_key(t_prev, 'v'),
                pi_key = time_to_key(t_prev, 'p'),
                Rj_key = time_to_key(t_curr, 'R'),
                vj_key = time_to_key(t_curr, 'v'),
                pj_key = time_to_key(t_curr, 'p'),
                ba_key = time_to_key(t_prev, 'a'),
                bg_key = time_to_key(t_prev, 'g'),
                imu_msmts=imu_msmts_by_state_times[(t_prev, t_curr)],
                accel_white_noise=I3*(imu_noise_model['accel_white_noise_sigma']**2)*accel_cov_scale,
                gyro_white_noise=I3*(imu_noise_model['gyro_white_noise_sigma']**2)*gyro_cov_scale,
                gravity = chain_gravities[i],
            )
            
            if new_pr_msmts is not None:
                relevant_pr = [pr for pr in new_pr_msmts if abs(pr[0] - t_curr) < 1e-6]
                for pr_msmt in relevant_pr:
                    if unknown_lever_arm:
                        graph.add_pseudorange_factor(
                            meas=pr_msmt[1],
                            time=pr_msmt[0],
                            # pnom=next_pos_init, 
                            p_key=time_to_key(t_curr, 'p'),
                            cb_key=time_to_key(t_curr, 'c'),
                            R_key=time_to_key(t_curr, 'R'),
                            sat_xyz=pr_msmt[4:7],
                            prn=int(pr_msmt[3]),
                            weight=1/(pr_msmt[2]*np.sqrt(pseudorange_cov_scale)),
                            lb_key='lb'
                        )
                    else:
                         graph.add_pseudorange_factor(
                            meas=pr_msmt[1],
                            time=pr_msmt[0],
                            # pnom=next_pos_init, 
                            p_key=time_to_key(t_curr, 'p'),
                            cb_key=time_to_key(t_curr, 'c'),
                            R_key=time_to_key(t_curr, 'R'),
                            sat_xyz=pr_msmt[4:7],
                            prn=int(pr_msmt[3]),
                            weight=1/(pr_msmt[2]*np.sqrt(pseudorange_cov_scale)),
                            lb=lever_arm_init if lever_arm_init is not None else np.zeros(3)
                        )
            
            if new_tdcp_msmts is not None:
                relevant_tdcp = [td for td in new_tdcp_msmts if abs(td[1] - t_curr) < 1e-6]
                for tdcp_msmt in relevant_tdcp:
                    if time_to_key(tdcp_msmt[0], 'p') in graph.state_dict:
                        if unknown_lever_arm:
                            graph.add_tdcp_factor(
                                meas=tdcp_msmt[2],
                                time0=tdcp_msmt[0],
                                time1=tdcp_msmt[1],
                                p0_key=time_to_key(tdcp_msmt[0], 'p'),
                                p1_key=time_to_key(tdcp_msmt[1], 'p'),
                                cb0_key=time_to_key(tdcp_msmt[0], 'c'),
                                cb1_key=time_to_key(tdcp_msmt[1], 'c'),
                                R0_key=time_to_key(tdcp_msmt[0], 'R'),
                                R1_key=time_to_key(tdcp_msmt[1], 'R'),
                                sat_xyz0=tdcp_msmt[5:8],
                                sat_xyz1=tdcp_msmt[8:11],
                                prn=int(tdcp_msmt[4]),
                                weight=1/(tdcp_msmt[3]*np.sqrt(tdcp_cov_scale)),
                                lb_key='lb'
                            )
                        else:
                            graph.add_tdcp_factor(
                                meas=tdcp_msmt[2],
                                time0=tdcp_msmt[0],
                                time1=tdcp_msmt[1],
                                p0_key=time_to_key(tdcp_msmt[0], 'p'),
                                p1_key=time_to_key(tdcp_msmt[1], 'p'),
                                cb0_key=time_to_key(tdcp_msmt[0], 'c'),
                                cb1_key=time_to_key(tdcp_msmt[1], 'c'),
                                R0_key=time_to_key(tdcp_msmt[0], 'R'),
                                R1_key=time_to_key(tdcp_msmt[1], 'R'),
                                sat_xyz0=tdcp_msmt[5:8],
                                sat_xyz1=tdcp_msmt[8:11],
                                prn=int(tdcp_msmt[4]),
                                weight=1/(tdcp_msmt[3]*np.sqrt(tdcp_cov_scale)),
                                lb=lever_arm_init if lever_arm_init is not None else np.zeros(3)
                            )
            
            t_state_window = np.append(t_state_window, t_curr)

        GN_opt_damped(graph, min_step_factor_power_2 = 3, rel_decrease_tol=1e-6, print_iterations=False)

        curr_time = t_state_window[-1]
        cutoff_time = curr_time - window_dur
        
        while len(t_state_window) > 0 and t_state_window[0] < cutoff_time:
            do_marginalize_step()
            t_state_window = t_state_window[1:]
        
        iter_end_time = time.time()
        sliding_window_opt_times.append(iter_end_time - iter_start_time)

    print(f"Sliding window optimization took {np.sum(sliding_window_opt_times)} seconds total.")
    print(f"Average time per sliding window optimization step: {np.mean(sliding_window_opt_times)} seconds.")

    # Add all remaining states to the estimate deques
    graph_state_keys = list(graph.state_dict.keys())
    
    def get_time_from_key(k):
        
        return float(k[1:])

    accel_bias_keys_sorted = sorted([key for key in graph_state_keys if key.startswith('a')], key=get_time_from_key)
    for key in accel_bias_keys_sorted:
        t_v = get_time_from_key(key)
        accel_bias_estimates.extend([np.concatenate(([t_v], graph.state_dict[key].value))])
        gyro_bias_estimates.extend([np.concatenate(([t_v], graph.state_dict[time_to_key(t_v, 'g')].value))])
        clock_bias_estimates.extend([[t_v, graph.state_dict[time_to_key(t_v, 'c')].value]])
        clock_drift_estimates.extend([[t_v, graph.state_dict[time_to_key(t_v, 'd')].value]])
        position_estimates.extend([np.concatenate(([t_v], graph.state_dict[time_to_key(t_v, 'p')].value))])
        velocity_estimates.extend([np.concatenate(([t_v], graph.state_dict[time_to_key(t_v, 'v')].value))])
        rpy_estimates.extend([np.concatenate(([t_v], r3f.dcm_to_rpy(graph.state_dict[time_to_key(t_v, 'R')].value.T)))])
        if save_covariances:
            position_covariances.extend([
                graph.marginal_covariance_of_states([time_to_key(t_v, 'p')])
            ])
            velocity_covariances.extend([
                graph.marginal_covariance_of_states([time_to_key(t_v, 'v')])
            ])


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
        final_lever_arm_estimate = graph.state_dict['lb'].value
        final_lever_arm_error = true_lever_arm - final_lever_arm_estimate

        print(f"True lever arm: {true_lever_arm}")
        print(f"Final lever arm estimate: {final_lever_arm_estimate}")
        print(f"Final lever arm error: {final_lever_arm_error}")
        print(f"Final lever arm error norm: {np.linalg.norm(final_lever_arm_error):.4f} m")

    if save:
        # Save the results
        if unknown_lever_arm:
             prior_suffix = '_prior' if lever_arm_prior_var is not None else '_no_prior'
             lever_arm_str = f'unknown_lever_arm{prior_suffix}_true_{true_lever_arm[0]:.2f}_{true_lever_arm[1]:.2f}_{true_lever_arm[2]:.2f}'
        else:
            if lever_arm_init is not None:
                lever_arm_str = f'known_lever_arm_true_{true_lever_arm[0]:.2f}_{true_lever_arm[1]:.2f}_{true_lever_arm[2]:.2f}'
            else:
                lever_arm_str = f'ignored_lever_arm_true_{true_lever_arm[0]:.2f}_{true_lever_arm[1]:.2f}_{true_lever_arm[2]:.2f}'
        pim_str = f'pim_{pim_interval:.2f}' if pim_interval else 'pim_synced'
        save_dir = os.path.join(output_dir, sim_subdir, lever_arm_str, f'window_{window_dur}_{pim_str}')

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
            'lever_arm_init': lever_arm_init,
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
    datasets = [
        'lasso/navigation_grade',
        # 'figure_8/navigation_grade',
    ]
    time_durs = [
        30.0,
    ]
    for dataset in datasets:
        print(f'Segment: {dataset}')
        for dur in time_durs:
            sliding_optimization_gps_imu(
                dataset,
                window_dur=dur,
                save=True,
                true_lever_arm=np.array([0.0, 0.0, 0.0]), # Set to actual true lever arm
                lever_arm_init=np.array([0.0,0.0,0.0]),  
                lever_arm_prior_var=0.1,
                unknown_lever_arm=True,
                init_rotation_error=np.eye(3)
            )
    print("Sliding optimization with unknown lever arm completed.")
