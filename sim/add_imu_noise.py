import numpy as np

def add_imu_noise(times, fbbi, wbbi, sensor_config, init_accel_bias = np.zeros(3), init_gyro_bias = np.zeros(3), seed = 12112025):
    """Add white and FOGM noise to IMU measurements based on sensor configuration.

    Inputs:
    fbbi: (N, 3) array of specific force measurements [m/s^2]
    wbbi: (N, 3) array of angular rate measurements [rad/s]
    sensor_config: dictionary with keys
        'sampling_period': float, sampling period in seconds
        'vel_random_walk_m_s_sqrthr': float, velocity random walk in m/s/sqrt(hr)
        'accelerometer_bias_stability_mg': float, accelerometer bias stability in mg
        'angle_random_walk_deg_sqrthr': float, angle random walk in deg/sqrt(hr)
        'gyro_bias_stability_deg_hr': float, gyro bias stability in deg/hr
        'time_constant_gyro_bias_s': float, time constant for gyro bias in seconds
        'time_constant_accel_bias_s': float, time constant for accelerometer bias in seconds
    init_accel_bias: (3,) array, initial accelerometer bias
    init_gyro_bias: (3,) array, initial gyro bias

    Returns:
    noisy_fbbi: (N, 3) array of noisy specific force measurements
    noisy_wbbi: (N, 3) array of noisy angular rate measurements
    true_accel_bias: (N, 3) array of true accelerometer biases over time
    true_gyro_bias: (N, 3) array of true gyro biases over time    

    """
    ## Convert units
    dt = sensor_config['sampling_period']
    N = len(fbbi)
    
    # Convert accelerometer noise parameters
    # Velocity random walk: m/s/sqrt(hr) -> m/s^2/sqrt(Hz)
    accel_white_noise_std = sensor_config['vel_random_walk_m_s_sqrthr'] / 60.0  # m/s^2/sqrt(Hz)
    
    # Accelerometer bias stability: mg -> m/s^2
    accel_bias_std = sensor_config['accelerometer_bias_stability_mg'] * 9.80665 / 1000.0  # m/s^2
    
    # Convert gyroscope noise parameters
    # Angle random walk: deg/sqrt(hr) -> rad/s/sqrt(Hz)
    gyro_white_noise_std = np.deg2rad(sensor_config['angle_random_walk_deg_sqrthr']) / 60.0  # rad/s/sqrt(Hz)
    
    # Gyro bias stability: deg/hr -> rad/s
    gyro_bias_std = np.deg2rad(sensor_config['gyro_bias_stability_deg_hr']) / 3600.0  # rad/s
    
    # Time constants
    tau_accel = sensor_config['time_constant_accel_bias_s']
    tau_gyro = sensor_config['time_constant_gyro_bias_s']

    ## Generate white noise
    # White noise for accelerometer (m/s^2)
    np.random.seed(seed)
    accel_white_noise = np.random.randn(N, 3) * accel_white_noise_std / np.sqrt(dt) #[m/s^2]
    
    # White noise for gyroscope (rad/s)
    gyro_white_noise = np.random.randn(N, 3) * gyro_white_noise_std / np.sqrt(dt) #[rad/s]

    ## Generate FOGM bias
    # FOGM process: db/dt = -1/tau * b + w, where w is white noise
    # Discrete-time: b[k+1] = exp(-dt/tau) * b[k] + sqrt(2*sigma^2/tau * (1 - exp(-2*dt/tau))) * n[k]
    
    # Accelerometer bias
    true_accel_bias = np.zeros((N, 3))
    true_accel_bias[0] = init_accel_bias
    phi_accel = np.exp(-dt / tau_accel)
    Q_accel = accel_bias_std * np.sqrt((1 - phi_accel**2))
    
    for i in range(1, N):
        true_accel_bias[i] = phi_accel * true_accel_bias[i-1] + Q_accel * np.random.randn(3)
    
    # Gyroscope bias
    true_gyro_bias = np.zeros((N, 3))
    true_gyro_bias[0] = init_gyro_bias
    phi_gyro = np.exp(-dt / tau_gyro)
    Q_gyro = gyro_bias_std * np.sqrt((1 - phi_gyro**2))
    
    for i in range(1, N):
        true_gyro_bias[i] = phi_gyro * true_gyro_bias[i-1] + Q_gyro * np.random.randn(3)
    
    # Add noise and bias to measurements
    noisy_fbbi = fbbi + accel_white_noise + true_accel_bias
    noisy_wbbi = wbbi + gyro_white_noise + true_gyro_bias
    
    return noisy_fbbi, noisy_wbbi, true_accel_bias, true_gyro_bias

def noisy_imu_from_tpva_fr_file(tpva_fr_file, sensor, ouput_file = None, seed=12112025):
    
    from util import config
    sensor_config = config.sim['imu'][sensor]
    tpva_fr = np.load(tpva_fr_file)['tpva_fr'] # 0-time, 1-3 pos, 4-6 vel, 7-9 rpy, 10-12 fbbi, 13-15 wbbi
    fbbi =tpva_fr[:, 10:13]
    wbbi = tpva_fr[:, 13:16]

    fbbi_noisy, wbbi_noisy, accel_bias, gyro_bias = add_imu_noise(tpva_fr[:, 0], fbbi, wbbi, sensor_config, seed=seed)
    if ouput_file is not None:
        imu_data = np.hstack([tpva_fr[:,0].reshape(-1,1), fbbi_noisy, wbbi_noisy])
        np.savez(ouput_file, imu_data = imu_data, accel_bias=accel_bias, gyro_bias=gyro_bias)
        
    return imu_data, accel_bias, gyro_bias
 