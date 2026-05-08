
from util.constants import DEG_TO_RADIANS,  c
import numpy as np

## SIM DATA PARAMETERS
sim = {
    'imu': {
        'tactical_grade': {
            'sampling_period': 0.01, # [s], equiv to 100 Hz
            'vel_random_walk_m_s_sqrthr': 0.015, # [m/s/sqrt(hr)]
            'accelerometer_bias_stability_mg': 0.1, # [mg]
            'angle_random_walk_deg_sqrthr': 0.1, # [deg/sqrt(hr)]
            'gyro_bias_stability_deg_hr': 1.0, # [deg/hr]
            'time_constant_gyro_bias_s': 3600.0, # [s]
            'time_constant_accel_bias_s': 3600.0, # [s]
            'accelerometer_bias_stability_ms2': 0.1*9.80665/1000.0, # [m/s^2]
            'gyro_bias_stability_rads': np.deg2rad(1.0/3600.0), # [rad/s]
            'accelerometer_bias_cov': (np.eye(3)*(0.1*9.80665/1000.0)*np.sqrt((1-np.exp(-2*0.01/3600.0))))**2, # [m/s^2]
            'gyro_bias_cov': (np.eye(3)*(np.deg2rad(1.0/3600.0))*np.sqrt((1-np.exp(-2*0.01/3600.0))))**2, # [rad/s]
        },
        'adis16495': {
            # ADIS16495-1, 100 Hz output data rate
            'sampling_period': 0.01, # [s], equiv to 100 Hz
            'vel_random_walk_m_s_sqrthr': 0.03, # [m/s/sqrt(hr)], datasheet VRW
            'accelerometer_bias_stability_mg': 0.0032, # [mg] (3.2 ug, datasheet in-run bias stability)
            'angle_random_walk_deg_sqrthr': 0.3, # [deg/sqrt(hr)], datasheet ARW
            'gyro_bias_stability_deg_hr': 0.3, # [deg/hr], datasheet in-run bias stability
            'time_constant_gyro_bias_s': 25.0, # [s], MEMS bias correlation time
            'time_constant_accel_bias_s': 25.0, # [s], MEMS bias correlation time
            'accelerometer_bias_stability_ms2': 3.2e-6*9.80665, # [m/s^2]
            'gyro_bias_stability_rads': np.deg2rad(0.3/3600.0), # [rad/s]
            'accelerometer_bias_cov': (np.eye(3)*(3.2e-6*9.80665)*np.sqrt((1-np.exp(-2*0.01/25.0))))**2, # [m/s^2]
            'gyro_bias_cov': (np.eye(3)*(np.deg2rad(0.3/3600.0))*np.sqrt((1-np.exp(-2*0.01/25.0))))**2, # [rad/s]
        },
        'navigation_grade': {
            'sampling_period': 0.004, # [s], equiv to 250 Hz
            'vel_random_walk_m_s_sqrthr': 0.01, # [m/s/sqrt(hr)]
            'accelerometer_bias_stability_mg': 0.01, # [mg]
            'angle_random_walk_deg_sqrthr': 0.01, # [deg/sqrt(hr)]
            'gyro_bias_stability_deg_hr': 0.01, # [deg/hr]
            'time_constant_gyro_bias_s': 3600.0, # [s]
            'time_constant_accel_bias_s': 3600.0, # [s]
            'accelerometer_bias_cov': (np.eye(3)*(0.01*9.80665/1000.0)*np.sqrt((1-np.exp(-2*0.004/3600.0))))**2, # [m/s^2]
            'gyro_bias_cov': (np.eye(3)*(np.deg2rad(0.01/3600.0))*np.sqrt((1-np.exp(-2*0.004/3600.0))))**2, # [rad/s]
        },
        'noiseless': {
            'sampling_period': 0.004, # [s], equiv to 250 Hz
            'vel_random_walk_m_s_sqrthr': 0.0, 
            'accelerometer_bias_stability_mg': 0.0, 
            'angle_random_walk_deg_sqrthr': 0.0, 
            'gyro_bias_stability_deg_hr': 0.0, 
            'time_constant_gyro_bias_s': 3600.0, 
            'time_constant_accel_bias_s': 3600.0, 
            'accelerometer_bias_cov': np.zeros((3,3)), 
            'gyro_bias_cov': np.zeros((3,3)),
        },
    },
    'gps': {
        'clock': {
            'rubidium': {
                'phase_sigma': 1e-10, # [s] (0.030 m, PSD: 9.0e-4 m^2/s)
                'freq_sigma': 2.810693864511039e-14, # [s/s] (8.42e-6 m/s, PSD: 7.1e-11 m^2/s^3)
            },
            'crystal_ovenized': {
                'phase_sigma': 2e-10, # [s] (0.060 m, PSD: 3.6e-3 m^2/s)
                'freq_sigma': 2.8106938645110392e-11 # [s] (8.42e-3 m/s, PSD: 7.1e-5 m^2/s^3)
            },
            'crytal_temp_comp': {
                'phase_sigma': 3.1622776601683795e-10, # [s] (0.095 m, PSD: 9.0e-3 m^2/s)
                'freq_sigma': 6.244997998398398e-10 # [s] (0.187 m/s, PSD: 3.5e-2 m^2/s^3)
            },
        },
        'pseudorange': {
            'white_noise_std': 1.0, # [m]
            'radial_sigma': 0.8, # [m]
            'sat_clock_sigma_m': 1.0, # [m]
            'along_track_sigma': 1.5, # [m]
            'cross_track_sigma': 1.0, # [m]
            'total_sigma': 1.39, # [m]
        },
        'adr': {
            'white_noise_std': 0.01, # [m]     
        }
    },
    'prior_params': {
        'prior_pos_var': (1)**2, # 1 m sd [m^2]
        'prior_att_var': (1*np.pi/180)**2, # 1 degree sd [rad^2]
        'prior_vel_var': (1e-6)**2, # 1 micrometer/s sd [m^2/s^2]
        'prior_clock_bias_var': (1.0)**2, # 1 m sd [m^2]
        'prior_clock_drift_var': (100.0)**2, # 100 m/s sd [m^2/s^2]
    }
}

real_sensors = {
    'imu':{
        'adis16495': {
            # ADIS16495-1, 100 Hz output data rate
            'sampling_period': 0.01, # [s], equiv to 100 Hz
            'vel_random_walk_m_s_sqrthr': 0.03, # [m/s/sqrt(hr)], datasheet VRW
            'accelerometer_bias_stability_mg': 0.0032, # [mg] (3.2 ug, datasheet in-run bias stability)
            'angle_random_walk_deg_sqrthr': 0.3, # [deg/sqrt(hr)], datasheet ARW
            'gyro_bias_stability_deg_hr': 0.3, # [deg/hr], datasheet in-run bias stability
            'time_constant_gyro_bias_s': 100.0, # [s], MEMS bias correlation time
            'time_constant_accel_bias_s': 100.0, # [s], MEMS bias correlation time
            'accelerometer_bias_stability_ms2': 3.2e-6*9.80665, # [m/s^2]
            'gyro_bias_stability_rads': np.deg2rad(0.8/3600.0), # [rad/s]
            'accelerometer_bias_cov': (np.eye(3)*(3.2e-6*9.80665)*np.sqrt((1-np.exp(-2*0.01/25.0))))**2, # [m/s^2]
            'gyro_bias_cov': (np.eye(3)*(np.deg2rad(0.3/3600.0))*np.sqrt((1-np.exp(-2*0.01/25.0))))**2, # [rad/s]
        },
        'adis16495_golf_cart': {
            # ADIS16495-1, 100 Hz output data rate
            'sampling_period': 0.01, # [s], equiv to 100 Hz
            'vel_random_walk_m_s_sqrthr': 0.6,  # [m/s/sqrt(hr)]     
            'accelerometer_bias_stability_mg': 0.0032, # [mg]  
            'angle_random_walk_deg_sqrthr': 1.7, # [deg/sqrt(hr)], ARW
            'gyro_bias_stability_deg_hr': 0.3, # [deg/hr], bias stability
            'time_constant_gyro_bias_s': 3600.0, # [s], bias correlation time
            'time_constant_accel_bias_s': 3600.0, # [s], bias correlation time
            'accelerometer_bias_stability_ms2': 3.2e-6*9.80665, # [m/s^2]
            'gyro_bias_stability_rads': np.deg2rad(0.8/3600.0), # [rad/s]
            'accelerometer_bias_cov': (np.eye(3)*(3.2e-6*9.80665)*np.sqrt((1-np.exp(-2*0.01/10000.0))))**2, # [m/s^2]
            'gyro_bias_cov': (np.eye(3)*(np.deg2rad(0.3/3600.0))*np.sqrt((1-np.exp(-2*0.01/10000.0))))**2, # [rad/s]
        },
    },
    
    'gps': {
        'zed_f9t': {
            # ublox ZED-F9T-00B-01 GNSS timing receiver with geodetic antenna
            'clock': {
                'phase_sigma': 8.944271909999159e-10, # [s] (~0.27 m, PSD: 7.2e-2 m^2/s)
                'freq_sigma': 2.828427124746190e-09,   # [s/s] (~0.85 m/s, PSD: 7.2e-1 m^2/s^3)
                'isb_sigma': 1e-4, # [m/sqrt(s)] Inter-system bias standard deviation. Convert to sigma_isb = isb_sigma * sqrt(dt)
            },
            'pseudorange': {
                # 'white_noise_std': 1.0, # [m]
                # 'radial_sigma': 0.8, # [m]
                # 'sat_clock_sigma_m': 1.0, # [m]
                # 'along_track_sigma': 1.5, # [m]
                # 'cross_track_sigma': 1.0, # [m] 
                'total_sigma': 3.0, # [m]
            },
            'adr': {
                'white_noise_std': 0.01, # [m]  
                'white_noise_std_glonass': 0.02, # [m]
                'white_noise_std_galileo': 0.03, # [m]
            },
            'prior_params': {
                'prior_clock_bias_var': (1.0)**2, # 1 m sd [m^2]
                'prior_clock_drift_var': (100.0)**2, # 100 m/s sd [m^2/s^2]
            }
        },
    }
}
