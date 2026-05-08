import numpy as np
from numpy import sin, cos, pi
from util.constants import DEG_TO_RADIANS
import r3f

def _add_static_start(pos_flight, vel_flight, acc_flight, t_flight, duration, static_duration=60.0):
    """Helper to add start/end static periods."""
    dt = t_flight[1] - t_flight[0]
    t_static = np.arange(0, static_duration, dt)
    zeros_static = np.zeros((len(t_static), 3))
    
    pos_start = np.zeros_like(zeros_static) + pos_flight[0]
    
    positions = np.vstack([pos_start, pos_flight])
    velocities = np.vstack([zeros_static, vel_flight])
    accelerations = np.vstack([zeros_static, acc_flight])
    
    t_total = np.concatenate([
        t_static, 
        t_flight + static_duration, 
    ])

    return {
        'time': t_total,
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations
    }

def gen_medium_figure_eight_path(sample_period=0.01, num_cycles=5, stationary_time=0.0):
    
    one_cycle_duration = 6*60
    T = sample_period
    t = np.arange(0, one_cycle_duration*num_cycles, T)
    K = t.shape[0]

    # Define the path
    R = 1000 # meters
    delta_h = 150 # meters
    theta_t = np.linspace(0, 2*pi*num_cycles, K) # radians

    lat0_deg = 39.76 # degrees
    lon0_deg = -84.19 # degrees
    hae0 = 226 # meters
    lat0 = lat0_deg * DEG_TO_RADIANS # radians
    lon0 = lon0_deg * DEG_TO_RADIANS # radians

    # Tangent frame path
    d_theta_d_t = 2*pi/(one_cycle_duration)
    xt = (R/2)*sin(2*theta_t) # meters
    yt = R*(cos(theta_t)-1) # meters
    zt = (delta_h/2)*(cos(theta_t)-1) # meters
    pt = np.vstack([xt,yt,zt]).T

    vx = R*cos(2*theta_t)*d_theta_d_t
    vy = -R*sin(theta_t)*d_theta_d_t
    vz = -(delta_h/2)*sin(theta_t)*d_theta_d_t
    vt = np.vstack([vx,vy,vz]).T

    ax = -2*R*sin(2*theta_t)*d_theta_d_t**2
    ay = -R*cos(theta_t)*d_theta_d_t**2
    az = -(delta_h/2)*cos(theta_t)*d_theta_d_t**2
    at = np.vstack([ax,ay,az]).T

    ## Attitude
    roll_nav = (-1/27)*cos(theta_t)
    pitch_nav = ((-1/48)*sin(2*theta_t) + (1/143)*sin(4*theta_t))
    yaw_nav = np.arctan2(-2*sin(theta_t),cos(2*theta_t))
    rpy_nav = np.vstack([roll_nav,pitch_nav,yaw_nav]).T
    bRt = r3f.rpy_to_dcm(rpy = rpy_nav)

    seed = 416202311
    np.random.seed(seed)
    llh0 = np.random.rand(3) * np.array([np.pi, 2*np.pi, 1000]) - np.array([np.pi/2, np.pi, 0])
    pe0 = r3f.geodetic_to_ecef(llh = llh0)
    pe = r3f.tangent_to_ecef(pt = pt, pe0 = pe0)
    tRe = r3f.dcm_ecef_to_navigation(llh0[0],llh0[1])
    ve = np.array([tRe.T @ vt[i] for i in range(K)])
    ae = np.array([tRe.T @ at[i] for i in range(K)])

    eRb = np.array([tRe.T @ bRt[i].T for i in range(K)])

    trajectory_data_curvilinear = _add_static_start(pt, vt, at, t, duration=t[-1], static_duration=stationary_time)
    trajectory_data_curvilinear['initial_Rnb'] = np.eye(3)

    trajectory_data_ecef = _add_static_start(pe, ve, ae, t, duration=t[-1], static_duration=stationary_time)
    trajectory_data_ecef['initial_Reb'] = eRb[0]

    return trajectory_data_curvilinear, trajectory_data_ecef


def gen_scaled_figure_eight_path(sample_period: float = 0.01, total_duration: float = 1800.0, scale_factor: float = 1.0, stationary_time: float = 0.0):
    
    base_one_cycle_duration = 6 * 60
    base_R = 1000  # meters
    base_delta_h = 150  # meters
    
    # Scaled parameters
    d_theta_d_t = scale_factor * (2 * pi / base_one_cycle_duration)
    R = base_R / scale_factor
    delta_h = base_delta_h / scale_factor
    
    T = sample_period
    t = np.arange(0, total_duration, T)
    K = t.shape[0]

    # Generate theta based on time directly
    theta_t = d_theta_d_t * t
    
    lat0_deg = 39.76 # degrees
    lon0_deg = -84.19 # degrees
    lat0 = lat0_deg * DEG_TO_RADIANS # radians
    lon0 = lon0_deg * DEG_TO_RADIANS # radians

    # Tangent frame path
    xt = (R/2)*sin(2*theta_t) # meters
    yt = R*(cos(theta_t)-1) # meters
    zt = (delta_h/2)*(cos(theta_t)-1) # meters
    pt = np.vstack([xt,yt,zt]).T

    vx = R*cos(2*theta_t)*d_theta_d_t
    vy = -R*sin(theta_t)*d_theta_d_t
    vz = -(delta_h/2)*sin(theta_t)*d_theta_d_t
    vt = np.vstack([vx,vy,vz]).T

    ax = -2*R*sin(2*theta_t)*d_theta_d_t**2
    ay = -R*cos(theta_t)*d_theta_d_t**2
    az = -(delta_h/2)*cos(theta_t)*d_theta_d_t**2
    at = np.vstack([ax,ay,az]).T

    ## Attitude
    roll_nav = (-1/27)*cos(theta_t)
    pitch_nav = ((-1/48)*sin(2*theta_t) + (1/143)*sin(4*theta_t))
    yaw_nav = np.arctan2(-2*sin(theta_t),cos(2*theta_t))
    rpy_nav = np.vstack([roll_nav,pitch_nav,yaw_nav]).T
    bRt = r3f.rpy_to_dcm(rpy = rpy_nav)

    seed = 416202311
    np.random.seed(seed)
    llh0 = np.random.rand(3) * np.array([np.pi, 2*np.pi, 1000]) - np.array([np.pi/2, np.pi, 0])
    pe0 = r3f.geodetic_to_ecef(llh = llh0)
    pe = r3f.tangent_to_ecef(pt = pt, pe0 = pe0)
    tRe = r3f.dcm_ecef_to_navigation(llh0[0],llh0[1])
    ve = np.array([tRe.T @ vt[i] for i in range(K)])
    ae = np.array([tRe.T @ at[i] for i in range(K)])

    eRb = np.array([tRe.T @ bRt[i].T for i in range(K)])

    trajectory_data_curvilinear = _add_static_start(pt, vt, at, t, duration=t[-1], static_duration=stationary_time)
    trajectory_data_curvilinear['initial_Rnb'] = np.eye(3)

    trajectory_data_ecef = _add_static_start(pe, ve, ae, t, duration=t[-1], static_duration=stationary_time)
    trajectory_data_ecef['initial_Reb'] = eRb[0]

    return trajectory_data_curvilinear, trajectory_data_ecef


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    duration = 1800.0
    dt = 0.1
    scale_factors = [0.2, 0.4, 1.0, 2.5, 5.0]
    
    fig = plt.figure(figsize=(14, 8))
    
    for scale in scale_factors:
        _, traj_ecef = gen_scaled_figure_eight_path(sample_period=dt, total_duration=duration, scale_factor=scale)
        
        # We need to calculate realistic wbbi (rotation rate in body-frame).
        # We have trajectory_data_ecef, let's use the actual simulator's logic to see it:
        
        # Just use the simple RPY nav calculation from earlier inside the function
        # to plot rotation magnitude directly since the function outputs TPVA_FR-ready Rnb matrices
        # but to keep it self-contained without using inu or r3f again externally here,
        # we can just approximate rotation rate from the derivative of velocity direction 
        # or use a generic math representation. Actually gen_scaled_figure_eight_path 
        # doesn't export wbbi, it exports curvilinear and ecef.
        
        # Let's import the same machinery gen_sim_data uses to get wbbi
        from sim.gen_sim_data import rotation_matrices_from_ecef_trajectory_data
        import inu
        
        Reb_t = rotation_matrices_from_ecef_trajectory_data(traj_ecef)
        llh = r3f.ecef_to_geodetic(traj_ecef['positions'])
        Rne = r3f.dcm_ecef_to_navigation(llh[:,0],llh[:,1])
        Rbn_t = np.array([(Rne[i] @ Reb_t[i]).T for i in range(Reb_t.shape[0])])
        vne = np.array([Rne[i] @ traj_ecef['velocities'][i] for i in range(traj_ecef['velocities'].shape[0])])
        rpy = r3f.dcm_to_rpy(Rbn_t).T
        
        _, wbbi = inu.inv_mech(llh_t=llh, vne_t=vne, rpy_t=rpy, T=dt)
        wbbi = np.vstack((wbbi[0], wbbi[:-1]))  # Account for time shift
        
        # Magnitude in deg/s
        rotation_rate_mag = np.linalg.norm(wbbi, axis=1) * (180.0 / np.pi)
        
        plt.plot(traj_ecef['time'], rotation_rate_mag, linewidth=1.5, label=f'Scale {scale}')

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Rotation Rate Magnitude (deg/s)', fontsize=12)
    plt.title('Figure-8 Rotation Rates for Different Scale Factors', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figure_8_comparison.png')
    plt.show()
