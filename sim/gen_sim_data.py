from sim.rpy import simulate_attitude_from_path
from sim.lasso_trajectory import generate_lasso_trajectory
from sim.fig_8_trajectory import gen_scaled_figure_eight_path
from util.imu import grav_somigliana_pe

import numpy as np
import matplotlib.pyplot as plt
import r3f
import inu

def rotation_matrices_from_ecef_trajectory_data(trajectory_data):
    """Calculate roll, pitch, yaw angles from ECEF trajectory data."""
    positions = trajectory_data['positions']
    velocities = trajectory_data['velocities']
    accelerations = trajectory_data['accelerations']
    initial_Reb = trajectory_data['initial_Reb']
    initial_xb = initial_Reb[:, 0]
    initial_yb = initial_Reb[:, 1]
    

    gravities = np.array([grav_somigliana_pe(positions[i]) for i in range(positions.shape[0])])
    
    Reb_t = simulate_attitude_from_path(
        velocities_t = velocities,
        accelerations_t = accelerations,
        gravities_t = gravities,
        init_xb = initial_xb,
        init_yb= initial_yb,
    )


    
    return Reb_t


if __name__ == "__main__":
    ## Generate true and noiseless IMU data for the lasso trajectory.
    llh0 = np.zeros(3)
    initial_timestamp = 0.0 # Wait to add offset for TLEs in measurement sim engine

    imu_dt = 0.004 # 250 Hz
    _, ecef_trajectory_data = generate_lasso_trajectory(llh0=llh0,dt = imu_dt)
    Reb0 = ecef_trajectory_data['initial_Reb']
    Reb_t = rotation_matrices_from_ecef_trajectory_data(ecef_trajectory_data)
    times = np.arange(Reb_t.shape[0]) * imu_dt
    Reb_t[times<61.0] = Reb0.copy()  # Hold initial attitude during static start

    llh = r3f.ecef_to_geodetic(ecef_trajectory_data['positions'])

    Rne = r3f.dcm_ecef_to_navigation(llh[:,0],llh[:,1])
    Rbn_t = np.array([(Rne[i] @ Reb_t[i]).T for i in range(Reb_t.shape[0])])

    ve = ecef_trajectory_data['velocities']
    vne = np.array([Rne[i] @ ve[i] for i in range(ve.shape[0])])
    

    rpy = r3f.dcm_to_rpy(Rbn_t).T

    fbbi, wbbi = inu.inv_mech(llh_t = llh, vne_t = vne, rpy_t = rpy, T = imu_dt)
    fbbi = np.vstack((fbbi[0],fbbi[:-1])) # Account for the fact that INU has one sample shift in time paradigm
    wbbi = np.vstack((wbbi[0],wbbi[:-1])) # Account for the fact that INU has one sample shift in time paradigm

    times = initial_timestamp + np.arange(llh.shape[0]) * imu_dt

    tpva_fr = np.hstack((times[:, None], llh, vne, rpy, fbbi, wbbi))
    np.savez("sim/sim_trajectory_lasso_250Hz.npz",tpva_fr=tpva_fr)


    imu_dt = 0.01
    _, ecef_trajectory_data = generate_lasso_trajectory(llh0=llh0,dt = imu_dt)
    Reb0 = ecef_trajectory_data['initial_Reb']
    Reb_t = rotation_matrices_from_ecef_trajectory_data(ecef_trajectory_data)
    times = np.arange(Reb_t.shape[0]) * imu_dt
    Reb_t[times<61.0] = Reb0.copy()  # Hold initial attitude during static start

    llh = r3f.ecef_to_geodetic(ecef_trajectory_data['positions'])

    Rne = r3f.dcm_ecef_to_navigation(llh[:,0],llh[:,1])
    Rbn_t = np.array([(Rne[i] @ Reb_t[i]).T for i in range(Reb_t.shape[0])])

    ve = ecef_trajectory_data['velocities']
    vne = np.array([Rne[i] @ ve[i] for i in range(ve.shape[0])])
    

    rpy = r3f.dcm_to_rpy(Rbn_t).T

    fbbi, wbbi = inu.inv_mech(llh_t = llh, vne_t = vne, rpy_t = rpy, T = imu_dt)
    fbbi = np.vstack((fbbi[0],fbbi[:-1])) # Account for the fact that INU has one sample shift in time paradigm
    wbbi = np.vstack((wbbi[0],wbbi[:-1])) # Account for the fact that INU has one sample shift in time paradigm
    # fbbi - specific force body frame inertial (m/s^2)
    # wbbi - angular rate body frame inertial (rad/s)

    times = initial_timestamp + np.arange(llh.shape[0]) * imu_dt

    tpva_fr = np.hstack((times[:, None], llh, vne, rpy, fbbi, wbbi))
    np.savez("sim/sim_trajectory_lasso_100Hz.npz",tpva_fr=tpva_fr)

    gen_fig_8_traj = False # Toggle on to generate true, noiseless data for the scaled figure-8 trajectory scenarios.
    if gen_fig_8_traj:
        initial_timestamp = 0.0
        stationary_time = 0.0

        # Define scenarios
        # Base trajectory is a 6-minute cycle -> 10 cycles per hour (scale_factor=1.0)
        # Therefore, scale_factor = (cycles per hour) / 10
        scenarios = {
            '4cyc_per_hr': 0.4,
            '8cyc_per_hr': 0.8,
            '10cyc_per_hr': 1.0,
            '16cyc_per_hr': 1.6,
            '32cyc_per_hr': 3.2
        }

        for scenario_name, scale in scenarios.items():
            print(f"Generating Figure 8 scenario: {scenario_name} (scale: {scale})")
            
            # Nav Grade - 250 Hz
            imu_dt = 0.004
            duration = 1800.0  # 30 mins
            _, ecef_trajectory_data = gen_scaled_figure_eight_path(sample_period=imu_dt, total_duration=duration, scale_factor=scale, stationary_time=stationary_time)
            Reb0 = ecef_trajectory_data['initial_Reb']
            Reb_t = rotation_matrices_from_ecef_trajectory_data(ecef_trajectory_data)

            llh = r3f.ecef_to_geodetic(ecef_trajectory_data['positions'])

            Rne = r3f.dcm_ecef_to_navigation(llh[:,0],llh[:,1])
            Rbn_t = np.array([(Rne[i] @ Reb_t[i]).T for i in range(Reb_t.shape[0])])

            ve = ecef_trajectory_data['velocities']
            vne = np.array([Rne[i] @ ve[i] for i in range(ve.shape[0])])

            rpy = r3f.dcm_to_rpy(Rbn_t).T

            fbbi, wbbi = inu.inv_mech(llh_t = llh, vne_t = vne, rpy_t = rpy, T = imu_dt)
            fbbi = np.vstack((fbbi[0],fbbi[:-1])) # Account for the fact that INU has one sample shift in time paradigm
            wbbi = np.vstack((wbbi[0],wbbi[:-1])) # Account for the fact that INU has one sample shift in time paradigm

            times = initial_timestamp + np.arange(llh.shape[0]) * imu_dt
            tpva_fr = np.hstack((times[:, None], llh, vne, rpy, fbbi, wbbi))
            np.savez(f"sim/sim_trajectory_fig_8_{scenario_name}_250Hz.npz",tpva_fr=tpva_fr)

            # Tactical Grade - 100 Hz
            imu_dt = 0.01
            _, ecef_trajectory_data = gen_scaled_figure_eight_path(sample_period=imu_dt, total_duration=duration, scale_factor=scale, stationary_time=stationary_time)
            Reb0 = ecef_trajectory_data['initial_Reb']
            Reb_t = rotation_matrices_from_ecef_trajectory_data(ecef_trajectory_data)

            llh = r3f.ecef_to_geodetic(ecef_trajectory_data['positions'])

            Rne = r3f.dcm_ecef_to_navigation(llh[:,0],llh[:,1])
            Rbn_t = np.array([(Rne[i] @ Reb_t[i]).T for i in range(Reb_t.shape[0])])

            ve = ecef_trajectory_data['velocities']
            vne = np.array([Rne[i] @ ve[i] for i in range(ve.shape[0])])

            rpy = r3f.dcm_to_rpy(Rbn_t).T

            fbbi, wbbi = inu.inv_mech(llh_t = llh, vne_t = vne, rpy_t = rpy, T = imu_dt)
            fbbi = np.vstack((fbbi[0],fbbi[:-1])) # Account for the fact that INU has one sample shift in time paradigm
            wbbi = np.vstack((wbbi[0],wbbi[:-1])) # Account for the fact that INU has one sample shift in time paradigm

            times = initial_timestamp + np.arange(llh.shape[0]) * imu_dt
            tpva_fr = np.hstack((times[:, None], llh, vne, rpy, fbbi, wbbi))
            np.savez(f"sim/sim_trajectory_fig_8_{scenario_name}_100Hz.npz",tpva_fr=tpva_fr)

        # Plot 100Hz tpva_fr data
        # tpva_fr columns: [time, lat, lon, height, vn, ve, vd, roll, pitch, yaw, fx, fy, fz, wx, wy, wz]
        time_rel = tpva_fr[:, 0] - tpva_fr[0, 0]  # Relative time in seconds
        # Convert rpy and wbbi from radians to degrees
        rpy_deg = np.rad2deg(tpva_fr[:, 7:10])
        wbbi_deg = np.rad2deg(tpva_fr[:, 13:16])
        # Create figure with subplots for all components
        fig, axes = plt.subplots(5, 3, figsize=(18, 14))
        fig.suptitle('100Hz TPVA_FR Data Over Time - Figure-8 Trajectory', fontsize=16, fontweight='bold')
        # LLH (geodetic coordinates)
        axes[0, 0].plot(time_rel, tpva_fr[:, 1], 'b-', linewidth=1)
        axes[0, 0].set_ylabel('Latitude (rad)', fontsize=10)    
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('LLH - Latitude')
        axes[0, 1].plot(time_rel, tpva_fr[:, 2], 'b-', linewidth=1)
        axes[0, 1].set_ylabel('Longitude (rad)', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('LLH - Longitude')
        axes[0, 2].plot(time_rel, tpva_fr[:, 3], 'b-', linewidth=1)
        axes[0, 2].set_ylabel('Height (m)', fontsize=10)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_title('LLH - Height')
        # VNE (velocity in NED frame)
        axes[1, 0].plot(time_rel, tpva_fr[:, 4], 'g-', linewidth=1)
        axes[1, 0].set_ylabel('V_North (m/s)', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('VNE - North')
        axes[1, 1].plot(time_rel, tpva_fr[:, 5], 'g-', linewidth=1)
        axes[1, 1].set_ylabel('V_East (m/s)', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('VNE - East')
        axes[1, 2].plot(time_rel, tpva_fr[:, 6], 'g-', linewidth=1)
        axes[1, 2].set_ylabel('V_Down (m/s)', fontsize=10)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_title('VNE - Down')
        # RPY (attitude in degrees)
        axes[2, 0].plot(time_rel, rpy_deg[:, 0], 'r-', linewidth=1)
        axes[2, 0].set_ylabel('Roll (deg)', fontsize=10)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_title('RPY - Roll')
        axes[2, 1].plot(time_rel, rpy_deg[:, 1], 'r-', linewidth=1)
        axes[2, 1].set_ylabel('Pitch (deg)', fontsize=10)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_title('RPY - Pitch')
        axes[2, 2].plot(time_rel, rpy_deg[:, 2], 'r-', linewidth=1)
        axes[2, 2].set_ylabel('Yaw (deg)', fontsize=10)
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_title('RPY - Yaw')
        # FBBI (specific force in body frame)
        axes[3, 0].plot(time_rel, tpva_fr[:, 10], 'purple', linewidth=1)
        axes[3, 0].set_ylabel('f_x (m/s²)', fontsize=10)
        axes[3, 0].grid(True, alpha=0.3)
        axes[3, 0].set_title('FBBI - X')
        axes[3, 1].plot(time_rel, tpva_fr[:, 11], 'purple', linewidth=1)
        axes[3, 1].set_ylabel('f_y (m/s²)', fontsize=10)
        axes[3, 1].grid(True, alpha=0.3)    
        axes[3, 1].set_title('FBBI - Y')
        axes[3, 2].plot(time_rel, tpva_fr[:, 12], 'purple', linewidth=1)
        axes[3, 2].set_ylabel('f_z (m/s²)', fontsize=10)
        axes[3, 2].grid(True, alpha=0.3)    
        axes[3, 2].set_title('FBBI - Z')
        # WBBI (angular velocity in body frame, degrees)
        axes[4, 0].plot(time_rel, wbbi_deg[:, 0], 'orange', linewidth=1)
        axes[4, 0].set_ylabel('ω_x (deg/s)', fontsize=10)
        axes[4, 0].set_xlabel('Time (s)', fontsize=10)
        axes[4, 0].grid(True, alpha=0.3)
        axes[4, 0].set_title('WBBI - X')
        axes[4, 1].plot(time_rel, wbbi_deg[:, 1], 'orange', linewidth=1)
        axes[4, 1].set_ylabel('ω_y (deg/s)', fontsize=10)
        axes[4, 1].set_xlabel('Time (s)', fontsize=10)
        axes[4, 1].grid(True, alpha=0.3)
        axes[4, 1].set_title('WBBI - Y')
        axes[4, 2].plot(time_rel, wbbi_deg[:, 2], 'orange', linewidth=1)
        axes[4, 2].set_ylabel('ω_z (deg/s)', fontsize=10)
        axes[4, 2].set_xlabel('Time (s)', fontsize=10)
        axes[4, 2].grid(True, alpha=0.3)
        axes[4, 2].set_title('WBBI - Z')
        plt.tight_layout()

        # Plot lat-lon path
        plt.figure(figsize=(8, 6))
        plt.plot(tpva_fr[:, 2], tpva_fr[:, 1], 'b-', linewidth=1)
        plt.title('Figure-8 Trajectory Path (Lat vs Lon)', fontsize=16, fontweight='bold')
        plt.xlabel('Longitude (rad)', fontsize=12)
        plt.ylabel('Latitude (rad)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
