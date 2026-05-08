
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import r3f
import folium
from scipy.interpolate import interp1d

# Folium Basemaps
basemaps = {
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    )
}

def ned_to_geodetic(pos_ned, eTw):
    # pos_ned is (N, 3)
    pos_ecef = np.zeros_like(pos_ned)
    for i in range(len(pos_ned)):
        p_ned_h = np.append(pos_ned[i], 1.0)
        p_ecef_h = eTw @ p_ned_h
        pos_ecef[i] = p_ecef_h[:3]
    
    # ECEF to LLH
    llh = r3f.ecef_to_geodetic(pos_ecef)
    lat = np.degrees(llh[:, 0])
    lon = np.degrees(llh[:, 1])
    return lat, lon

def plot_estimated_path_only(lat, lon, output_file_name):
    # lat, lon are (N,)
    map_center = [np.mean(lat), np.mean(lon)]
    map = folium.Map(location=map_center, zoom_start=18, control_scale=True)

    # Estimated Path (Yellow)
    path_coords = np.hstack((lat[:,np.newaxis], lon[:,np.newaxis]))
    folium.PolyLine(path_coords, color="yellow", weight=4.0, opacity=0.7, tooltip='Estimated Trajectory').add_to(map)

    basemaps['Google Satellite'].add_to(map)
    map.save(output_file_name + ".html")

def plot_golf_cart_results(result_path, max_time_m=None):
    """Generate plots for golf cart experiment results similar to simulation plots."""
    print(f"Loading results from {result_path}...")
    if not os.path.exists(result_path):
        print(f"Error: File {result_path} does not exist.")
        return

    data = np.load(result_path, allow_pickle=True)
    
    times = data['times'] if 'times' in data else data.get('t_state', [])
    pos_est = data['position_estimates']
    pos_true = data['true_positions']
    vel_est = data.get('velocity_estimates')
    vel_true = data.get('true_velocities')
    rpy_est = data.get('rpy_estimates')
    
    # Calculate RPY errors if available
    pw_errors = data.get('pw_errors')
    vw_errors = data.get('vw_errors')
    
    # Biases
    accel_bias = data.get('accel_bias_estimates')
    gyro_bias = data.get('gyro_bias_estimates')
    clock_bias = data.get('clock_bias_estimates')
    clock_drift = data.get('clock_drift_estimates')
    
    # Lever arm
    lever_arm_est = data.get('lever_arm_estimates')
    lever_arm_err = data.get('lever_arm_errors')
    true_lever_arm = data.get('true_lever_arm')
    lever_arm_cov = data.get('lever_arm_covariances')

    # PIM errors
    final_pim_errors = data.get('final_pim_errors')
    final_pim_error_times = data.get('final_pim_error_times')

    # eTw for maps
    eTw = data.get('eTw')

    # Time relative to start
    t_rel = times - times[0]

    if max_time_m is not None and len(times) > 0:
        valid_time_idx = t_rel <= (max_time_m * 60.0)
        times = times[valid_time_idx]
        pos_est = pos_est[valid_time_idx]
        pos_true = pos_true[valid_time_idx]
        if vel_est is not None: vel_est = vel_est[valid_time_idx]
        if vel_true is not None: vel_true = vel_true[valid_time_idx]
        if rpy_est is not None: rpy_est = rpy_est[valid_time_idx]
        if pw_errors is not None: pw_errors = pw_errors[valid_time_idx]
        if vw_errors is not None: vw_errors = vw_errors[valid_time_idx]
        if accel_bias is not None: accel_bias = accel_bias[valid_time_idx]
        if gyro_bias is not None: gyro_bias = gyro_bias[valid_time_idx]
        
        if clock_bias is not None:
            min_len = min(len(clock_bias), len(valid_time_idx))
            clock_bias = clock_bias[:min_len][valid_time_idx[:min_len]]
        if clock_drift is not None:
            min_len = min(len(clock_drift), len(valid_time_idx))
            clock_drift = clock_drift[:min_len][valid_time_idx[:min_len]]
        t_rel = t_rel[valid_time_idx]
        
        if lever_arm_est is not None:
            la_t = lever_arm_est[:, 0] - lever_arm_est[0, 0]
            la_idx = la_t <= (max_time_m * 60.0)
            lever_arm_est = lever_arm_est[la_idx]
            if lever_arm_err is not None:
                lever_arm_err = lever_arm_err[la_idx]
            if lever_arm_cov is not None and hasattr(lever_arm_cov, 'ndim') and lever_arm_cov.ndim == 3:
                lever_arm_cov = lever_arm_cov[la_idx]

    # Output directory
    res_dir = os.path.dirname(result_path)

    # ---------------------------------------------------------
    # 1. Trajectory Summary (Similar to sim plots)
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle(f"Trajectory Summary: {os.path.basename(res_dir)}", fontsize=16)

    # N-E Position (Assuming World frame is local-level, likely NED or similar)
    # Based on sliding_optimization_gps_imu_golf_cart.py, the order is likely [X, Y, Z]
    # We'll plot Y vs X or X vs Y depending on the convention. 
    # Usually X is North, Y is East.
    ax1 = axes1[0, 0]
    ax1.plot(pos_true[:, 1], pos_true[:, 0], 'k--', label='Truth', alpha=0.5)
    ax1.plot(pos_est[:, 1], pos_est[:, 0], 'b-', label='Estimate')
    ax1.set_xlabel('East (m)')
    ax1.set_ylabel('North (m)')
    ax1.set_title('N-E Position')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend()

    # Altitude vs Time
    ax2 = axes1[0, 1]
    # Altitude is -Z in NED
    ax2.plot(t_rel, -pos_true[:, 2], 'k--', label='Truth', alpha=0.5)
    ax2.plot(t_rel, -pos_est[:, 2], 'tab:orange', label='Estimate')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude vs Time')
    ax2.grid(True)
    ax2.legend()

    # Velocity (Local Frame)
    if vel_est is not None:
        ax3 = axes1[1, 0]
        ax3.plot(t_rel, vel_est[:, 0], label='Vx (North)')
        ax3.plot(t_rel, vel_est[:, 1], label='Vy (East)')
        ax3.plot(t_rel, vel_est[:, 2], label='Vz (Down)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Velocity (Local Frame)')
        ax3.grid(True)
        ax3.legend()

    # Attitude (RPY)
    if rpy_est is not None:
        ax4 = axes1[1, 1]
        rpy_deg = np.rad2deg(rpy_est)
        ax4.plot(t_rel, rpy_deg[:, 0], label='Roll')
        ax4.plot(t_rel, rpy_deg[:, 1], label='Pitch')
        ax4.plot(t_rel, rpy_deg[:, 2], label='Yaw')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angle (deg)')
        ax4.set_title('Attitude (RPY)')
        ax4.grid(True)
        ax4.legend()

    plt.tight_layout()
    summary_path = os.path.join(res_dir, "trajectory_summary.png")
    plt.savefig(summary_path)
    print(f"Saved trajectory summary to {summary_path}")

    # ---------------------------------------------------------
    # 2. Diagnostic Summary
    # ---------------------------------------------------------
    fig2, axes2 = plt.subplots(3, 2, figsize=(18, 15))
    fig2.suptitle(f"Diagnostic Summary: {os.path.basename(res_dir)}", fontsize=16)

    # Position Errors
    if pw_errors is not None:
        ax_err_p = axes2[0, 0]
        ax_err_p.plot(t_rel, pw_errors[:, 0], label='North Error')
        ax_err_p.plot(t_rel, pw_errors[:, 1], label='East Error')
        ax_err_p.plot(t_rel, pw_errors[:, 2], label='Down Error')
        ax_err_p.set_xlabel('Time (s)')
        ax_err_p.set_ylabel('Error (m)')
        ax_err_p.set_title('Position Errors')
        ax_err_p.grid(True)
        ax_err_p.legend()

    # Velocity Errors
    if vw_errors is not None:
        ax_err_v = axes2[0, 1]
        ax_err_v.plot(t_rel, vw_errors[:, 0], label='Vx Error')
        ax_err_v.plot(t_rel, vw_errors[:, 1], label='Vy Error')
        ax_err_v.plot(t_rel, vw_errors[:, 2], label='Vz Error')
        ax_err_v.set_xlabel('Time (s)')
        ax_err_v.set_ylabel('Error (m/s)')
        ax_err_v.set_title('Velocity Errors')
        ax_err_v.grid(True)
        ax_err_v.legend()

    # IMU Biases (Accel)
    if accel_bias is not None:
        ax_bias_a = axes2[1, 0]
        ax_bias_a.plot(t_rel, accel_bias[:, 0], label='Accel X')
        ax_bias_a.plot(t_rel, accel_bias[:, 1], label='Accel Y')
        ax_bias_a.plot(t_rel, accel_bias[:, 2], label='Accel Z')
        ax_bias_a.set_xlabel('Time (s)')
        ax_bias_a.set_ylabel('Bias (m/s^2)')
        ax_bias_a.set_title('Accelerometer Biases')
        ax_bias_a.grid(True)
        ax_bias_a.legend()

    # IMU Biases (Gyro)
    if gyro_bias is not None:
        ax_bias_g = axes2[1, 1]
        ax_bias_g.plot(t_rel, np.rad2deg(gyro_bias[:, 0]), label='Gyro X')
        ax_bias_g.plot(t_rel, np.rad2deg(gyro_bias[:, 1]), label='Gyro Y')
        ax_bias_g.plot(t_rel, np.rad2deg(gyro_bias[:, 2]), label='Gyro Z')
        ax_bias_g.set_xlabel('Time (s)')
        ax_bias_g.set_ylabel('Bias (deg/s)')
        ax_bias_g.set_title('Gyroscope Biases')
        ax_bias_g.grid(True)
        ax_bias_g.legend()

    # Clock State
    if clock_bias is not None and clock_drift is not None:
        ax_clock = axes2[2, 0]
        ax_clock_drift = ax_clock.twinx()
        p1, = ax_clock.plot(t_rel, clock_bias, 'b-', label='Bias')
        p2, = ax_clock_drift.plot(t_rel, clock_drift, 'r-', label='Drift')
        ax_clock.set_xlabel('Time (s)')
        ax_clock.set_ylabel('Bias (m)', color='b')
        ax_clock_drift.set_ylabel('Drift (m/s)', color='r')
        ax_clock.set_title('Receiver Clock State')
        ax_clock.grid(True)
        ax_clock.legend(handles=[p1, p2])

    # Lever Arm
    ax_lb = axes2[2, 1]
    if lever_arm_est is not None:
        # lever_arm_est is often saved with [time, x, y, z] or just [x, y, z] per step
        # In sliding_optimization_gps_imu_golf_cart.py line 982: 'lever_arm_estimates': lever_arm_estimates
        # where lever_arm_estimates is position_estimates.pop_all() style: [t, x, y, z]
        lb_t = lever_arm_est[:, 0] - times[0]
        lb_vals = lever_arm_est[:, 1:]
        
        ax_lb.plot(lb_t, lb_vals[:, 0], 'b-', label='X')
        ax_lb.plot(lb_t, lb_vals[:, 1], 'g-', label='Y')
        ax_lb.plot(lb_t, lb_vals[:, 2], 'r-', label='Z')
        
        if true_lever_arm is not None:
            ax_lb.axhline(y=true_lever_arm[0], color='b', linestyle='--', alpha=0.5)
            ax_lb.axhline(y=true_lever_arm[1], color='g', linestyle='--', alpha=0.5)
            ax_lb.axhline(y=true_lever_arm[2], color='r', linestyle='--', alpha=0.5)
            
        if lever_arm_cov is not None and hasattr(lever_arm_cov, 'ndim') and lever_arm_cov.ndim == 3:
            # lever_arm_cov shape (N, 3, 3)
            sigmas = np.sqrt(np.diagonal(lever_arm_cov, axis1=1, axis2=2))
            ax_lb.fill_between(lb_t, lb_vals[:, 0] - sigmas[:, 0], lb_vals[:, 0] + sigmas[:, 0], color='b', alpha=0.1)
            ax_lb.fill_between(lb_t, lb_vals[:, 1] - sigmas[:, 1], lb_vals[:, 1] + sigmas[:, 1], color='g', alpha=0.1)
            ax_lb.fill_between(lb_t, lb_vals[:, 2] - sigmas[:, 2], lb_vals[:, 2] + sigmas[:, 2], color='r', alpha=0.1)

        ax_lb.set_xlabel('Time (s)')
        ax_lb.set_ylabel('Lever Arm (m)')
        ax_lb.set_title('Lever Arm Estimate')
        ax_lb.grid(True)
        ax_lb.legend()
    else:
        ax_lb.text(0.5, 0.5, 'No Lever Arm Estimates', ha='center', va='center')
        ax_lb.set_title('Lever Arm Estimate')

    plt.tight_layout()
    diag_path = os.path.join(res_dir, "diagnostic_summary.png")
    plt.savefig(diag_path)
    print(f"Saved diagnostic summary to {diag_path}")

    # ---------------------------------------------------------
    # 3. Velocity Comparison
    # ---------------------------------------------------------
    if vel_est is not None and vel_true is not None:
        fig3, axes3 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig3.suptitle(f"Velocity Comparison: {os.path.basename(res_dir)}", fontsize=16)

        labels = ['North', 'East', 'Down']
        for i in range(3):
            ax = axes3[i]
            ax.plot(t_rel, vel_true[:, i], 'k--', label='Truth', alpha=0.7)
            ax.plot(t_rel, vel_est[:, i], 'b-', label='Estimate', alpha=0.7)
            ax.set_ylabel(f'Velocity {labels[i]} (m/s)')
            ax.grid(True)
            ax.legend()
        axes3[-1].set_xlabel('Time (s)')

        plt.tight_layout()
        vel_comp_path = os.path.join(res_dir, "velocity_comparison.png")
        plt.savefig(vel_comp_path)
        print(f"Saved velocity comparison to {vel_comp_path}")


    # ---------------------------------------------------------
    # 3. Folium Map
    # ---------------------------------------------------------
    if eTw is not None:
        try:
            # lat_true, lon_true = ned_to_geodetic(pos_true, eTw) # Not using truth for map
            lat_est, lon_est = ned_to_geodetic(pos_est, eTw)
            map_path = os.path.join(res_dir, "trajectory_map")
            plot_estimated_path_only(lat_est, lon_est, map_path)
            print(f"Saved folium map to {map_path}.html")
        except Exception as e:
            print(f"Failed to generate Folium map: {e}")

def load_data(path, max_time=1000000):
    print(f"Loading {path}...")
    data = np.load(path, allow_pickle=True)
    # Some results might use different keys, but our data processing standardizes them
    times = data['times'] if 'times' in data else data.get('t_state', [])
    pos = data['position_estimates']
    vel = data['velocity_estimates']
    
    # Relative time from start
    t0 = times[0]
    t_rel = times - t0
    
    # Truncate
    mask = t_rel <= max_time
    return t_rel[mask], pos[mask], vel[mask]

def plot_comparison(results_dir):
    # Use the new "no_prior" results provided by the user
    results_path = os.path.join(results_dir, 'results.npz')
        # ---------------------------------------------------------
    # Lever Arm Estimation Errors (Absolute Error)
    # ---------------------------------------------------------
    res_est = np.load(results_path, allow_pickle=True)
    if 'lever_arm_estimates' in res_est:
        la_est = res_est['lever_arm_estimates']
        true_la = res_est.get('true_lever_arm', np.array([0.0508, -0.854075, -0.0384625]))
        
        la_t = la_est[:, 0] - la_est[0, 0]
        la_vals = la_est[:, 1:]
        
        # Calculate Absolute Errors
        abs_la_err = np.abs(la_vals - true_la)
        
        fig_la, ax_la = plt.subplots(1, 1, figsize=(6, 3.5))
        
        # Absolute Errors (X, Y, Z)
        colors = ['red', 'green', 'blue']
        axis_subscripts = ['x', 'y', 'z']
        axis_styles = ['-', '--', ':']
        for i, (color, sub, style) in enumerate(zip(colors, axis_subscripts, axis_styles)):
            label = rf'$\hat{{l}}_{{b,{sub}}}$ Error'
            ax_la.plot(la_t, abs_la_err[:, i], color=color, linestyle=style, label=label)
        
        ax_la.set_ylabel('Absolute Error [m]')
        ax_la.set_xlabel('Time [s]')
        ax_la.grid(True)
        # Use a tight layout for better appearance with potentially large axis labels
        ax_la.legend(loc='upper right')
            
        plt.tight_layout()
        plt.savefig('lever_arm_estimation_errors.png')
        print("Saved absolute lever arm error plot to lever_arm_estimation_errors.png")
        
        # Save a copy in the estimated run folder
        plt.savefig(os.path.join(results_dir, 'lever_arm_estimation_errors.png'))
        print(f"Saved copy to {os.path.join(results_dir, 'lever_arm_estimation_errors.png')}")
    else:
        print("No lever arm estimates found in the estimated run file.")

def main(args: argparse.Namespace = None):
    if args is not None and args.result_path:
        if os.path.isfile(args.result_path) and args.result_path.endswith('.npz'):
            plot_golf_cart_results(args.result_path, max_time_m=args.max_time_m)
        else:
            print(f"Error: {args.result_path} is not a valid .npz file")
    else:
        # Search for results to plot automatically
        for path in [
            'golf_cart', 
            ]:
            base_dir = os.path.join('results', path)
            found = False
            for root, dirs, files in os.walk(base_dir):
                if 'results.npz' in files:
                    plot_golf_cart_results(os.path.join(root, 'results.npz'), max_time_m=args.max_time_m)
                    plot_comparison(root)
                    found = True    
                    
            if not found:
                print(f"No results.npz found in {base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot golf cart results.")
    parser.add_argument('result_path', type=str, nargs='?', default=None, help='Path to results.npz.')
    parser.add_argument('--max_time_m', type=float, default=None, help='Max time in minutes to plot.')
    args: argparse.Namespace = parser.parse_args()
    main(args)
    

