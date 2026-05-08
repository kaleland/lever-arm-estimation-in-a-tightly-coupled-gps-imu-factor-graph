import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any

import scipy
from experiment.sliding_optimization_gps_imu import sliding_optimization_gps_imu
from util import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams['mathtext.fontset'] = 'stix'

def plot_comparison(result_paths, labels, output_dir, title_prefix="", plot_clock=False, plot_lever_arm_norm=False):
    """Generates comparison plots for the given results."""
    print(f"Generating comparison plots in {output_dir}...")
    
    results = []
    for path in result_paths:
        results.append(np.load(path, allow_pickle=True))
        
    # Assume all align roughly in time, but we plot against their own times to be safe.
    
    # 1. 2D Position Error Comparison
    plt.figure(figsize=(10, 6))
    for res, label in zip(results, labels):
        times = res['times']
        if len(times) > 0: times = times - times[0]
        pos_err = res['pw_errors']
        horiz_err = np.sqrt(pos_err[:,0]**2 + pos_err[:,1]**2)
        
        min_len = min(len(times), len(horiz_err))
        plt.plot(times[:min_len], horiz_err[:min_len], label=label, linewidth=2)
        
    plt.xlabel('Time [s]')
    plt.ylabel('Horizontal Position Error [m]')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'comparison_pos_error_2d.png'))
    plt.close()

    # 2. 2D Velocity Error Comparison
    plt.figure(figsize=(10, 6))
    for res, label in zip(results, labels):
        times = res['times']
        if len(times) > 0: times = times - times[0]
        vel_err = res['vw_errors']
        horiz_vel_err = np.sqrt(vel_err[:,0]**2 + vel_err[:,1]**2)
        
        min_len = min(len(times), len(horiz_vel_err))
        plt.plot(times[:min_len], horiz_vel_err[:min_len], label=label, linewidth=2)
        
    plt.xlabel('Time [s]')
    plt.ylabel('Horizontal Velocity Error [m/s]')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'comparison_vel_error_2d.png'))
    plt.close()

def run_ignored_vs_estimated_study():
    print("\n" + "="*50)
    print("Running Ignored vs. Estimated Lever Arm Comparison")
    print("="*50)

    sim_base_dir = os.path.join('sim', 'lasso')
    grade = "tactical_grade"
    true_lever_arm = np.array([1.0, 0.0, 0.0])
    
    sim_subdir = f"lasso/{grade}"
    sim_dir = os.path.join(sim_base_dir, grade, 'lever_arm_simulations')
    gps_file = os.path.join(sim_dir, "gps_lever_arm_1.00_0.00_0.00.npz")
    
    if not os.path.exists(gps_file):
        print(f"Error: GPS file not found: {gps_file}")
        return

    output_base_dir = os.path.join("results", "ignored_vs_estimated_lever_arm")
    os.makedirs(output_base_dir, exist_ok=True)

    # Sample initial position error
    np.random.seed(42)
    initial_position_error = np.random.multivariate_normal(np.zeros(3), 4.0*np.eye(3))
    print(f"  Initial Position Error: {initial_position_error}")

    method_init = np.zeros(3)

    # --- Method 1: Ignored Lever Arm (Fixed at 0,0,0) ---
    print("  Running Method 1: Ignored (Fixed at 0,0,0)")
    try:
        res_m1 = sliding_optimization_gps_imu(
            sim_subdir=sim_subdir,
            window_dur=30.0,
            save=True,
            save_covariances=True,
            true_lever_arm=true_lever_arm,
            lever_arm_init=method_init,
            unknown_lever_arm=False,
            lever_arm_prior_var=None,
            output_dir=os.path.join(output_base_dir, "method_1_ignored"),
            gps_data_file=gps_file,
            init_rotation_error=np.eye(3),
            initial_position_error=initial_position_error
        )
    except Exception as e:
        print(f"FAILED Method 1: {e}")
        import traceback
        traceback.print_exc()
        res_m1 = {}

    # --- Method 2: Estimated Lever Arm ---
    print("  Running Method 2: Estimated")
    try:
        res_m2 = sliding_optimization_gps_imu(
            sim_subdir=sim_subdir,
            window_dur=30.0,
            save=True,
            save_covariances=True,
            true_lever_arm=true_lever_arm,
            lever_arm_init=method_init,
            unknown_lever_arm=True,
            lever_arm_prior_var=None,
            output_dir=os.path.join(output_base_dir, "method_2_estimated"),
            gps_data_file=gps_file,
            init_rotation_error=np.eye(3),
            initial_position_error=initial_position_error
        )
    except Exception as e:
        print(f"FAILED Method 2: {e}")
        import traceback
        traceback.print_exc()
        res_m2 = {}

    # --- Plotting & Velocity Table ---
    if res_m1.get('save_path') and res_m2.get('save_path'):
        data_m1 = np.load(res_m1['save_path'], allow_pickle=True)
        data_m2 = np.load(res_m2['save_path'], allow_pickle=True)

        # 1. Custom Position Error Plot
        plt.figure(figsize=(8, 6))
        
        # Ignored (Method 1)
        times_1 = data_m1['times']
        if len(times_1) > 0: times_1 = times_1 - times_1[0]
        pos_err_1 = data_m1['pw_errors']
        horiz_err_1 = np.sqrt(pos_err_1[:,0]**2 + pos_err_1[:,1]**2)
        min_len_1 = min(len(times_1), len(horiz_err_1))
        plt.plot(times_1[:min_len_1], horiz_err_1[:min_len_1], label='Ignored Lever Arm', linestyle='--', linewidth=2)
        
        # Estimated (Method 2)
        times_2 = data_m2['times']
        if len(times_2) > 0: times_2 = times_2 - times_2[0]
        pos_err_2 = data_m2['pw_errors']
        horiz_err_2 = np.sqrt(pos_err_2[:,0]**2 + pos_err_2[:,1]**2)
        min_len_2 = min(len(times_2), len(horiz_err_2))
        plt.plot(times_2[:min_len_2], horiz_err_2[:min_len_2], label='Estimated Lever Arm', linewidth=2)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Horizontal IMU Position Error [m]')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_base_dir, 'comparison_imu_pos_error_2d.png'))
        plt.close()

        # 1.b. Custom Position Error Plot of Estimated GPS Receiver Position
        plt.figure(figsize=(8, 6))

        # Ignored (Method 1)
        imu_position_estimates = data_m1['position_estimates'] # shape (N, 3)
        gps_position_estimates = imu_position_estimates.copy() # Ignored lever arm for Method 1 means GPS position is same as IMU position estimate
        imu_position_true = data_m1['true_positions'] # shape (N, 3)
        imu_attitude_true = data_m1['true_attitudes'] # shape (N, 3, 3)
        true_lever_arm = data_m1['true_lever_arm'] # shape (3,)
        gps_positions_true = np.array([imu_attitude_true[i] @ true_lever_arm + imu_position_true[i] for i in range(len(imu_position_true))]) # shape (N, 3)
        gps_position_errors_2D = np.linalg.norm(gps_position_estimates[:min_len_1, :2] - gps_positions_true[:min_len_1, :2], axis=1)
        plt.plot(times_1[:min_len_1], gps_position_errors_2D, label='Ignored Lever Arm', linestyle='--', linewidth=2)
        
        # Estimated (Method 2)
        imu_position_estimates = data_m2['position_estimates']
        imu_attitude_estimates = data_m2['attitude_estimates']
        lever_arm_estimates = data_m2['lever_arm_estimates'][:, 1:]
        # Lever arm estimates is shorter than attitude estimates. Extend the array by repeating the last estimate for the remaining time steps.
        if len(lever_arm_estimates) < len(imu_attitude_estimates):
            last_estimate = lever_arm_estimates[-1]
            extended_lever_arm_estimates = np.vstack([lever_arm_estimates, np.tile(last_estimate, (len(imu_attitude_estimates) - len(lever_arm_estimates), 1))])
        else:
            extended_lever_arm_estimates = lever_arm_estimates
        gps_position_estimates = np.array([imu_attitude_estimates[i] @ extended_lever_arm_estimates[i] + imu_position_estimates[i] for i in range(len(imu_position_estimates))])
        gps_position_errors_2D = np.linalg.norm(gps_position_estimates[:min_len_2, :2] - gps_positions_true[:min_len_2, :2], axis=1)
        plt.plot(times_2[:min_len_2], gps_position_errors_2D, label='Estimated Lever Arm', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Horizontal GPS Position Error [m]')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_base_dir, 'comparison_gps_pos_error_2d.png'))
        plt.close()

        # 2. LaTeX Velocity Table
        def get_imu_vel_stats(data):
            vel_err = data['vw_errors']
            horiz_vel_err = np.sqrt(vel_err[:,0]**2 + vel_err[:,1]**2)
            mean_err = np.mean(horiz_vel_err)
            median_err = np.median(horiz_vel_err)
            std_err = np.std(horiz_vel_err)
            max_err = np.max(horiz_vel_err)
            return mean_err, median_err, std_err, max_err

        def get_gps_pos_stats(gps_pos_errs_2d):
            mean_err = np.mean(gps_pos_errs_2d)
            median_err = np.median(gps_pos_errs_2d)
            std_err = np.std(gps_pos_errs_2d)
            max_err = np.max(gps_pos_errs_2d)
            rmse_err = np.sqrt(np.mean(gps_pos_errs_2d**2))
            return mean_err, median_err, std_err, max_err, rmse_err
        
        mean_1_imu, med_1_imu, std_1_imu, max_1_imu = get_imu_vel_stats(data_m1)
        mean_2_imu, med_2_imu, std_2_imu, max_2_imu = get_imu_vel_stats(data_m2)

        # Calculate GPS Position Stats
        # M1
        # Re-derive errors to be safe with slicing
        gps_pos_errs_1 = np.linalg.norm(data_m1['position_estimates'][:min_len_1, :2] - gps_positions_true[:min_len_1, :2], axis=1)
        mean_1_pos, med_1_pos, std_1_pos, max_1_pos, rmse_1_pos = get_gps_pos_stats(gps_pos_errs_1)

        # M2
        if len(lever_arm_estimates) < len(imu_attitude_estimates):
            last_estimate = lever_arm_estimates[-1]
            ext_la = np.vstack([lever_arm_estimates, np.tile(last_estimate, (len(imu_attitude_estimates) - len(lever_arm_estimates), 1))])
        else:
            ext_la = lever_arm_estimates
        gps_pos_est_2 = np.array([imu_attitude_estimates[i] @ ext_la[i] + imu_position_estimates[i] for i in range(len(imu_position_estimates))])
        gps_pos_errs_2 = np.linalg.norm(gps_pos_est_2[:min_len_2, :2] - gps_positions_true[:min_len_2, :2], axis=1)
        mean_2_pos, med_2_pos, std_2_pos, max_2_pos, rmse_2_pos = get_gps_pos_stats(gps_pos_errs_2)

        def get_gps_vel_stats(data, true_tpva_fr_path = './sim/sim_trajectory_lasso_100Hz.npz'):
            # True GPS velocity is v_IMU,world + wRb@(w_b x l_b))
            true_tpva_fr = np.load(true_tpva_fr_path, allow_pickle=True)['tpva_fr'] # Time (1), Position (3), Velocity (3), Attitude (3), Specific Forces (3), Angular Velocity (3)
            true_times = true_tpva_fr[:,0]
            true_rates = true_tpva_fr[:,-3:]
            est_times = data['times']
            true_imu_attitudes_at_est_times = data['true_attitudes']
            true_imu_velocities_at_est_times = data['true_velocities']
            true_rotation_rates_at_est_times = scipy.interpolate.interp1d(true_times, true_rates, axis=0, bounds_error=False, fill_value="extrapolate")(est_times)
            true_lever_arm = data['true_lever_arm']
            gps_velocity_at_est_times = np.array([true_imu_velocities_at_est_times[i] + true_imu_attitudes_at_est_times[i]@(np.cross(true_rotation_rates_at_est_times[i], true_lever_arm))for i in range(len(est_times))])

            est_imu_velocities = data['velocity_estimates']
            est_imu_attitudes = data['attitude_estimates']
            if 'lever_arm_estimates' in data and data['lever_arm_estimates'] is not None:
                est_lever_arm = data['lever_arm_estimates'][:, 1:]
                est_lever_arm_extended = np.vstack([est_lever_arm, np.tile(est_lever_arm[-1], (len(est_imu_attitudes) - len(est_lever_arm), 1))]) if len(est_lever_arm) < len(est_imu_attitudes) else est_lever_arm
            else:
                estimated_lever_arm = data['lever_arm_init']        
                est_lever_arm_extended = np.tile(estimated_lever_arm, (len(est_imu_attitudes), 1))
            est_gps_velocities = np.array([est_imu_velocities[i] + est_imu_attitudes[i]@(np.cross(true_rotation_rates_at_est_times[i], est_lever_arm_extended[i])) for i in range(len(est_imu_velocities))])
            vel_err = est_gps_velocities - gps_velocity_at_est_times
            horiz_vel_err = np.sqrt(vel_err[:,0]**2 + vel_err[:,1]**2)
            mean_err = np.mean(horiz_vel_err)
            median_err = np.median(horiz_vel_err)
            std_err = np.std(horiz_vel_err)
            max_err = np.max(horiz_vel_err)
            return mean_err, median_err, std_err, max_err
        
        mean_1_gps_vel, med_1_gps_vel, std_1_gps_vel, max_1_gps_vel = get_gps_vel_stats(data_m1)
        mean_2_gps_vel, med_2_gps_vel, std_2_gps_vel, max_2_gps_vel = get_gps_vel_stats(data_m2)

        latex_table = r"""\begin{table}[ht!]
    \centering
    \caption{Velocity Error Comparison Statistics (Experiment 4)}
    \label{tab:exp4_velocity_comparison_stats}
    \begin{subtable}[b]{0.48\linewidth}
        \centering
        \caption{IMU 2D Velocity Error [cm/s]}
        \begin{tabular}{|l|c|c|c|c|}
            \hline
            Method & Mean & Median & Std Dev & Max \\
            \hline
            Ignored & """ + f"{100*mean_1_imu:.2f} & {100*med_1_imu:.2f} & {100*std_1_imu:.2f} & {100*max_1_imu:.2f}" + r""" \\
            \hline
            Estimated & """ + f"{100*mean_2_imu:.2f} & {100*med_2_imu:.2f} & {100*std_2_imu:.2f} & {100*max_2_imu:.2f}" + r""" \\
            \hline
        \end{tabular}
    \end{subtable}
    \hfill
    \begin{subtable}[b]{0.48\linewidth}
        \centering
        \caption{GPS 2D Velocity Error [cm/s]}
        \begin{tabular}{|l|c|c|c|c|}
            \hline
            Method & Mean & Median & Std Dev & Max \\
            \hline
            Ignored & """ + f"{100*mean_1_gps_vel:.2f} & {100*med_1_gps_vel:.2f} & {100*std_1_gps_vel:.2f} & {100*max_1_gps_vel:.2f}" + r""" \\
            \hline
            Estimated & """ + f"{100*mean_2_gps_vel:.2f} & {100*med_2_gps_vel:.2f} & {100*std_2_gps_vel:.2f} & {100*max_2_gps_vel:.2f}" + r""" \\
            \hline
        \end{tabular}
    \end{subtable}
\end{table}"""
        
        with open(os.path.join(output_base_dir, "velocity_metrics_table.tex"), "w") as f:
            f.write(latex_table)

        pos_latex_table = r"""\begin{table}[ht!]
    \caption{GPS Position Error Comparison Statistics (Experiment 4)}
    \label{tab:exp4_gps_position_comparison_stats}
    \centering
    \begin{tabularx}{\linewidth}{
        | >{\centering\arraybackslash}X 
        || >{\centering\arraybackslash}X 
        | >{\centering\arraybackslash}X
        | >{\centering\arraybackslash}X
        | >{\centering\arraybackslash}X
        | >{\centering\arraybackslash}X |
    }
    \hline
    Method & Mean 2D Err [m] & Median 2D Err [m] & Max 2D Err [m] & RMSE 2D Err [m] & Std Dev 2D Err [m] \\
    \thickhline
    Ignored & $""" + f"{mean_1_pos:.4f}$ & ${med_1_pos:.4f}$ & ${max_1_pos:.4f}$ & ${rmse_1_pos:.4f}$ & ${std_1_pos:.4f}" + r"""$\\
    \hline
    Estimated & $""" + f"{mean_2_pos:.4f}$ & ${med_2_pos:.4f}$ & ${max_2_pos:.4f}$ & ${rmse_2_pos:.4f}$ & ${std_2_pos:.4f}" + r"""$\\
    \hline
    \end{tabularx}
\end{table}"""

        with open(os.path.join(output_base_dir, "gps_position_metrics_table.tex"), "w") as f:
            f.write(pos_latex_table)

        # 3. Lever Arm Estimate Plot (Method 2)
        plt.figure(figsize=(10, 6))
        times_2 = data_m2['times']
        if len(times_2) > 0: times_2 = times_2 - times_2[0]
        la_est = data_m2['lever_arm_estimates'][:, 1:]
        min_len_la = min(len(times_2), len(la_est))

        plt.plot(times_2[:min_len_la], la_est[:min_len_la, 0], label=r'$\hat{l}_{b,x}$', color='r')
        plt.plot(times_2[:min_len_la], la_est[:min_len_la, 1], label=r'$\hat{l}_{b,y}$', color='g')
        plt.plot(times_2[:min_len_la], la_est[:min_len_la, 2], label=r'$\hat{l}_{b,z}$', color='b')

        plt.axhline(true_lever_arm[0], color='r', linestyle='--', label=r'$l_{b,x}$')
        plt.axhline(true_lever_arm[1], color='g', linestyle='--', label=r'$l_{b,y}$')
        plt.axhline(true_lever_arm[2], color='b', linestyle='--', label=r'$l_{b,z}$')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Lever Arm Estimate [m]')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_base_dir, 'method_2_lever_arm_estimates.png'))
        plt.close()

        print("Generated the requested plots and LaTeX velocity table.")

if __name__ == "__main__":
    run_ignored_vs_estimated_study()
