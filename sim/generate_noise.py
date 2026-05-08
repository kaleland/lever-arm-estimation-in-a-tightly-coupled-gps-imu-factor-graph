import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.add_imu_noise import noisy_imu_from_tpva_fr_file

def generate_noise():
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_dir = os.path.join(base_dir, 'sim')
    
    # Define tasks: (input_traj_file, output_directory, sensor_type)
    tasks = [
        ('sim_trajectory_lasso_100Hz.npz', os.path.join(sim_dir, 'lasso', 'tactical_grade'), 'tactical_grade'),
        ('sim_trajectory_lasso_250Hz.npz', os.path.join(sim_dir, 'lasso', 'navigation_grade'), 'navigation_grade'),
    ]

    for traj_file, output_dir, sensor in tasks:
        input_path = os.path.join(sim_dir, traj_file)
        output_path = os.path.join(output_dir, 'imu.npz')
        
        if not os.path.exists(input_path):
            print(f"Error: Input file NOT FOUND: {input_path}")
            continue
            
        print(f"Generating noise for {output_dir} using {sensor}...")
        try:
            noisy_imu_from_tpva_fr_file(input_path, sensor, output_path)
            print("Done.\n")
        except Exception as e:
            print(f"Failed to process {traj_file}: {e}\n")

if __name__ == "__main__":
    generate_noise()
