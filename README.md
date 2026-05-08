# Lever Arm Estimation in a Tightly-Coupled GPS/IMU Factor Graph

This repository provides the code for implementing the approach for online navigation and lever arm estimation presented in "Lever Arm Estimation in a Tightly-Coupled GPS/IMU Factor Graph" by K. Leland, C. Taylor, and F. Van Graas.

- `pyproject.toml` lists the dependencies, which can be installed via `uv`.
- `results/ignored_vs_estimated_lever_arm` reproduces the first study of the paper, demonstrating the importance of estimating the lever arm.
- `experiment/lever_arm_estimation_with_collected_data.py` reproduces the fourth study of the paper, demonstrating the presented approach on real, collected data.
- The second and third studies from the paper were ran on a total of 340 Monte Carlo iterations of simulated measurement data, and that simulated data is not included here in order avoid storing the large amount of data. Thus, the execution scripts for those studies are also not included, though an example usage of the tightly-coupled EKF is in `experiment/ekf_estimation_gps_imu.py` and an example usage of the loosely-coupled FGO approach is in `experiment/sliding_optimization_gps_imu_lc.py`.
