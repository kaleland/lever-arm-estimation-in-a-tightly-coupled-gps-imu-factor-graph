import numpy as np

from experiment.sliding_optimization_gps_imu_golf_cart import sliding_optimization_gps_imu_golf_cart
from experiment.plot_golf_cart_results import main as plot_golf_cart_results

if __name__ == "__main__":
    dur = 30.0
    sliding_optimization_gps_imu_golf_cart(
        window_dur=dur,
        save=True,
        true_lever_arm=np.array([0.0508, -0.854075, -0.0384625]), # Set to actual true lever arm
        lever_arm_init = np.zeros(3),
        unknown_lever_arm=True,
    )
    plot_golf_cart_results()


