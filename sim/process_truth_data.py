
import numpy as np
import r3f


def process_simulated_truth_data(tpva_fr_path: str, output_npz_path: str):
    """Process simulated truth data from tpva_fr file and save in the same format
    as process_truth_data() from process_data.py.
    
    Parameters
    ----------
    tpva_fr_path : str
        Path to the tpva_fr .npz file containing trajectory data
    output_npz_path : str
        Path where the output truth.npz file will be saved
        
    Notes
    -----
    The tpva_fr array has columns:
    [time, lat, lon, height, vn, ve, vd, roll, pitch, yaw, fx, fy, fz, wx, wy, wz]
    
    The output truth.npz file will contain:
    - times: (N,) array of time values
    - llh: (N, 3) array of geodetic coordinates [lat, lon, height] in [rad, rad, m]
    - pw: (N, 3) array of positions in local tangent frame (NED)
    - eTw: (4, 4) homogeneous transformation matrix from local frame to ECEF
    - ned_std: (N, 3) array of position standard deviations (set to zeros for simulation)
    - ned_cov_world_frame: (N, 3, 3) array of covariance matrices (set to zeros)
    - true_v_ECEF: (N, 3) array of velocities in ECEF frame
    - vw: (N, 3) array of velocities in local tangent frame
    - rpy: (N, 3) array of roll, pitch, yaw in body frame w.r.t. local frame

    """
    # Load tpva_fr data
    print(f"Loading trajectory data from {tpva_fr_path}")
    tpva_fr_data = np.load(tpva_fr_path)
    tpva_fr = tpva_fr_data['tpva_fr']
    
    # Extract data
    # tpva_fr columns: [time, lat, lon, height, vn, ve, vd, roll, pitch, yaw, fx, fy, fz, wx, wy, wz]
    times = tpva_fr[:, 0]
    llh = tpva_fr[:, 1:4].copy()  # [lat, lon, height] - already in radians
    vne = tpva_fr[:, 4:7]  # [vn, ve, vd]
    rpy_Rbn = tpva_fr[:, 7:10]  # [roll, pitch, yaw] - already in radians

    
    
    N = len(times)
    
    # Define local tangent frame (w-frame) based on starting position
    # Use NED frame aligned with ECEF at the starting location
    llh_start = llh[0, :]
    wRe = r3f.dcm_ecef_to_navigation(llh_start[0], llh_start[1])  # NED to ECEF rotation
    pew = r3f.geodetic_to_ecef(llh_start)  # Position of w-frame origin in ECEF
    
    # Create homogeneous transformation matrix eTw (ECEF to local tangent frame)
    eTw = np.vstack([
        np.hstack([wRe.T, pew.reshape(-1, 1)]),
        np.array([0, 0, 0, 1])
    ])
    
    # Compute positions in local tangent frame (pw)
    pw = r3f.geodetic_to_tangent(llh, llh_start)
    
    # Convert velocities from NED to ECEF frame
    true_v_ECEF = np.zeros((N, 3))
    vw = np.zeros((N, 3))
    
    for i in range(N):
        # NED to ECEF rotation at current position
        nRe_i = r3f.dcm_ecef_to_navigation(llh[i, 0], llh[i, 1])
        
        # Convert velocity from NED to ECEF
        v_ne = vne[i, :]
        true_v_ECEF[i] = nRe_i.T @ v_ne
        
        # Convert velocity from ECEF to local tangent frame
        vw[i] = wRe @ true_v_ECEF[i]
    
    # For simulation data, we don't have real position uncertainty
    # Set ned_std to zeros
    ned_std = np.zeros((N, 3))
    
    # Compute ned_cov_world_frame (will be zeros since ned_std is zero)
    eRw = wRe.T
    nRe_path = r3f.dcm_ecef_to_navigation(llh[:, 0], llh[:, 1])
    ned_cov_world_frame = np.array([
        (nRe_path[i] @ eRw).T @ np.diag(ned_std[i]**2) @ (nRe_path[i] @ eRw)
        for i in range(N)
    ])

    # Compute RPY in world frame
    Rbn = r3f.rpy_to_dcm(rpy_Rbn)  # Body to navigation frame DCMs
    Rne = r3f.dcm_ecef_to_navigation(llh[:, 0], llh[:, 1])
    Rwb = np.array([(Rbn[i]@Rne[i]@eRw).T for i in range(Rbn.shape[0])])
    rpy_world_frame = np.array([r3f.dcm_to_rpy(Rwb[i].T) for i in range(N)])

    
    # Save to .npz file
    print(f"Saving truth data to {output_npz_path}")
    np.savez(
        output_npz_path,
        times=times,
        llh=llh,
        pw=pw,
        eTw=eTw,
        ned_std=ned_std,
        ned_cov_world_frame=ned_cov_world_frame,
        true_v_ECEF=true_v_ECEF,
        vw=vw,
        rpy=rpy_world_frame
    )
    
    print(f"Successfully processed {N} truth samples")
    print(f"Time range: {times[0]:.1f} to {times[-1]:.1f} seconds")
    print(f"Starting position (LLH): lat={np.rad2deg(llh[0,0]):.6f}°, lon={np.rad2deg(llh[0,1]):.6f}°, h={llh[0,2]:.2f}m")
    
    return times, llh, pw, eTw, ned_std, ned_cov_world_frame, true_v_ECEF, vw, rpy_world_frame


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python process_truth_data.py <tpva_fr_npz> <output_truth_npz>")
        print("\nExample:")
        print("  python process_truth_data.py \\")
        print("    sim/sim_trajectory_lasso_100Hz.npz \\")
        print("    sim/lasso/tactical_grade/truth.npz")
    else:
        process_simulated_truth_data(sys.argv[1], sys.argv[2])
