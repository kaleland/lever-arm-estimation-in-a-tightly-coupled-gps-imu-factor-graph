
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm



def simulate_rpy_from_path(
        velocities_t: np.ndarray,
        accelerations_t: np.ndarray,
        gravities_t: np.ndarray,
        init_xb = np.array([1.0, 0.0, 0.0]),
        init_yb = np.array([0.0, 1.0, 0.0]),
) -> np.ndarray:
    """Calculate roll, pitch, yaw (RPY) angles from a given path defined by
    positions, velocities, accelerations, and gravities in a tangent frame's 
    coordinates.

    Assumptions:
    1. No Sideslip: The body x-axis is aligned with the velocity vector.
    2. Coordinated Turn: The body z-axis is aligned with the apparent gravity 
       vector (gravity - acceleration), ensuring zero lateral G-force.

    Returns:
        rpy_t: np.ndarray
            An array of shape (N, 3) where N is the number of time steps, 
            containing the roll, pitch, and yaw angles at each time step.

    """
    # 0. Input validation and Default Gravity
    if gravities_t is None:
        gravities_t = np.zeros_like(velocities_t)
        gravities_t[:, 2] = 9.80665 # Standard gravity pointing Down

    # 1. Calculate Body X-Axis (Longitudinal)
    # Assumption: No sideslip, nose points along velocity vector
    norm_v = np.linalg.norm(velocities_t, axis=1, keepdims=True)

    # Handle zero velocity to avoid division by zero (hold previous or default to North)
    # For this snippet, we assume V > 0.
    x_b = velocities_t / norm_v
    # Find all velocities with near-zero magnitude and replace their directions
    zero_velocity_indices = np.where(norm_v.flatten() < 1e-6)[0]
    for idx in tqdm(zero_velocity_indices, desc="Processing zero velocities", leave=False):
        if idx == 0:
            # If the first velocity is zero, use the initial direction
            x_b[idx] = init_xb
            continue

        # Find previous defined velocity and next defined velocity
        prev_vel_idx = idx - 1
        found_next_vel_idx = False
        next_vel_idx = idx + 1
        while not found_next_vel_idx and next_vel_idx < len(velocities_t):
            if next_vel_idx not in zero_velocity_indices:
                found_next_vel_idx = True
            else:
                next_vel_idx += 1
        if not found_next_vel_idx:
            # If no next defined velocity, use previous defined velocity
            x_b[idx] = x_b[prev_vel_idx]
        else:
            prev_dir = x_b[prev_vel_idx]
            Rot_prev_dir = Rotation.align_vectors([prev_dir], [[1.0, 0.0, 0.0]])[0]
            next_dir = x_b[next_vel_idx]
            Rot_next_dir = Rotation.align_vectors([next_dir], [[1.0, 0.0, 0.0]])[0]
            rotations = Rotation.concatenate([Rot_prev_dir, Rot_next_dir])
            slerp = Slerp([prev_vel_idx, next_vel_idx], rotations)
            x_b[idx] = slerp(idx).apply([1.0, 0.0, 0.0])[0]

    # 2. Calculate Body Y-Axis (Lateral/Right Wing)
    # Assumption: Coordinated Turn.
    # The pilot feels a net force vector f = g - a.
    # The "floor" of the plane (Body XY plane) should be perpendicular to this vector
    # if we want 0 lateral G.
    # Therefore, the Right Wing (Y_b) is perpendicular to both Forward (X_b)
    # and the Apparent Gravity vector (g - a).
    g_apparent = gravities_t - accelerations_t
    
    # Cross product: (g_apparent) x (Forward) results in a vector pointing Right
    # Check Right Hand Rule: Down x North = East (Right). Correct.
    y_b_raw = np.cross(g_apparent, x_b)
    norm_y = np.linalg.norm(y_b_raw, axis=1, keepdims=True)
    y_b = y_b_raw / norm_y
    # Find all zero magnitude g_apparent and fix them
    zero_g_apparent_indices = np.where(norm_y.flatten() < 1e-6)[0]
    for idx in tqdm(zero_g_apparent_indices, desc="Processing zero g_apparent", leave=False):
        if idx == 0:
            # If the first g_apparent is zero, use the initial direction
            y_b[idx] = init_yb
            continue
        # Find previous defined y_b and next defined y_b
        prev_yb_idx = idx - 1
        found_next_yb_idx = False
        next_yb_idx = idx + 1
        while not found_next_yb_idx and next_yb_idx < len(y_b):
            if next_yb_idx not in zero_g_apparent_indices:
                found_next_yb_idx = True
            else:
                next_yb_idx += 1
        if not found_next_yb_idx:
            # If no next defined y_b, use previous defined y_b
            y_b[idx] = y_b[prev_yb_idx]
        else:
            prev_dir = y_b[prev_yb_idx]
            Rot_prev_dir = Rotation.align_vectors([prev_dir], [[0.0, 1.0, 0.0]])[0]
            next_dir = y_b[next_yb_idx]
            Rot_next_dir = Rotation.align_vectors([next_dir], [[0.0, 1.0, 0.0]])[0]
            rotations = Rotation.concatenate([Rot_prev_dir, Rot_next_dir])
            slerp = Slerp([prev_yb_idx, next_yb_idx], rotations)
            y_b[idx] = slerp(idx).apply([0.0, 1.0, 0.0])[0]

    # 3. Calculate Body Z-Axis (Vertical/Down)
    # Complete the orthonormal basis: Z = X cross Y
    z_b = np.cross(x_b, y_b)

    # 4. Extract Euler Angles from the Rotation Matrix
    # We now have the Rotation Matrix R_nb (NED to Body) components.
    # The columns of R_nb are the body axes expressed in NED.
    # R_nb = [x_b | y_b | z_b]
    
    # Extract components for clarity (x_b_n means X_body expressed in North, etc.)
    # x_b = [r11, r21, r31]^T
    # y_b = [r12, r22, r32]^T
    # z_b = [r13, r23, r33]^T

    R_wb = np.stack([x_b, y_b, z_b], axis=2)  # Shape (N, 3, 3).
    # Standard extraction for NED (Z-Y-X sequence)
    
    # Pitch (Theta): -asin(r31)
    # Clamp value to [-1, 1] to prevent NaN from float precision errors
    r31 = R_wb[:, 2, 0]
    pitch = np.arcsin(-np.clip(r31, -1.0, 1.0))

    # Yaw (Psi): atan2(r21, r11)
    r21 = R_wb[:, 1, 0]
    r11 = R_wb[:, 0, 0]
    yaw = np.arctan2(r21, r11)

    # Roll (Phi): atan2(r32, r33)
    r32 = R_wb[:, 2, 1]
    r33 = R_wb[:, 2, 2]
    roll = np.arctan2(r32, r33)

    return np.stack([roll, pitch, yaw], axis=1)


def simulate_attitude_from_path(
        velocities_t: np.ndarray,
        accelerations_t: np.ndarray,
        gravities_t: np.ndarray,
        init_xb = np.array([1.0, 0.0, 0.0]), # Column 0 of R_wb at t=0
        init_yb = np.array([0.0, 1.0, 0.0]), # Column 1 of R_wb at t=0
) -> np.ndarray:
    """Calculate roll, pitch, yaw (RPY) angles from a given path defined by
    positions, velocities, accelerations, and gravities in a tangent frame's 
    coordinates.

    Assumptions:
    1. No Sideslip: The body x-axis is aligned with the velocity vector.
    2. Coordinated Turn: The body z-axis is aligned with the apparent gravity 
       vector (gravity - acceleration), ensuring zero lateral G-force.

    Returns:
        Rwb_t: np.ndarray
            An array of shape (N, 3,3) where N is the number of time steps, 
            containing the rotation matrices from body to world frame at each time step.
            Rwb@p_body = p_world

    """
    # 0. Input validation and Default Gravity
    if gravities_t is None:
        gravities_t = np.zeros_like(velocities_t)
        gravities_t[:, 2] = 9.80665 # Standard gravity pointing Down

    # 1. Calculate Body X-Axis (Longitudinal)
    # Assumption: No sideslip, nose points along velocity vector
    norm_v = np.linalg.norm(velocities_t, axis=1, keepdims=True)

    # Handle zero velocity to avoid division by zero (hold previous or default to North)
    # For this snippet, we assume V > 0.
    x_b = velocities_t / norm_v
    # Find all velocities with near-zero magnitude and replace their directions
    zero_velocity_indices = np.where(norm_v.flatten() < 1e-6)[0]
    next_vel_idx = None
    for idx in tqdm(zero_velocity_indices, desc="Processing zero velocities", leave=False):
        if idx == 0:
            # If the first velocity is zero, use the initial direction
            x_b[idx] = init_xb
            continue

        # Find previous defined velocity and next defined velocity
        prev_vel_idx = idx - 1
        found_next_vel_idx = False
        if next_vel_idx == None or next_vel_idx <= idx: # Only search for a new next_vel_idx if we haven't already found one later than the current idx
            next_vel_idx = idx + 1
        while not found_next_vel_idx and next_vel_idx < len(velocities_t):
            if next_vel_idx not in zero_velocity_indices:
                found_next_vel_idx = True
            else:
                next_vel_idx += 1
        if not found_next_vel_idx:
            # If no next defined velocity, use previous defined velocity
            x_b[idx] = x_b[prev_vel_idx]
        else:
            prev_dir = x_b[prev_vel_idx]
            Rot_prev_dir = Rotation.align_vectors([prev_dir], [[1.0, 0.0, 0.0]])[0]
            next_dir = x_b[next_vel_idx]
            Rot_next_dir = Rotation.align_vectors([next_dir], [[1.0, 0.0, 0.0]])[0]
            rotations = Rotation.concatenate([Rot_prev_dir, Rot_next_dir])
            slerp = Slerp([prev_vel_idx, next_vel_idx], rotations)
            x_b[idx] = slerp(idx).apply([1.0, 0.0, 0.0])[0]

    # 2. Calculate Body Y-Axis (Lateral/Right Wing)
    # Assumption: Coordinated Turn.
    # The pilot feels a net force vector f = g - a.
    # The "floor" of the plane (Body XY plane) should be perpendicular to this vector
    # if we want 0 lateral G.
    # Therefore, the Right Wing (Y_b) is perpendicular to both Forward (X_b)
    # and the Apparent Gravity vector (g - a).
    g_apparent = gravities_t - accelerations_t
    
    # Cross product: (g_apparent) x (Forward) results in a vector pointing Right
    # Check Right Hand Rule: Down x North = East (Right). Correct.
    y_b_raw = np.cross(g_apparent, x_b)
    norm_y = np.linalg.norm(y_b_raw, axis=1, keepdims=True)
    y_b = y_b_raw / norm_y
    # Find all zero magnitude g_apparent and fix them
    zero_g_apparent_indices = np.where(norm_y.flatten() < 1e-6)[0]
    for idx in tqdm(zero_g_apparent_indices, desc="Processing zero g_apparent", leave=False):
        if idx == 0:
            # If the first g_apparent is zero, use the initial direction
            y_b[idx] = init_yb
            continue
        # Find previous defined y_b and next defined y_b
        prev_yb_idx = idx - 1
        found_next_yb_idx = False
        next_yb_idx = idx + 1
        while not found_next_yb_idx and next_yb_idx < len(y_b):
            if next_yb_idx not in zero_g_apparent_indices:
                found_next_yb_idx = True
            else:
                next_yb_idx += 1
        if not found_next_yb_idx:
            # If no next defined y_b, use previous defined y_b
            y_b[idx] = y_b[prev_yb_idx]
        else:
            prev_dir = y_b[prev_yb_idx]
            Rot_prev_dir = Rotation.align_vectors([prev_dir], [[0.0, 1.0, 0.0]])[0]
            next_dir = y_b[next_yb_idx]
            Rot_next_dir = Rotation.align_vectors([next_dir], [[0.0, 1.0, 0.0]])[0]
            rotations = Rotation.concatenate([Rot_prev_dir, Rot_next_dir])
            slerp = Slerp([prev_yb_idx, next_yb_idx], rotations)
            y_b[idx] = slerp(idx).apply([0.0, 1.0, 0.0])[0]

    # 3. Calculate Body Z-Axis (Vertical/Down)
    # Complete the orthonormal basis: Z = X cross Y
    z_b = np.cross(x_b, y_b)

    # 4. Extract Euler Angles from the Rotation Matrix
    # We now have the Rotation Matrix R_nb (NED to Body) components.
    # The columns of R_nb are the body axes expressed in NED.
    # R_nb = [x_b | y_b | z_b]
    
    # Extract components for clarity (x_b_n means X_body expressed in North, etc.)
    # x_b = [r11, r21, r31]^T
    # y_b = [r12, r22, r32]^T
    # z_b = [r13, r23, r33]^T

    R_wb = np.stack([x_b, y_b, z_b], axis=2)  # Shape (N, 3, 3).
    return R_wb

