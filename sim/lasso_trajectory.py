import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import r3f
'''
The generated path includes takeoff, climbing, transit to orbit, elliptical
orbits, and then a return along the same pattern. The path is generated with a
CubicSpline for analytically determined position, velocity, and acceleration at
each time step. The waypoints and durations are tuned to roughly match the
realistic limits and performance of a single-engine propeller aircraft
(e.g., Cessna 172), including climbing rate, g-limits, and cruising speed.
The output positions, velocities, and accelerations are in the ECEF frame and
or in a NED curvilinear frame with a specified initial latitude, longitude, and
height.
'''


def generate_lasso_trajectory(
        altitude=600.0,
        orbits=3,
        duration=1140.0,
        dt=0.1,
        approach_leg_length=2000,
        major_axis = 2500.0,
        minor_axis = 1250.0,
        orbit_center = (8000.0,6000.0),
        takeoff_length = 800.0,
        llh0 = np.zeros(3)
        ):
    """Generates a flight path that climbs, performs elliptical orbits, and returns to land.
    
    Args:
        major_axis (float): Major axis length of the elliptical orbit in meters.
        minor_axis (float): Minor axis length of the elliptical orbit in meters.
        altitude (float): Altitude of the orbit in meters (positive value).
        orbits (int): Number of full circles to complete.
        duration (float): Total duration of the active flight phase.
        dt (float): Time step.
        approach_leg_length (float): Length of straight leg before entering orbit. 
                                     If None, defaults to radius.
        orbit_center (tuple): (x,y) coordinates of the center of the elliptical orbit.
        llh0 (np.array): Initial latitude, longitude, and height in radians and meters.

    """
    # Phase 1: Takeoff (2%), Climb (5%), and Transit to Orbit (15%)
    t_takeoff = 0.02 * duration
    t_approach = 0.04 * duration
    t_orbit = 0.15 * duration
    orbit_entry_point = (orbit_center[0], orbit_center[1]-major_axis)
    
    # Calculate spacing for smooth horizontal velocity during climb
    climb_x_spacing = (approach_leg_length - takeoff_length) * 0.001 * duration / (t_approach - t_takeoff)
    wp_takeoff = [
        [0.0, 0.0, 0.0, 0.0],              # Start
        [t_takeoff, takeoff_length, 0, 0.0],  # End of takeoff roll - still on ground
        [t_takeoff + 0.001*duration, takeoff_length + climb_x_spacing, 0, -0.1],  # Immediately start descending (negative z only)
        # [t_approach - 0.001*duration, approach_leg_length - climb_x_spacing, 0, -altitude/4 + 0.1],  # Just before reaching cruise altitude
        [t_approach, approach_leg_length, 0, -altitude*0.125],  # Finish climb to orbit altitude
        [t_approach + (t_orbit - t_approach)*0.75,(orbit_entry_point[0]-approach_leg_length)*0.75+approach_leg_length, orbit_entry_point[1]*0.75, -altitude*0.8], # Midway to orbit
        [t_orbit, orbit_entry_point[0], orbit_entry_point[1], -altitude]  # Enter orbit
    ]
    
    # Phase 2: Orbits (15% to 85%)
    orbit_start_time = t_orbit
    orbit_duration = 0.70 * duration
    orbit_end_time = orbit_start_time + orbit_duration
    cx, cy = orbit_center
    
    # Generate waypoints around the ellipse spaced by arc length for constant speed
    # Space waypoints evenly in angle, but time them according to arc length
    points_per_orbit = 100
    total_orbit_points = orbits * points_per_orbit
    early_departure_points = points_per_orbit // 10  # Leave last 10% of points for smooth exit
    
    # First, compute arc length for evenly spaced angles
    theta_values = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*orbits, total_orbit_points + 1)
    
    # Compute positions and arc lengths between consecutive points
    arc_lengths = [0.0]
    for i in range(len(theta_values) - 1):
        x1 = cx + minor_axis * np.cos(theta_values[i])
        y1 = cy + major_axis * np.sin(theta_values[i])
        x2 = cx + minor_axis * np.cos(theta_values[i+1])
        y2 = cy + major_axis * np.sin(theta_values[i+1])
        ds = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        arc_lengths.append(arc_lengths[-1] + ds)
    
    total_arc_length = arc_lengths[-1]
    
    # Now generate waypoints with evenly spaced angles but time proportional to arc length
    wp_orbit = []
    for i in range(1, total_orbit_points + 1-early_departure_points):
        theta = theta_values[i]
        
        # Time is proportional to cumulative arc length
        s = arc_lengths[i]
        t = orbit_start_time + (s / total_arc_length) * orbit_duration
        
        px = cx + minor_axis * np.cos(theta)
        py = cy + major_axis * np.sin(theta)
        pz = -altitude
        
        wp_orbit.append([t, px, py, pz])


    # Remove the first early departure point to allow smooth entry
    wp_orbit = wp_orbit[early_departure_points:]
        
    # Phase 3: Return and Land (85% to 100%)
    # After orbits, we are back at orbit entry point.
    # We need to turn around and transit to (approach_leg_length, 0, -altitude).
    # Simple approach: Fly a "tear drop".
    
    t_return_from_orbit = orbit_end_time
    
    # Calculate spacing for smooth horizontal velocity during descent and landing
    descent_x_spacing = (approach_leg_length - takeoff_length) * 0.001 * duration / (0.95 * duration - 0.90 * duration)
    landing_x_spacing = (takeoff_length - 0.0) * 0.001 * duration / (0.98 * duration - 0.95 * duration)
    
    wp_return = [
        # Wide turn to align for landing or direct spline return
        [0.87 * duration, (orbit_entry_point[0]-approach_leg_length)*0.75+approach_leg_length, orbit_entry_point[1]*0.75, -altitude*0.8], # Base leg
        [duration - t_approach, approach_leg_length, 0, -altitude*0.125],  # Just before final approach
        # [duration-(t_approach-0.001*duration), approach_leg_length - climb_x_spacing, 0, -altitude + 0.1],  # Final approach at cruise altitude
        [duration - (t_takeoff + 0.001*duration), takeoff_length + climb_x_spacing, 0, -0.1],  # Just before touchdown - still slightly above ground
        [duration-t_takeoff, takeoff_length, 0, 0.0],  # Touchdown
        [1.00 * duration, 0.0, 0.0, 0.0]   # Landed
    ]
    
    # Combine all waypoints
    waypoints = np.vstack([wp_takeoff, wp_orbit, wp_return])
    
    # Sort just in case, though they should be ordered
    waypoints = waypoints[waypoints[:, 0].argsort()]
    
    wp_times = waypoints[:, 0]
    wp_pos = waypoints[:, 1:]

    pe0 = r3f.geodetic_to_ecef(llh0)
    wp_pos_ecef = r3f.curvilinear_to_ecef(wp_pos, pe0 = pe0)
    
    # Generate Splines
    cs_curvilinear = CubicSpline(wp_times, wp_pos, bc_type='clamped')
    cs_ecef = CubicSpline(wp_times, wp_pos_ecef, bc_type='clamped')

    
    
    # t_flight = np.arange(0, duration, dt)
    t_flight = np.arange(0.0, duration+dt , dt)
    pos_flight = cs_curvilinear(t_flight)
    vel_flight = cs_curvilinear(t_flight, nu=1)
    acc_flight = cs_curvilinear(t_flight, nu=2)

    p_e = cs_ecef(t_flight)
    v_e = cs_ecef(t_flight, nu=1)
    a_e = cs_ecef(t_flight, nu=2)
    
    trajectory_data_curvilinear = _add_static_start(pos_flight, vel_flight, acc_flight, t_flight, duration, static_duration =60.0)
    trajectory_data_curvilinear['waypoints'] = waypoints  # Add waypoints to the returned data

    trajectory_data_ecef = _add_static_start(p_e, v_e, a_e, t_flight, duration, static_duration =60.0)
    trajectory_data_ecef['waypoints'] = waypoints  # Add waypoints to the returned data

    # Add initial orientation data
    initial_Rnb = np.eye(3)
    trajectory_data_curvilinear['initial_Rnb'] = initial_Rnb

    Rne = r3f.dcm_ecef_to_navigation(llh0[0],llh0[1])
    initial_Reb = Rne.T @ initial_Rnb
    trajectory_data_ecef['initial_Reb'] = initial_Reb


    return trajectory_data_curvilinear, trajectory_data_ecef

def _add_static_start(pos_flight, vel_flight, acc_flight, t_flight, duration, static_duration=60.0):
    """Helper to add start/end static periods."""
    dt = t_flight[1] - t_flight[0]
    t_static = np.arange(0, static_duration, dt)
    zeros_static = np.zeros((len(t_static), 3))
    
    pos_start = np.zeros_like(zeros_static)+ pos_flight[0]
    final_pos = pos_flight[-1]

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


def plot_trajectory(trajectory_data, show_waypoints=False):
    """Plot the trajectory data showing position, velocity, and acceleration.
    
    Args:
        trajectory_data (dict): Dictionary with 'time', 'positions', 'velocities', 'accelerations', 'waypoints'
        show_waypoints (bool): Whether to display waypoints on the plots (default: False)

    """
    time = trajectory_data['time']
    positions = trajectory_data['positions']
    velocities = trajectory_data['velocities']
    accelerations = trajectory_data['accelerations']
    waypoints = trajectory_data.get('waypoints', None)
    
    # Calculate speed (norm of velocity)
    speed = np.linalg.norm(velocities, axis=1)
    
    # Create figure with subplots for time series
    fig1 = plt.figure(figsize=(15, 10))
    
    # Position plots - one for each axis
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time, positions[:, 0], 'b-', linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position (m)')
    ax1.set_title('X Position')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time, positions[:, 1], 'b-', linewidth=1)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Y Position')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(time, positions[:, 2], 'b-', linewidth=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_title('Z Position (Altitude)')
    ax3.grid(True, alpha=0.3)
    
    # Velocity plots
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time, velocities[:, 0], 'r-', linewidth=1)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X Velocity (m/s)')
    ax4.set_title('X Velocity')
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(time, velocities[:, 1], 'r-', linewidth=1)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Y Velocity (m/s)')
    ax5.set_title('Y Velocity')
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(time, velocities[:, 2], 'r-', linewidth=1)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Z Velocity (m/s)')
    ax6.set_title('Z Velocity')
    ax6.grid(True, alpha=0.3)
    
    # Acceleration plots
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(time, accelerations[:, 0], 'g-', linewidth=1)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('X Acceleration (m/s²)')
    ax7.set_title('X Acceleration')
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(time, accelerations[:, 1], 'g-', linewidth=1)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Y Acceleration (m/s²)')
    ax8.set_title('Y Acceleration')
    ax8.grid(True, alpha=0.3)
    
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(time, accelerations[:, 2], 'g-', linewidth=1)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Z Acceleration (m/s²)')
    ax9.set_title('Z Acceleration')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create separate figure for speed
    fig2 = plt.figure(figsize=(12, 5))
    ax_speed = plt.subplot(1, 1, 1)
    ax_speed.plot(time, speed, 'purple', linewidth=2)
    
    # Add waypoints to speed plot if requested
    if show_waypoints and waypoints is not None:
        wp_times = waypoints[:, 0]
        # Calculate speed at waypoint times by interpolating
        wp_speeds = []
        for wp_t in wp_times:
            idx = np.argmin(np.abs(time - wp_t))
            wp_speeds.append(speed[idx])
        ax_speed.plot(wp_times, wp_speeds, 'gs', markersize=4, label='Waypoints')
        ax_speed.legend(loc='best')
    
    ax_speed.set_xlabel('Time (s)', fontsize=12)
    ax_speed.set_ylabel('Speed (m/s)', fontsize=12)
    ax_speed.set_title('Speed Over Time (Velocity Magnitude)', fontsize=14)
    ax_speed.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create separate figure for XY position with percentage markers
    fig3 = plt.figure(figsize=(10, 10))
    ax_xy = plt.subplot(1, 1, 1)
    
    # Plot the trajectory
    ax_xy.plot(positions[:, 1], positions[:, 0], 'b-', linewidth=2, label='Flight Path')
    
    # Plot waypoints if requested
    if show_waypoints and waypoints is not None:
        ax_xy.plot( waypoints[:, 2], waypoints[:, 1], 'gs', markersize=10, 
                  markeredgecolor='darkgreen', markeredgewidth=2, 
                  label='Waypoints', zorder=5)
    
    # Add markers at every 5% of the path
    num_points = len(positions)
    percentages = np.arange(0, 101, 5)  # 0%, 5%, 10%, ..., 100%
    
    for pct in percentages:
        idx = int((pct / 100.0) * (num_points - 1))
        x, y = positions[idx, 0], positions[idx, 1]
        
        # Plot marker
        ax_xy.plot(y, x, 'ro', markersize=8, markeredgecolor='darkred', markeredgewidth=1.5)
        
        # Add label
        ax_xy.annotate(f'{pct}%', 
                      xy=(y, x), 
                      xytext=(10, 10), 
                      textcoords='offset points',
                      fontsize=9,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1))
    
    ax_xy.set_xlabel('East (Y) Position (m)', fontsize=12)
    ax_xy.set_ylabel('North (X) Position (m)', fontsize=12)
    ax_xy.set_title('North-East Position - Top View with Path Percentage Markers', fontsize=14)
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis('equal')
    ax_xy.legend(loc='best')
    
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # Generate and plot a sample trajectory
    trajectory_curvilinear, trajectory_ecef = generate_lasso_trajectory(   )
    
    print(f"Generated trajectory with {len(trajectory_curvilinear['time'])} points")
    print(f"Duration: {trajectory_curvilinear['time'][-1]:.1f} seconds")
    print(f"Max altitude: {-np.min(trajectory_curvilinear['positions'][:, 2]):.1f} meters")
    
    plot_trajectory(trajectory_curvilinear, show_waypoints=True)

