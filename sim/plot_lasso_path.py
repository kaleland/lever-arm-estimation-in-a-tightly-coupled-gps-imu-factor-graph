import numpy as np
import matplotlib.pyplot as plt
import os

def geodetic_to_ecef(lat, lon, h):
    
    a = 6378137.0
    f = 1/298.257223563
    e2 = 2*f - f*f
    s_lat = np.sin(lat)
    c_lat = np.cos(lat)
    s_lon = np.sin(lon)
    c_lon = np.cos(lon)
    N = a / np.sqrt(1 - e2 * s_lat**2)
    x = (N + h) * c_lat * c_lon
    y = (N + h) * c_lat * s_lon
    z = (N * (1 - e2) + h) * s_lat
    return np.stack([x, y, z], axis=1)

# Set styling
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams['mathtext.fontset'] = 'stix'

# Load trajectory data
data_file = 'sim/sim_trajectory_lasso_100Hz.npz'
data = np.load(data_file)
tpva_fr = data['tpva_fr']

times = tpva_fr[:, 0]
llh = tpva_fr[:, 1:4]

# Convert LLH to ECEF
ecef = geodetic_to_ecef(llh[:, 0], llh[:, 1], llh[:, 2])

# Relative ECEF from start
ecef0 = ecef[0, :]
delta_ecef = ecef - ecef0

# Transform to Local NED Frame
lat0 = llh[0, 0]
lon0 = llh[0, 1]
s_lat = np.sin(lat0)
c_lat = np.cos(lat0)
s_lon = np.sin(lon0)
c_lon = np.cos(lon0)
Rne0 = np.array([
    [-s_lat * c_lon, -s_lat * s_lon,  c_lat],
    [-s_lon,          c_lon,          0.0],
    [-c_lat * c_lon, -c_lat * s_lon, -s_lat]
])

pos_ned = (Rne0 @ delta_ecef.T).T
pos_n = pos_ned[:, 0]
pos_e = pos_ned[:, 1]

output_dir = 'results/experiment_4_final'
os.makedirs(output_dir, exist_ok=True)

# Figure 1: Horizontal path
fig1, ax1 = plt.subplots(figsize=(4.5, 3.5))
ax1.plot(pos_e, pos_n, 'b-', linewidth=1.5)
ax1.set_xlabel('East [m]')
ax1.set_ylabel('North [m]')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'lasso_horizontal_path.png'), dpi=300, bbox_inches='tight')
plt.close(fig1)

# Figure 2: Height over time
fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
ax2.plot(times, llh[:, 2], 'b-', linewidth=1.5)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Height [m]')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'lasso_height_over_time.png'), dpi=300, bbox_inches='tight')
plt.close(fig2)

print(f"Generated and saved figures to {output_dir}")
