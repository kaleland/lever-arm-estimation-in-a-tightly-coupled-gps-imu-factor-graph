
import numpy as np

DEG_TO_RADIANS = np.pi/180
RADIANS_TO_DEG = 180/np.pi

wei = np.array([0,0,7.2921151467e-05]) # Rotation rate of ECEF frame relative to
# inertial frame, rad/s

I3 = np.eye(3) # 3x3 identity matrix
I4 = np.eye(4) # 4x4 identity matrix

# WGS84 constants (IS-GPS-200M and NIMA TR8350.2)
A_E = 6378137.0             # Earth's semi-major axis (m) (p. 109)
E2 = 6.694379990141317e-3   # Earth's eccentricity squared (ND) (derived)

c = 299792458.0 # Speed of light in vacuum, m/s
