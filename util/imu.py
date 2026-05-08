
import numpy as np
from numba import njit
import r3f
from util.constants import wei, A_E, E2
from functools import partial

### IMU Utils

# FROM INU Library, moved here so that I could use njit
@njit
def somigliana(llh):
    # From INU Library, moved here so that I could use njit. Modified for (3,) case only.
    """Calculate the local acceleration of gravity vector in the navigation frame
    using the Somigliana equation. The navigation frame here has the North,
    East, Down (NED) orientation.

    Parameters
    ----------
    llh : (3,) np.ndarray
        Geodetic position vector of latitude (radians), longitude (radians), and
        height above ellipsoid (meters) or matrix of such vectors.

    Returns
    -------
    gamma : (3,)  np.ndarray
        Acceleration of gravity in meters per second squared.

    """
    # WGS84 constants (IS-GPS-200M and NIMA TR8350.2)
    A_E = 6378137.0             # Earth's semi-major axis (m) (p. 109)
    E2 = 6.694379990141317e-3   # Earth's eccentricity squared (ND) (derived)

    # gravity coefficients
    ge = 9.7803253359
    k = 1.93185265241e-3
    f = 3.35281066475e-3
    m = 3.44978650684e-3


    # Get local acceleration of gravity for height equal to zero.
    slat2 = np.sin(llh[0])**2
    klat = np.sqrt(1 - E2*slat2)
    grav_z0 = ge*(1 + k*slat2)/klat

    # Calculate gamma for the given height.
    grav_z = grav_z0*(1 + (3/A_E**2)*llh[2]**2
        - 2/A_E*(1 + f + m - 2*f*slat2)*llh[2])

    grav = np.array([0.0, 0.0, grav_z])


    return grav

def grav_harmonic_pe(pe):
    
    # Calculate gravity at ECEF position pe
    kM = 3.986005e14
    J2 = 9.3324e-9
    a = 6378137
    r = np.sqrt(np.dot(pe,pe))
    A = -kM
    B = (-1/2)*J2*(a**2)*kM
    C = 15*J2*(a**2)*kM/2
    return (A/np.power(r,3) + B/np.power(r,5) + C*(pe[2]**2)/np.power(r,7))*pe + wei[2]**2 * np.array([pe[0],pe[1],0])

@njit
def grav_somigliana_pe(pe):
    
    llh = ecef_to_geodetic(pe)
    dcm_en = dcm_ecef_to_navigation(llh[0],llh[1]).T
    return dcm_en @ somigliana(llh)

@njit
def dcm_ecef_to_navigation(
        lat: float,
        lon: float,
        ned: bool = True,
        degs: bool = False,
    ) -> np.ndarray:
    # From r3f Library, moved here so that I could use njit. Modified for K=1 case only.
    """Create the passive rotation matrix from the Earth-centered, Earth-fixed
    (ECEF) frame to the local-level navigation frame.

    Parameters
    ----------
    lat : float 
        Geodetic latitude in radians (or degrees if `degs` is True).
    lon : float
        Geodetic longitude in radians (or degrees if `degs` is True).
    ned : bool, default True
        Flag to use NED (True) or ENU (False) orientation.
    degs : bool, default False
        Flag to interpret angles as degrees.

    Returns
    -------
    C : (3, 3) np.ndarray
        Passive rotation matrix or stack of K such matrices.

    Examples
    --------
    Single position:

        >>> C = r3f.dcm_ecef_to_navigation(np.pi/4, 0)
        >>> C
        array([[-0.70710678, -0.        ,  0.70710678],
               [-0.        ,  1.        ,  0.        ],
               [-0.70710678, -0.        , -0.70710678]])

    Multiple positions:

        >>> lat = np.array([np.pi/6, np.pi/4])
        >>> lon = np.array([0, np.pi/4])
        >>> C = r3f.dcm_ecef_to_navigation(lat, lon)
        >>> C
        array([[[-0.5       , -0.        ,  0.8660254 ],
                [-0.        ,  1.        ,  0.        ],
                [-0.8660254 , -0.        , -0.5       ]],
        <BLANKLINE>
               [[-0.5       , -0.5       ,  0.70710678],
                [-0.70710678,  0.70710678,  0.        ],
                [-0.5       , -0.5       , -0.70710678]]])

    """
    s = np.pi/180 if degs else 1.0

    # Get cosine and sine of latitude and longitude.
    clat = np.cos(s*lat)
    slat = np.sin(s*lat)
    clon = np.cos(s*lon)
    slon = np.sin(s*lon)

    # Get the rotation matrix elements.
    if ned:
        c11 = -slat*clon;   c12 = -slat*slon;   c13 = clat
        c21 = -slon;        c22 = clon;         c23 = 0.0
        c31 = -clat*clon;   c32 = -clat*slon;   c33 = -slat
    else:
        c11 = -slon;        c12 = clon;         c13 = 0.0
        c21 = -slat*clon;   c22 = -slat*slon;   c23 = clat
        c31 = clat*clon;    c32 = clat*slon;    c33 = slat

    # Assemble the rotation matrix.
    C = np.array([
        [c11, c12, c13],
        [c21, c22, c23],
        [c31, c32, c33]])


    return C


@njit
def ecef_to_geodetic(
        pe: np.ndarray,
        degs: bool = False
    ) -> np.ndarray:
    # From r3f Library, moved here so that I could use njit. Modified for (3,) case only.
    """Convert an ECEF (Earth-centered, Earth-fixed) position to geodetic
    coordinates. This follows the WGS-84 definitions (see WGS-84 Reference
    System (DMA report TR 8350.2)).

    Parameters
    ----------
    pe : (3,) list, tuple, or np.ndarray
        Position vector in ECEF coordinates in meters or matrix of such vectors.
    degs : bool, default False
        Flag to convert angles to degrees.

    Returns
    -------
    llh : (3,) np.ndarray
        Vector of geodetic position in terms of latitude in radians (or degrees
        if `degs` is True), longitude in radians (or degrees if `degs` is True),
        and height above ellipsoid in meters, or matrix of such vectors.

    See Also
    --------
    geodetic_to_ecef

    Notes
    -----
    Note that inherent in solving the problem of getting the geodetic latitude
    and ellipsoidal height is finding the roots of a quartic polynomial because
    we are looking for the intersection of a line with an ellipse. While there
    are closed-form solutions to this problem (see Wikipedia), each point has
    potentially four solutions and the solutions are not numerically stable.
    Instead, this function uses the Newton-Raphson method to iteratively solve
    for the geodetic coordinates.

    First, we want to approximate the values for geodetic latitude, `lat`, and
    height above ellipsoid, `hae`, given the pe = [xe, ye, ze] position in the
    ECEF frame:

                                .--------
         ^                     /  2     2            ^
        hae = 0         re = `/ xe  + ye            lat = arctan2(ze, re),

    where `re` is the distance from the z axis of the ECEF frame. (While there
    are better approximations for `hae` than zero, the improvement in accuracy
    was not enough to reduce the number of iterations and the additional
    computational burden could not be justified.)  Then, we will iteratively use
    this approximation for `lat` and `hae` to calculate what `re` and `ze` would
    be, get the residuals given the correct `re` and `ze` values in the ECEF
    frame, use the inverse Jacobian to calculate the corresponding residuals of
    `lat` and `hae`, and update our approximations for `lat` and `hae` with
    those residuals. In testing millions of randomly generated points, three
    iterations was sufficient to reach the limit of numerical precision for
    64-bit floating-point numbers.

    So, first, let us define the transverse, `Rt`, and meridional, `Rm`, radii
    and the cosine and sine of the latitude:

                                                              .---------------
              aE               aE  .-      2 -.              /      2   2  ^
        Rt = ----       Rm = ----- | 1 - eE   |     klat = `/ 1 - eE sin (lat) ,
             klat                3 '-        -'
                             klat
                  ^                               ^
        co = cos(lat)                   si = sin(lat)

    where `eE` is the eccentricity of the Earth, and `aE` is the semi-major
    radius of the Earth. The ECEF-frame `re` and `ze` values given the
    approximations to geodetic latitude and height above ellipsoid are

         ^             ^                 ^              2   ^
        re = co (Rt + hae)              ze = si (Rm klat + hae) .

    We already know the correct values for `re` and `ze`, so we can get
    residuals:

         ~         ^                     ~         ^
        re = re - re                    ze = ze - ze .

    We can relate the `re` and `ze` residuals to the `lat` and `hae` residuals
    by using the inverse Jacobian matrix:

        .-  ~  -.       .-  ~ -.
        |  lat  |    -1 |  re  |
        |       | = J   |      | .
        |   ~   |       |   ~  |
        '- hae -'       '- ze -'

    With a bit of algebra, we can combine and simplify the calculation of the
    Jacobian with the calculation of the `lat` and `hae` residuals:

         ~         ~       ~             ~         ~       ~         ^
        hae = (si ze + co re)           lat = (co ze - si re)/(Rm + hae) .

    Conceptually, this is the backwards rotation of the (`re`, `ze`) residuals
    vector by the angle `lat`, where the resulting y component of the rotated
    vector is treated as an arc length and converted to an angle, `lat`, using
    the radius `Rm` + `hae`. With the residuals for `lat` and `hae`, we can
    update our approximations for `lat` and `hae`:

         ^     ^     ~                   ^     ^     ~
        hae = hae + hae                 lat = lat + lat

    and iterate again. Finally, the longitude, `lon`, is exactly the arctangent
    of the ECEF `xe` and `ye` values:

        lon = arctan2(ye, xe) .

    Examples
    --------
    Single position:

        >>> pe = np.array([6378137.0, 0, 0])
        >>> llh = r3f.ecef_to_geodetic(pe)
        >>> llh
        array([0., 0., 0.])

    Multiple positions:

        >>> pe = np.array([[5528256.63929284, 4518297.98563012],
        ...         [0, 0],
        ...         [3170373.73538364, 4488055.51564711]])
        >>> llh = r3f.ecef_to_geodetic(pe)
        >>> llh
        array([[5.23598776e-01, 7.85398163e-01],
               [0.00000000e+00, 0.00000000e+00],
               [4.70041200e-09, 1.00000000e+03]])

    References
    ----------
    .. [1]  WGS-84 Reference System (DMA report TR 8350.2)
    .. [2]  Inertial Navigation: Theory and Implementation by David Woodburn

    """
    s = np.pi/180 if degs else 1.0

    # Parse input.
    xe, ye, ze = pe

    # Initialize the height above the ellipsoid.
    hhae = 0

    # Get the true radial distance from the z axis.
    re = np.sqrt(xe**2 + ye**2)

    # Initialize the estimated ground latitude.
    hlat = np.arctan2(ze, re) # bound to [-pi/2, pi/2]

    # Iterate to reduce residuals of the estimated closest point on the ellipse.
    for _ in range(3):
        # Using the estimated ground latitude, get the cosine and sine.
        co = np.cos(hlat)
        si = np.sin(hlat)
        klat2 = 1 - E2*si**2
        klat = np.sqrt(klat2)
        Rt = A_E/klat
        Rm = (Rt/klat2)*(1 - E2)

        # Get the estimated position in the meridional plane (the plane defined
        # by the longitude and the z axis).
        hre = co*(Rt + hhae)
        hze = si*(Rm*klat2 + hhae)

        # Get the residuals.
        tre = re - hre
        tze = ze - hze

        # Using the inverse Jacobian, get the residuals in lat and hae.
        tlat = (co*tze - si*tre)/(Rm + hhae)
        thae = si*tze + co*tre

        # Adjust the estimated ground latitude and ellipsoidal height.
        hlat = hlat + tlat
        hhae = hhae + thae

    # Get the longitude.
    lon = np.arctan2(ye, xe)

    # Assemble the matrix.
    llh = np.array([hlat/s, lon/s, hhae])

    return llh

@njit
def gen_coriolis_tile_so3r3(v):
    
    tile = np.empty((4,3,3))
    for i in range(4):
        for j in range(3):
            tile[i,j,:] = v
    return tile

def _calculate_fogm_sigma(dt, sigma_ss, tau):
    """Base function to be wrapped by partial.
    Calculates the driving noise sigma required to maintain a steady-state
    sigma over a time interval dt.
    """
    # Protect against divide by zero or negative time
    if dt <= 0: return 0.0
    return sigma_ss * np.sqrt(1 - np.exp(-2 * dt / tau))

def create_imu_noise_model(imu_dict):
    """Parses IMU params and returns static white noise values and
    dynamic bias noise functions.
    """
    dt_sample = imu_dict['sampling_period']
    
    # --- Constants ---
    G_M_S2 = 9.80665
    DEG_TO_RAD = np.pi / 180.0
    HR_TO_SEC = 3600.0
    SQRT_HR_TO_SQRT_SEC = 60.0
    
    # ==========================================
    # 1. Gyroscope Logic
    # ==========================================
    # White Noise (Static)
    arw_deg_sqrthr = imu_dict['angle_random_walk_deg_sqrthr']
    N_gyro = (arw_deg_sqrthr * DEG_TO_RAD) / SQRT_HR_TO_SQRT_SEC
    gyro_white_sigma = N_gyro / np.sqrt(dt_sample)
    
    # Bias Parameters (for Partial)
    bias_stab_deg_hr = imu_dict['gyro_bias_stability_deg_hr']
    sigma_bg_ss = (bias_stab_deg_hr * DEG_TO_RAD) / HR_TO_SEC 
    tau_g = imu_dict['time_constant_gyro_bias_s']

    # ==========================================
    # 2. Accelerometer Logic
    # ==========================================
    # White Noise (Static)
    vrw_metric = imu_dict['vel_random_walk_m_s_sqrthr']
    N_accel = vrw_metric / SQRT_HR_TO_SQRT_SEC
    accel_white_sigma = N_accel / np.sqrt(dt_sample)
    
    # Bias Parameters (for Partial)
    bias_stab_mg = imu_dict['accelerometer_bias_stability_mg']
    sigma_ba_ss = bias_stab_mg * 1e-3 * G_M_S2
    tau_a = imu_dict['time_constant_accel_bias_s']

    # ==========================================
    # 3. Create Partials
    # ==========================================
    # We freeze 'sigma_ss' and 'tau'. The user only needs to provide 'dt'.
    
    gyro_bias_func = partial(_calculate_fogm_sigma, 
                             sigma_ss=sigma_bg_ss, 
                             tau=tau_g)
    
    accel_bias_func = partial(_calculate_fogm_sigma, 
                              sigma_ss=sigma_ba_ss, 
                              tau=tau_a)
    
    return {
        'gyro_white_noise_sigma': gyro_white_sigma,       # [rad/s]
        'accel_white_noise_sigma': accel_white_sigma,     # [m/s^2]
        'gyro_bias_driving_std_func': gyro_bias_func,     # Callable(dt) -> sigma [rad/s]
        'accel_bias_driving_std_func': accel_bias_func    # Callable(dt) -> sigma [m/s^2]
    }


def filter_imu_data(data, cutoff_hz=10, fs=100):
    
    from scipy.signal import butter, filtfilt
    """
    Applies a zero-phase low-pass filter to IMU columns.
    
    Parameters:
    data: np.array with columns [time, ax, ay, az, gx, gy, gz]
    cutoff_hz: Filter cutoff (try 5-15 Hz for golf cart vibration)
    fs: Sampling rate (100 Hz as specified)
    """
    # Define the filter (2nd order is usually sufficient)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(N=2, Wn=normal_cutoff, btype='low', analog=False)
    
    # Copy to keep the original time column intact
    filtered_data = np.copy(data)
    
    # Apply to accel (cols 1,2,3) and gyro (cols 4,5,6)
    for i in range(1, 7):
        filtered_data[:, i] = filtfilt(b, a, data[:, i])
        
    return filtered_data