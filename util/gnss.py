#%%

import numpy as np
import r3f
from numba import njit
from util.constants import c

def calcEl(satXYZ, rxXYZ):
    """Computes elevation angle in radians between receiver and satellite
    positions in ECEF coordinates using the local NED frame.
    
    Parameters
    ----------
    satXYZ : (3,) np.ndarray  Satellite ECEF [m]
    rxXYZ  : (3,) np.ndarray  Receiver ECEF [m]
    
    Returns
    -------
    el : float  Elevation angle [rad]

    """
    # Vector from receiver to satellite in local NED frame
    sat_ned = r3f.ecef_to_tangent(satXYZ, rxXYZ, ned=True)
    
    # Range (3D)
    dist = np.linalg.norm(sat_ned)
    
    # Elevation = asin(-Down / Range)
    # sat_ned[2] is Down (positive towards Earth center)
    el = np.arcsin(-sat_ned[2] / dist)
    
    return el

@njit
def elDepWeight(el, measStdev):
    """Transpiled from 'Enabling-Robust-State-Estimation-through-Measurement-Error-Covariance-Adaptation."""
    # inputs ::
    # el --> elevation angle [rad]
    # measStdev ---> initial noise applied to observable [meters]
    # output ::
    # r --> elevation angle dep. GNSS obs. weight (std dev) [meters]

    r = (1/np.sin(el))*measStdev
    r[r>10*measStdev] = measStdev*10
    return r

@njit
def elDepWeightScalar(el, measStdev):
    """Transpiled from 'Enabling-Robust-State-Estimation-through-Measurement-Error-Covariance-Adaptation."""
    # inputs ::
    # el --> elevation angle [rad]
    # measStdev ---> initial noise applied to observable [meters]
    # output ::
    # r --> elevation angle dep. GNSS obs. weight (std dev) [meters]

    r = (1/np.sin(el))*measStdev
    if r > 10*measStdev:
        r = measStdev*10
    return r

@njit
def tropMap(El):
    """Transpiled from 'Enabling-Robust-State-Estimation-through-Measurement-Error-Covariance-Adaptation."""
    return 1.001/np.sqrt(0.002001 + np.power(np.sin(El),2))

def obsMap(p1: np.ndarray, p2: np.ndarray, trop: bool):
    """Transpiled from 'Enabling-Robust-State-Estimation-through-Measurement-Error-Covariance-Adaptation
    /*
        inputs ::
        p1 --> ECEF xyz coordinates of satellite [meter]
        p2 --> ECEF xyz coordinates of receiver [meter]
        trop --> Troposphere modeling switch
        outputs ::
        h --> measurement mapping
    */.    
    """
    r = np.linalg.norm(p1-p2)
    if trop:
        el = calcEl(p1,p2)
        mapT = tropMap(el)
        h = np.hstack([(p1-p2)/r,np.array([1.0,mapT])])
    else:
        h = np.hstack([(p1-p2)/r,np.array([1.0,0.0])])
    return h

def calc_H_values(satXYZ:np.ndarray, rxXYZ:np.ndarray) -> tuple[np.ndarray, float]:
    
    r = np.linalg.norm(satXYZ - rxXYZ)
    return (rxXYZ-satXYZ) / r, 1.0

def clock_drift_variance(clock_noise_model: dict, delta_t: float) -> float:
    """Calculate the variance of clock drift over a time interval.
    
    Parameters
    ----------
    clock_noise_model : dict
        Dictionary containing:
        - 'phase_sigma': clock phase noise standard deviation [s]
        - 'freq_sigma': clock frequency noise standard deviation [s/s]
    delta_t : float
        Time interval [s]
    
    Returns
    -------
    float
        Variance of clock drift in meters^2 over the time interval
    
    Notes
    -----
    For a random walk model, the variance of clock bias change is:
    variance = (freq_sigma * sqrt(delta_t) * c)^2
    where c is the speed of light in m/s.

    """
    freq_sigma = clock_noise_model['freq_sigma']  # [s/s]
    # Standard deviation of change in clock bias over delta_t
    # sigma_delta_b = freq_sigma * sqrt(delta_t) * c
    variance = (freq_sigma * np.sqrt(delta_t) * c)**2
    return variance

def clock_transition_variance(clock_noise_model: dict, delta_t: float) -> float:
    """Compute discrete-time process noise covariance Q for a
    clock bias / clock drift model.

    State: x = [clock_bias (s), clock_drift (s/s)]

    Continuous-time model:
        dot(b) = d + w_b
        dot(d) = w_d

    where:
        w_b ~ N(0, phase_sigma^2)
        w_d ~ N(0, freq_sigma^2)

    Parameters
    ----------
    clock_noise_model : dict
        Dictionary with keys:
            'phase_sigma' : float  (seconds)
            'freq_sigma'  : float  (seconds/second)
    delta_t : float
        Time step in seconds

    Returns
    -------
    Q : (2, 2) ndarray
        Discrete-time process noise covariance is meters^2

    """
    phase_sigma = clock_noise_model['phase_sigma']
    freq_sigma  = clock_noise_model['freq_sigma']

    q_b = phase_sigma ** 2
    q_d = freq_sigma ** 2
    dt  = delta_t

    Q = np.array([
        [q_b * dt + (1.0 / 3.0) * q_d * dt**3,
         (1.0 / 2.0) * q_d * dt**2],
        [(1.0 / 2.0) * q_d * dt**2,
         q_d * dt]
    ])

    return Q*(c**2)  # Convert to meters^2

def estimate_clock_noise_model(times: np.ndarray, clock_biases: np.ndarray) -> dict:
    """Estimate clock noise model parameters from observed clock bias data.
    
    Uses variance analysis of the clock bias time series to estimate the
    continuous-time noise parameters for the clock model:
        dot(b) = d + w_b
        dot(d) = w_d
    
    where w_b ~ N(0, phase_sigma^2) and w_d ~ N(0, freq_sigma^2).
    
    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Time stamps in seconds
    clock_biases : np.ndarray, shape (N,)
        Clock bias measurements in meters
    
    Returns
    -------
    dict
        Dictionary with estimated parameters:
            'phase_sigma' : float (seconds)
            'freq_sigma'  : float (seconds/second)
    
    Notes
    -----
    The estimation uses:
    - Second-order differences to estimate freq_sigma (drift variation)
    - First-order residuals to estimate phase_sigma (bias noise)
    
    Assumes uniform or nearly-uniform time spacing for best results.

    """
    # Convert biases from meters to seconds
    clock_biases_s = clock_biases / c
    
    # Compute time differences
    dt = np.diff(times)
    mean_dt = np.mean(dt)
    
    # First differences (estimate of drift * dt + noise)
    first_diff = np.diff(clock_biases_s)
    
    # Estimate drift at each point
    drift_estimates = first_diff / dt
    
    # Second differences of drift (change in drift)
    drift_diff = np.diff(drift_estimates)
    dt_drift = dt[:-1] + dt[1:]  # Time between drift estimates
    
    # Variance of drift changes
    # For continuous white noise w_d, Var(d[k+1] - d[k]) ≈ freq_sigma^2 * dt
    var_drift_change = np.var(drift_diff)
    mean_dt_drift = np.mean(dt_drift) / 2  # Effective dt for drift
    
    # Estimate freq_sigma
    freq_sigma = np.sqrt(var_drift_change / mean_dt_drift)
    
    # Remove drift trend from first differences to isolate phase noise
    # Expected variance from phase noise: phase_sigma^2 * dt
    # Expected variance from freq noise: (1/3) * freq_sigma^2 * dt^3
    var_first_diff = np.var(first_diff)
    expected_freq_contribution = (1.0 / 3.0) * (freq_sigma ** 2) * (mean_dt ** 3)
    
    # Isolate phase noise contribution
    phase_variance_contribution = var_first_diff - expected_freq_contribution
    
    if phase_variance_contribution > 0:
        phase_sigma = np.sqrt(phase_variance_contribution / mean_dt)
    else:
        # If negative (can happen with noisy data), use simpler estimate
        # Detrend the first differences
        mean_drift = np.mean(drift_estimates)
        detrended = first_diff - mean_drift * dt
        phase_sigma = np.sqrt(np.var(detrended) / mean_dt)
    
    return {
        'phase_sigma': phase_sigma,
        'freq_sigma': freq_sigma
    }