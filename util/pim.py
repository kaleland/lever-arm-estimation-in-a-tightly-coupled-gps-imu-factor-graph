#%%

from dataclasses import dataclass
from functools import partial

from util import se23, se3, so3
from util.general import weight_from_covariance
from util.imu import grav_somigliana_pe, filter_imu_data

import r3f
import numpy as np
from matplotlib import pyplot as plt
from util.constants import DEG_TO_RADIANS,RADIANS_TO_DEG, wei
from scipy import linalg as spla
from numba import njit
from typing import Callable, Any, List


#%%
# Load some noiseless IMU data with true pva
def load_noiseless_data() -> tuple[np.ndarray,np.ndarray]:
    """Load a preset noiseless IMU dataset with true pva values.
    
    Output:
    - imu_data: np.ndarray of shape (n, 7) where n is the number of samples.
        The columns are [t, ax, ay, az, wx, wy, wz].
    - pva_data: np.ndarray of shape (n, 10) where n is the number of samples.
        The columns are [t, p, v, a].
    - eTw: np.ndarray of shape (4, 4).
        The transformation matrix from the ECEF frame to the local world frame.
    """
    file_path = './data/medium_figure_8_path/noiseless_imu_data_from_path.npz'
    # file_path = './data/gentle_random_path/noiseless_imu_data_from_path.npz'
    data = np.load(file_path)
    imu_data = data['imu_data']
    pva_data = data['true_pva_data']
    eTw = data['Tnom']
    imu_data[:,0] = imu_data[:,0] + np.median(np.diff(imu_data[:,0]))/2    # account for midpoint offset
    return imu_data, pva_data, eTw

# def remove_rotation_of_earth(imu_data: np.ndarray, pva:np.ndarray, eTw:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
#     '''
#     Remove Coriolis acceleration from the accelerometer measurements and the
#     rotation of the Earth from the gyroscope measurements
#     '''
#%%


# Calculate preintegrated measurements for delta_R, delta_v, and delta_p
def preintegrate_imu(
        pim_times: np.ndarray, 
        imu_data: np.ndarray, 
        accel_biases: np.ndarray,
        gyro_biases: np.ndarray
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Preintegrate the IMU data to get delta_R, delta_v, and delta_p.

    Inputs:
    - pim_times: np.ndarray of shape (n+1,) where n is the number of preintegration intervals.
        The times at which the preintegration intervals start. These times must
        be IMU measurement times.
    - imu_data: np.ndarray of shape (n, 7) where n is the number of samples.
        The columns are [t, ax, ay, az, wx, wy, wz].
    - accel_biases: np.ndarray of shape (n, 3) where n is the number of intervals.
        The columns are [bx, by, bz].
    - gyro_biases: np.ndarray of shape (n, 3) where n is the number of intervals.
        The columns are [bx, by, bz].
    - gravities: np.ndarray of shape (n, 3) where n is the number of intervals.
        The columns are [gx, gy, gz].

    Outputs:
    - delta_R: np.ndarray of shape (n, 3, 3) where n is the number of intervals.
        The rotation matrices.
    - delta_v: np.ndarray of shape (n, 3) where n is the number of intervals.
        The velocity differences.
    - delta_p: np.ndarray of shape (n, 3) where n is the number of intervals.
        The position differences.
    """
    msmt_times = imu_data[:, 0]
    accel_msmts = imu_data[:, 1:4]
    gyro_msmts = imu_data[:, 4:7]
    n_pim_intervals = len(pim_times) - 1
    pim_deltaR = np.zeros((n_pim_intervals, 3, 3))
    pim_deltav = np.zeros((n_pim_intervals, 3))
    pim_deltap = np.zeros((n_pim_intervals, 3))

    for pim_interval in range(n_pim_intervals):
        i_idx = np.searchsorted(msmt_times, pim_times[pim_interval])
        j_idx = np.searchsorted(msmt_times, pim_times[pim_interval+1])

        delta_R_ik = np.eye(3)
        delta_v_ik = np.zeros(3)
        delta_p_ik = np.zeros(3)

        a=0
        for k in range(i_idx+1, j_idx+1):
            a+=1
            deltaT = msmt_times[k] - msmt_times[k-1]
            Rk = delta_R_ik
            vk = delta_v_ik
            delta_p_ik = delta_p_ik + vk*deltaT + 0.5*Rk@(accel_msmts[k] - accel_biases[pim_interval])*(deltaT**2) 
            delta_v_ik = vk + Rk@(accel_msmts[k] - accel_biases[pim_interval])*deltaT 
            delta_R_ik = Rk @ so3.expmap((gyro_msmts[k] - gyro_biases[pim_interval]) * deltaT)


        # Save the final changes over the interval
        deltaR_ij = delta_R_ik
        deltav_ij = delta_v_ik
        deltap_ij = delta_p_ik

        pim_deltaR[pim_interval] = deltaR_ij
        pim_deltav[pim_interval] = deltav_ij
        pim_deltap[pim_interval] = deltap_ij

    return pim_deltaR, pim_deltav, pim_deltap




#%%
# Compare the preintegrated measurements to the true pva
def calc_true_deltas(pim_times: np.ndarray, true_pva: np.ndarray, eTw: np.ndarray, remove_rotation_of_earth:bool = False) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Calculate the true delta_R, delta_v, and delta_p given the true PVA values.

    Inputs:
    - pim_times: np.ndarray of shape (n,) where n is the number of preintegration intervals.
        The times at which the preintegration intervals start. These times must
        be IMU measurement times.
    - true_pva: np.ndarray of shape (n, 10) where n is the number of samples.
        The columns are [t, p, v, a].
    - eTw: np.ndarray of shape (4, 4).
        The transformation matrix from the ECEF frame to the local world frame.

    Outputs:
    - deltaRijs: np.ndarray of shape (n, 3, 3) where n is the number of samples.
        The rotation matrices.
    - deltavijs: np.ndarray of shape (n, 3) where n is the number of samples.
        The velocity differences.
    - deltapijs: np.ndarray of shape (n, 3) where n is the number of samples.
        The position differences.
    """
    rounded_pim_times = np.round(pim_times, 2)
    rounded_pva_times = np.round(true_pva[:, 0], 2)
    true_pva_pim_times = true_pva[np.isin(rounded_pva_times, rounded_pim_times)]
    times = true_pva_pim_times[:, 0]
    ps = true_pva_pim_times[:, 1:4]
    vs = true_pva_pim_times[:, 4:7]
    rpys = true_pva_pim_times[:, 7:10]
    Rs = np.array([r3f.rpy_to_dcm(rpy*DEG_TO_RADIANS).T for rpy in rpys])
    pes = np.array([eTw[:3,:3]@p+eTw[:3,3] for p in ps])
    grav_es = np.array([grav_somigliana_pe(pe) for pe in pes])
    gravs = np.array([eTw[:3, :3].T @ grav_e for grav_e in grav_es])
    if not remove_rotation_of_earth:
        deltaRijs = np.array([ Rs[i].T @ Rs[i+1] for i in range(len(Rs)-1)])
        deltavijs = np.array([Rs[i].T @ (vs[i+1] - vs[i] - gravs[i]*(times[i+1] - times[i])) for i in range(len(Rs)-1)])
        # deltapijs = np.array([Rs[i].T @ (ps[i+1] - ps[i] - vs[i]*(times[i+1] - times[i]) - 0.5*gravs[i]*((times[i+1] - times[i]+np.median(np.diff(true_pva[:,0])))**2)) for i in range(len(Rs)-1)])
        deltapijs = np.array([Rs[i].T @ (ps[i+1] - ps[i] - vs[i]*(times[i+1] - times[i]) - 0.5*gravs[i]*((times[i+1] - times[i])**2)) for i in range(len(Rs)-1)])

    else:
        wwi = eTw[:3,:3].T@wei
        deltaRijs = np.array([ Rs[i].T @ Rs[i+1] @ so3.expmap(Rs[i].T@wwi*(times[i+1] - times[i])) for i in range(len(Rs)-1)])
        deltavijs = np.array([Rs[i].T @ (vs[i+1] - vs[i] + np.cross(2*wwi,vs[i])*(times[i+1]-times[i]) - gravs[i]*(times[i+1] - times[i])) for i in range(len(Rs)-1)])
        # deltapijs = np.array([Rs[i].T @ (ps[i+1] - ps[i] - vs[i]*(times[i+1] - times[i]) - 0.5*np.cross(2*wwi,vs[i])*(times[i+1]-times[i])**2 - 0.5*gravs[i]*(times[i+1] - times[i]+np.median(np.diff(true_pva[:,0])))**2) for i in range(len(Rs)-1)]) # The +np.median(np.diff(true_pva[:,0])) term removes one samples worth of gravity displacement. Why is this necessary?
        deltapijs = np.array([Rs[i].T @ (ps[i+1] - ps[i] - vs[i]*(times[i+1] - times[i]) + 0.5*np.cross(2*wwi,vs[i])*(times[i+1]-times[i])**2 - 0.5*gravs[i]*(times[i+1] - times[i])**2) for i in range(len(Rs)-1)]) 
    return deltaRijs, deltavijs, deltapijs


#%%
def test_so3r3(
        remove_rotation_of_earth:bool = False
):
    
    # Load the noiseless data
    imu_data, pva_data, eTw = load_noiseless_data()

    # # Plot the true pva data
    # plt.figure()
    # plt.plot(pva_data[:, 0], pva_data[:, 1], label='p_x')
    # plt.plot(pva_data[:, 0], pva_data[:, 2], label='p_y')
    # plt.plot(pva_data[:, 0], pva_data[:, 3], label='p_z')
    # plt.legend()
    # plt.title('True Position')
    # plt.ylabel('m')
    # plt.grid()

    # plt.figure()
    # plt.plot(pva_data[:, 0], pva_data[:, 4], label='v_x')
    # plt.plot(pva_data[:, 0], pva_data[:, 5], label='v_y')
    # plt.plot(pva_data[:, 0], pva_data[:, 6], label='v_z')
    # plt.legend()
    # plt.title('True Velocity')
    # plt.ylabel('m/s')
    # plt.grid()

    # plt.figure()
    # plt.plot(pva_data[:, 0], pva_data[:, 7], label='roll')
    # plt.plot(pva_data[:, 0], pva_data[:, 8], label='pitch')
    # plt.plot(pva_data[:, 0], pva_data[:, 9], label='yaw')
    # plt.legend()
    # plt.title('True Atttitude')
    # plt.ylabel('degrees')
    # plt.grid()

    # Preintegrate the IMU data
    pim_times = imu_data[:, 0][::100]
    accel_biases = np.zeros((len(imu_data), 3))
    gyro_biases = np.zeros((len(imu_data), 3))
    pim_deltaR, pim_deltav, pim_deltap = preintegrate_imu(pim_times, imu_data, accel_biases, gyro_biases)

    # Calculate the true delta_R, delta_v, and delta_p
    deltaRijs, deltavijs, deltapijs = calc_true_deltas(pim_times, pva_data, eTw, remove_rotation_of_earth=remove_rotation_of_earth)

    # Compare the preintegrated measurements to the true pva
    delta_R_diffs = np.array([so3.logmap(pim_deltaR[i].T @ deltaRijs[i]) for i in range(len(pim_deltaR))])
    delta_v_diffs = pim_deltav - deltavijs
    delta_p_diffs = pim_deltap - deltapijs

    # Plot the differences over time
    plt.figure()
    plt.plot(pim_times[1:], delta_R_diffs[:, 0], label=r'$\Delta$R diffs [x]')
    plt.plot(pim_times[1:], delta_R_diffs[:, 1], label=r'$\Delta$R diffs [y]')
    plt.plot(pim_times[1:], delta_R_diffs[:, 2], label=r'$\Delta$R diffs [z]')
    plt.legend()
    plt.title(r'$\Delta$R diffs with $SO(3)\times\mathbb{R}^3$')
    plt.grid()

    plt.figure()
    plt.plot(pim_times[1:], delta_v_diffs[:, 0], label=r'$\Delta$v diffs [x]')
    plt.plot(pim_times[1:], delta_v_diffs[:, 1], label=r'$\Delta$v diffs [y]')
    plt.plot(pim_times[1:], delta_v_diffs[:, 2], label=r'$\Delta$v diffs [z]')
    plt.legend()
    plt.title(r'$\Delta$v diffs with $SO(3)\times\mathbb{R}^3$')
    plt.grid()

    plt.figure()
    plt.plot(pim_times[1:], delta_p_diffs[:, 0], label=r'$\Delta$p diffs [x]')
    plt.plot(pim_times[1:], delta_p_diffs[:, 1], label=r'$\Delta$p diffs [y]')
    plt.plot(pim_times[1:], delta_p_diffs[:, 2], label=r'$\Delta$p diffs [z]')
    plt.legend()
    plt.title(r'$\Delta$p diffs with $SO(3)\times\mathbb{R}^3$')
    plt.grid()

    # Plot the true deltas and the preintegrated deltas over time
    pim_logmapped = np.array([so3.logmap(pim_deltaR[i]) for i in range(len(pim_deltaR))])
    delta_R_logmapped = np.array([so3.logmap(deltaRijs[i]) for i in range(len(deltaRijs))])
    plt.figure()
    plt.plot(pim_times[1:], pim_logmapped[:, 0], label=r'$\Delta$R PIM [x]')
    plt.plot(pim_times[1:], pim_logmapped[:, 1], label=r'$\Delta$R PIM [y]')
    plt.plot(pim_times[1:], pim_logmapped[:, 2], label=r'$\Delta$R PIM [z]')
    plt.plot(pim_times[1:], delta_R_logmapped[:, 0], label=r'$\Delta$R true [x]',linestyle='--')
    plt.plot(pim_times[1:], delta_R_logmapped[:, 1], label=r'$\Delta$R true [y]',linestyle='--')
    plt.plot(pim_times[1:], delta_R_logmapped[:, 2], label=r'$\Delta$R true [z]',linestyle='--')
    plt.legend()
    plt.title(r'$\Delta$R with $SO(3)\times\mathbb{R}^3$')
    plt.grid()

    plt.figure()
    plt.plot(pim_times[1:], pim_deltav[:, 0], label=r'$\Delta$v PIM [x]')
    plt.plot(pim_times[1:], pim_deltav[:, 1], label=r'$\Delta$v PIM [y]')
    plt.plot(pim_times[1:], pim_deltav[:, 2], label=r'$\Delta$v PIM [z]')
    plt.plot(pim_times[1:], deltavijs[:, 0], label=r'$\Delta$v true [x]',linestyle='--')
    plt.plot(pim_times[1:], deltavijs[:, 1], label=r'$\Delta$v true [y]',linestyle='--')
    plt.plot(pim_times[1:], deltavijs[:, 2], label=r'$\Delta$v true [z]',linestyle='--')
    plt.legend()
    plt.title(r'$\Delta$v with $SO(3)\times\mathbb{R}^3$')
    plt.grid()

    plt.figure()
    plt.plot(pim_times[1:], pim_deltap[:, 0], label=r'$\Delta$p PIM [x]')
    plt.plot(pim_times[1:], pim_deltap[:, 1], label=r'$\Delta$p PIM [y]')
    plt.plot(pim_times[1:], pim_deltap[:, 2], label=r'$\Delta$p PIM [z]')
    plt.plot(pim_times[1:], deltapijs[:, 0], label=r'$\Delta$p true [x]',linestyle='--')
    plt.plot(pim_times[1:], deltapijs[:, 1], label=r'$\Delta$p true [y]',linestyle='--')
    plt.plot(pim_times[1:], deltapijs[:, 2], label=r'$\Delta$p true [z]',linestyle='--')
    plt.legend()
    plt.title(r'$\Delta$p with $SO(3)\times\mathbb{R}^3$')
    plt.grid()




def _Upsilon_se23_pim(dR,dv,dp):
    """Equation 28 from Associating Uncertainty to Extended Poses."""
    if dR.ndim == 2:
        return np.vstack((np.hstack((dR, dv[:,np.newaxis],dp[:,np.newaxis])),np.array([[0,0,0,1,0]]),np.array([[0,0,0,0,1]])))
    upsilon = np.zeros((dR.shape[0],5,5))
    upsilon[:,:3,:3] = dR
    upsilon[:,:3,3] = dv
    upsilon[:,:3,4] = dp
    upsilon[:,3,3] = 1
    upsilon[:,4,4] = 1
    return upsilon

def _Phi_se23_pim(R,v,p,delta_t):
    """Equation 26 from Associating Uncertainty to Extended Poses."""
    if R.ndim == 2:
        return np.vstack((np.hstack((R, v[:,np.newaxis],p[:,np.newaxis]+delta_t*v[:,np.newaxis])),np.array([[0,0,0,1,0]]),np.array([[0,0,0,0,1]])))
    phi = np.zeros((R.shape[0],5,5))
    phi[:,:3,:3] = R
    phi[:,:3,3] = v
    phi[:,:3,4] = p+delta_t[:,np.newaxis]*v
    phi[:,3,3] = 1
    phi[:,4,4] = 1
    return phi

def _Gamma_se23_pim(delta_t,g):
    """Equation 27 from Associating Uncertainty to Extended Poses."""
    if g.ndim == 1:
        return np.vstack((np.hstack((np.eye(3),delta_t*g[:,np.newaxis]),(1/2)*delta_t**2*g[:,np.newaxis]),np.array([[0,0,0,1,0]]),np.array([[0,0,0,0,1]])))
    gamma = np.zeros((g.shape[0],5,5))
    gamma[:,:3,:3] = np.eye(3)
    gamma[:,:3,3] = delta_t[:,np.newaxis]*g
    gamma[:,:3,4] = (1/2)*delta_t[:,np.newaxis]**2*g
    gamma[:,3,3] = 1
    gamma[:,4,4] = 1
    return gamma

def _Gamma_R_se23_pim(delta_t, omega):
    
    if np.isscalar(delta_t):
        return so3.expmap(-delta_t*omega)
    return np.array([so3.expmap(-delta_t_i*omega) for delta_t_i in delta_t])

def _Gamma_v_se23_pim(delta_t,omega,g):
    
    if np.isscalar(delta_t):
        return so3.left_jacobian(-delta_t*omega) @ (g*delta_t)

    if g.ndim == 1:
        return np.array([so3.left_jacobian(-delta_t_i*omega) @ (g*delta_t_i) for delta_t_i in delta_t])
    
    return np.array([so3.left_jacobian(-delta_t_i*omega) @ (g_i*delta_t_i) for delta_t_i, g_i in zip(delta_t, g)])
    
def _Gamma_p_se23_pim(delta_t,omega,g):
    
    phi = np.linalg.norm(omega)
    phi_x_delta_t = phi*delta_t
    a = (phi**-3)*(phi_x_delta_t*np.cos(phi_x_delta_t) - np.sin(phi_x_delta_t))
    b = (phi**-4)*(0.5*(phi_x_delta_t)**2 - np.cos(phi_x_delta_t) - phi_x_delta_t*np.sin(phi_x_delta_t)+1)

    if np.isscalar(delta_t):
        return (0.5*so3.I3*delta_t**2 + a*so3.skew(omega) + b*so3.skew_squared(omega)) @ g
    
    Omega = so3.skew(omega)
    Omega_2 = so3.skew_squared(omega)

    if g.ndim == 1:
        return np.array([(0.5*so3.I3*delta_t_i**2 + a_i*Omega + b_i*Omega_2) @ g for delta_t_i, a_i, b_i in zip(delta_t, a, b)])
    
    return np.array([(0.5*so3.I3*delta_t_i**2 + a_i*Omega + b_i*Omega_2) @ g_i for delta_t_i, a_i, b_i, g_i in zip(delta_t, a, b, g)])

def _F_se23_pim(delta_t, accels):
    """Equation 34 from Associating Uncertainty to Extended Poses."""
    if np.isscalar(delta_t):
        F = np.eye(9)
        F[6:,3:6] = delta_t*np.eye(3)
        # Velocity coupling from Rotation error (skew(a) * dt)
        # Note: Brossard Eq 34 implies Ad_{-u dt}.
        # -u dt = [-omega dt, -accel dt].
        # Adjoint block (2,1) is skew(v)R.
        # Here v is -accel * dt.
        # So we expect skew(-accel * dt).
        F[3:6,:3] = so3.skew(-accels*delta_t)
        
        # Position coupling from Rotation error?
        # Adjoint block (3,1) is skew(p)R.
        # p is -0.5 * accel * dt^2.
        F[6:,:3] = so3.skew(-0.5*accels*delta_t**2)
        
        return F

    F = np.zeros((delta_t.shape[0],9,9))
    F[:] = np.eye(9)
    F[:,6:,3:6] = delta_t[:,None,None]*np.eye(3)
    
    # Vectorized skew symmetric
    # accels (N,3), delta_t (N,)
    v_inc = -accels * delta_t[:,None]
    p_inc = -0.5 * accels * delta_t[:,None]**2
    
    # We need to construct skew matrices for each
    for i in range(len(delta_t)):
        F[i, 3:6, :3] = so3.skew(v_inc[i])
        F[i, 6:, :3] = so3.skew(p_inc[i])
        
    return F


def _G_se23_pim(delta_t, omega):
    """Equation 39 from Associating Uncertainty to Extended Poses
    Inputs:
    - delta_t: time difference(s)
    - omega: gyroscope angular rate measurement(s).
    """
    omega_x_delta_t = omega*delta_t[:,np.newaxis]
    if np.isscalar(delta_t):
        G = np.zeros((9,6))
        G[:3,:3] = -so3.inverse_left_jacobian(omega*delta_t)*delta_t
        G[3:6,3:] = -so3.expmap(-omega_x_delta_t)*delta_t
        G[6:,3:] = -0.5*so3.expmap(-omega_x_delta_t)*(delta_t**2)
        return G
    G = np.zeros((omega.shape[0],9,6))
    G[:,:3,:3] = -np.array([so3.inverse_left_jacobian(omega_x_delta_t_i)*delta_t_i for omega_x_delta_t_i, delta_t_i in zip(omega_x_delta_t, delta_t)])
    G[:,3:6,3:] = -np.array([so3.expmap(-omega_x_delta_t_i)*delta_t_i for omega_x_delta_t_i, delta_t_i in zip(omega_x_delta_t, delta_t)])
    G[:,6:,3:] = -0.5*np.array([so3.expmap(-omega_x_delta_t_i)*delta_t_i**2 for omega_x_delta_t_i, delta_t_i in zip(omega_x_delta_t, delta_t)])

    return G


def _Q_se23_pim(G, gyro_white_noise, accel_white_noise):
    """Equation 40 from Associating Uncertainty to Extended Poses
    Inputs:
    - G: np.ndarray of shape (9,6) or (n,9,6) where n is the number of preintegration intervals.
    - gyro_white_noise: np.ndarray of shape (3,3)
    - accel_white_noise: np.ndarray of shape (3,3).
    """
    cov = np.zeros((6, 6))
    cov[:3, :3] = gyro_white_noise
    cov[3:, 3:] = accel_white_noise

    if G.ndim == 2:
        return G @ cov @ G.T
    
    return np.array([G_i @ cov @ G_i.T for G_i in G])

def _Sigma_ij_se23_pim(delta_ts, omegas, accels, gyro_white_noise, accel_white_noise, inv_upsilons):
    """Equation 66 from Associating Uncertainty to Extended Poses
    Inputs:
    - delta_ts: np.ndarray of shape (n,) where n is the number of preintegration intervals.
    - omegas: gyroscope measurements, np.ndarray of shape (n,3) 
    - accels: accelerometer measurements, np.ndarray of shape (n,3)
    - gyro_white_noise: np.ndarray of shape (3,3)
    - accel_white_noise: np.ndarray of shape (3,3)
    - inv_upsilons: np.ndarray of shape (n,5,5) where n is the number of preintegration intervals.

    Outputs:
    - Sigma_ij: np.ndarray of shape (9,9)
    """
    n = len(delta_ts)
    Gs = _G_se23_pim(delta_ts, omegas)
    Qs = _Q_se23_pim(Gs, gyro_white_noise, accel_white_noise)
    Fs = _F_se23_pim(delta_ts, accels)
    adjInvUpsilons = se23.adjoint_se23(inv_upsilons)
    Sigma = Qs[0].copy()
    # Debug print
    # print(f"DEBUG_PIM: n={n}, Q[0,0]={Qs[0,0,0]:.2e}, Q_accel[0]={Qs[0,3,3]:.2e}")
    for i in range(1,n):
        Sigma = adjInvUpsilons[i-1] @ Fs[i-1] @ Sigma @ (adjInvUpsilons[i-1] @ Fs[i-1]).T + Qs[i]

    # Debug print
    # print(f"DEBUG_PIM: Final Sigma[0,0]={Sigma[0,0]:.2e}")
    return Sigma

def update_Upsilon_with_bias_one_step(upsilon: np.ndarray, bias_increment: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Equation 68 from Associating Uncertainty to Extended Poses."""
    return upsilon @ so3.expmap(G@ bias_increment) 

def calc_As_se23_pim(adjInvUpsilons, Fs):
    """Sub-equation from (65) in Associating Uncertainty to Extended Poses."""
    As = np.zeros((len(adjInvUpsilons), 9, 9))
    A = np.eye(5)
    for i in range(len(adjInvUpsilons)):
        As[-i] = adjInvUpsilons[-i] @ Fs[-i]  @ A

    return As

def update_Upsilon_with_bias(upsilon: np.ndarray, As: np.ndarray, Gs, bias_increment) -> np.ndarray:
    """Equation 69 from Associating Uncertainty to Extended Poses."""
    n = len(As)
    return upsilon @ so3.expmap(np.sum([As[i] @ Gs[i] @ bias_increment for i in range(n)], axis=0))

def calc_bias_update_jacobian(upsilons,Fs,Gs,deltaR):
    """Equation 70 from Associating Uncertainty to Extended Poses."""
    n = len(upsilons)
    adjUpsilons = se23.adjoint_se23(upsilons)
    d_ups_d_bias = np.zeros((9,6))
    d_ups_d_bias = Gs[0]
    for i in range(1,n):
        d_ups_d_bias = adjUpsilons[i-1]@Fs[i-1]@d_ups_d_bias + Gs[i-1]

    d_ups_d_bias[3:,:3] = d_ups_d_bias[3:,:3]*-1 # Why negative 1? No idea...
    d_ups_d_bias[:3,:3] = d_ups_d_bias[:3,:3] @ deltaR.T # Why this? Somewhere along the way we end up with a Left Jacobian instead of a Right.

    return d_ups_d_bias

def _Upsilon_from_measurements(delta_ts,omegas,accels):
    
    upsilons = np.zeros((len(delta_ts), 5, 5))
    for i in range(len(delta_ts)):
        upsilons[i] = _Upsilon_se23_pim(
            so3.expmap(delta_ts[i]*omegas[i]),
            delta_ts[i]*accels[i],
            (1/2)*delta_ts[i]**2*accels[i]
        )
    return upsilons
def calc_true_deltas_se23(pim_times: np.ndarray, true_pva: np.ndarray, eTw: np.ndarray, remove_rotation_of_earth:bool = False) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Calculate the true delta_R, delta_v, and delta_p given the true PVA values.

    Inputs:
    - pim_times: np.ndarray of shape (n,) where n is the number of preintegration intervals.
        The times at which the preintegration intervals start. These times must
        be IMU measurement times.
    - true_pva: np.ndarray of shape (n, 10) where n is the number of samples.
        The columns are [t, p, v, a].
    - eTw: np.ndarray of shape (4, 4).
        The transformation matrix from the ECEF frame to the local world frame.

    Outputs:
    - deltaRijs: np.ndarray of shape (n, 3, 3) where n is the number of samples.
        The rotation matrices.
    - deltavijs: np.ndarray of shape (n, 3) where n is the number of samples.
        The velocity differences.
    - deltapijs: np.ndarray of shape (n, 3) where n is the number of samples.
        The position differences.
    """
    rounded_pim_times = np.round(pim_times, 2)
    rounded_pva_times = np.round(true_pva[:, 0], 2)
    true_pva_pim_times = true_pva[np.isin(rounded_pva_times, rounded_pim_times)]
    times = true_pva_pim_times[:, 0]
    ps = true_pva_pim_times[:, 1:4]
    vs = true_pva_pim_times[:, 4:7]
    rpys = true_pva_pim_times[:, 7:10]
    Rs = np.array([r3f.rpy_to_dcm(rpy*DEG_TO_RADIANS).T for rpy in rpys])
    pes = np.array([eTw[:3,:3]@p+eTw[:3,3] for p in ps])
    grav_es = np.array([grav_somigliana_pe(pe) for pe in pes])
    gravs = np.array([eTw[:3, :3].T @ grav_e for grav_e in grav_es])

    Ts = np.zeros((len(Rs),5,5))
    Ts[:,:3,:3] = Rs
    Ts[:,:3,3] = vs
    Ts[:,:3,4] = ps
    Ts[:,3,3] = 1
    Ts[:,4,4] = 1

    delta_ts = np.diff(times)

    gammas = _Gamma_se23_pim(delta_ts, gravs[:-1])
    phis = _Phi_se23_pim(Rs[:-1],vs[:-1],ps[:-1],delta_ts)

    if not remove_rotation_of_earth:
        upsilons = np.array([np.linalg.inv(gammas[i]@phis[i])@Ts[i+1] for i in range(len(Rs)-1)])
        deltaRijs = upsilons[:,:3,:3]
        deltavijs = upsilons[:,:3,3]
        deltapjs = upsilons[:,:3,4]

    else:
        omega = eTw[:3,:3].T@wei
        Omega = so3.skew(omega)
        gamma_Rijs = _Gamma_R_se23_pim(delta_ts, omega)
        gamma_vijs = _Gamma_v_se23_pim(delta_ts, omega, gravs[:-1])
        gamma_pijs = _Gamma_p_se23_pim(delta_ts, omega, gravs[:-1])

        deltaRijs = np.array([(gamma_Rijs[i] @ Rs[i]).T @ Rs[i+1] for i in range(len(Rs)-1)])
        deltavijs = np.array([Rs[i].T @ (gamma_Rijs[i].T @ (vs[i+1] + Omega@ps[i+1] - gamma_vijs[i]) - vs[i] - Omega@ps[i]) for i in range(len(Rs)-1)])
        deltapjs = np.array([Rs[i].T @ (gamma_Rijs[i].T @ (ps[i+1] - gamma_pijs[i]) - (vs[i] + Omega@ps[i])* delta_ts[i] - ps[i]) for i in range(len(Rs)-1)])
    

    return deltaRijs, deltavijs, deltapjs

def delta_R_jacobian_rotating_earth(Ri, Rj, GammaRij):
    """Calculate the Jacobian of deltaR when accounting for the rotation of the Earth
    And a update vector r is applied to the right of Ri or Rj.

    deltaR is (GammaRij @ Ri).T @ Rj = Ri.T @ GammaRij.T @ Rj

    The result after updating Ri is: deltaR' = Exp(r).T @ Ri.T @ GammaRij.T @ Rj
            = Exp(-r) @ Ri.T @ GammaRij.T @ Rj

    The result after updating Rj is: deltaR' = Ri.T @ GammaRij.T @ Rj @ Exp(r)

    outputs:
    - d_deltaR_d_Ri: np.ndarray of shape (3,3,3)
        The Jacobian of the deltaR with respect to Ri.
    - d_deltaR_d_Rj: np.ndarray of shape (3,3,3)
        The Jacobian of the deltaR with respect to Rj.
    """
    d_deltaR_dRi = np.zeros((3, 3, 3))
    d_expmaprT_dr = -so3.deriv_expmap(np.zeros(3))
    for i in range(3):
        d_deltaR_dRi[:,:,i] = d_expmaprT_dr[:,:,i] @ Ri.T @ GammaRij.T @ Rj

    d_deltaR_dRj = np.zeros((3, 3, 3))
    d_expmapr_dr = so3.deriv_expmap(np.zeros(3))
    for i in range(3):
        d_deltaR_dRj[:,:,i] = Ri.T @ GammaRij.T @ Rj @ d_expmapr_dr[:,:,i]

    return d_deltaR_dRi, d_deltaR_dRj

@njit
def deltaR_lie_jacobian_rotating_earth(Ri, Rj, GammaRij):
    """Lie Jacobian of deltaR when accounting for the rotation of the Earth and a
    3 element update vector is applied to the right of Ri or Rj.
    """
    d_deltaR_dRi = -Rj.T @ GammaRij @ Ri
    d_deltaR_dRj = np.eye(3)#(GammaRij @ Ri).T
    return d_deltaR_dRi, d_deltaR_dRj

def delta_v_jacobian_rotating_earth(Ri,vi,vj,pi,pj,GammaRij,Gammavij,Omega):
    """Calculate the Jacobian of delta_v when accounting for the rotation of the Earth
    And a 3 element update vector is applied to the right of Ri or arithmetically
    to vi, vj, pi, or pi.

    
    Outputs:
    - d_delta_v_d_Ri: np.ndarray of shape (3,3)
        The Jacobian of the delta_v with respect to Ri.
    - d_delta_v_d_vi: np.ndarray of shape (3,3)
        The Jacobian of the delta_v with respect to vi.
    - d_delta_v_d_vj: np.ndarray of shape (3,3)
        The Jacobian of the delta_v with respect to vj.
    - d_delta_v_d_pi: np.ndarray of shape (3,3)
        The Jacobian of the delta_v with respect to pi.
    - d_delta_v_d_pj: np.ndarray of shape (3,3)
        The Jacobian of the delta_v with respect to pj.
    """
    d_delta_v_dRi = np.zeros((3, 3))
    d_expmaprT_dr = -so3.deriv_expmap(np.zeros(3))
    for i in range(3):
        d_delta_v_dRi[:,i] = d_expmaprT_dr[:,:,i] @ Ri.T @ (GammaRij.T @ (vj +Omega@pj - Gammavij) - vi - Omega@pi)

    d_delta_v_d_vi = np.zeros((3, 3))
    for i in range(3):
        d_delta_v_d_vi[:,i] = - Ri.T @ so3.I3[i]

    d_delta_v_d_vj = np.zeros((3, 3))
    for i in range(3):
        d_delta_v_d_vj[:,i] = Ri.T @ GammaRij.T @ so3.I3[i]

    d_delta_v_d_pi = np.zeros((3, 3))
    for i in range(3):
        d_delta_v_d_pi[:,i] = -Ri.T @ Omega @ so3.I3[i]

    d_delta_v_d_pj = np.zeros((3, 3))
    for i in range(3):
        d_delta_v_d_pj[:,i] = Ri.T @ GammaRij.T @ Omega @ so3.I3[i]

    return d_delta_v_dRi, d_delta_v_d_vi, d_delta_v_d_vj, d_delta_v_d_pi, d_delta_v_d_pj

@njit
def delta_v_lie_jacobian_rotating_earth(Ri,vi,vj,pi,pj,GammaRij,Gammavij,Omega):
    
    d_delta_v_dRi = Ri.T @ so3.skew(GammaRij.T @ (vj + Omega@pj - Gammavij) - vi - Omega@pi) @ Ri
    d_delta_v_d_vi = -Ri.T
    d_delta_v_d_vj = Ri.T @ GammaRij.T
    d_delta_v_d_pi = -Ri.T @ Omega
    d_delta_v_d_pj = Ri.T @ GammaRij.T @ Omega
    return d_delta_v_dRi, d_delta_v_d_vi, d_delta_v_d_vj, d_delta_v_d_pi, d_delta_v_d_pj    

def delta_p_jacobian_rotating_earth(Ri,vi,pi,pj,GammaRij,Gammapij,Omega,delta_t):
    """Calculate the Jacobian of delta_p when accounting for the rotation of the Earth
    And a 3 element update vector is applied to the right of Ri or arithmetically
    to vi, vj, pi, or pi.

    Outputs:
    - d_delta_p_d_Ri: np.ndarray of shape (3,3)
        The Jacobian of the delta_p with respect to Ri.
    - d_delta_p_d_vi: np.ndarray of shape (3,3)
        The Jacobian of the delta_p with respect to vi.
    - d_delta_p_d_pi: np.ndarray of shape (3,3)
        The Jacobian of the delta_p with respect to pi.
    - d_delta_p_d_pj: np.ndarray of shape (3,3)
        The Jacobian of the delta_p with respect to pj.
    """
    d_delta_p_dRi = np.zeros((3, 3))
    d_expmaprT_dr = -so3.deriv_expmap(np.zeros(3))
    for i in range(3):
        d_delta_p_dRi[:,i] = d_expmaprT_dr[:,:,i] @ Ri.T @ (GammaRij.T @ (pj - Gammapij) - (vi + Omega@pi) * delta_t - pi)

    d_delta_p_d_vi = np.zeros((3, 3))
    for i in range(3):
        d_delta_p_d_vi[:,i] = - Ri.T @ so3.I3[i] * delta_t

    d_delta_p_d_pi = np.zeros((3, 3))
    for i in range(3):
        d_delta_p_d_pi[:,i] = -Ri.T @ (Omega @ so3.I3[i] * delta_t + so3.I3[i])

    d_delta_p_d_pj = np.zeros((3, 3))
    for i in range(3):
        d_delta_p_d_pj[:,i] = Ri.T @ GammaRij.T @ so3.I3[i]

    return d_delta_p_dRi, d_delta_p_d_vi, d_delta_p_d_pi, d_delta_p_d_pj
    
@njit
def delta_p_lie_jacobian_rotating_earth(Ri,vi,pi,pj,GammaRij,Gammapij,Omega,delta_t):
    
    d_delta_p_dRi = Ri.T @ so3.skew(GammaRij.T @ (pj - Gammapij) - (vi + Omega@pi) * delta_t - pi) @ Ri
    d_delta_p_d_vi = -Ri.T * delta_t
    d_delta_p_d_pi = -Ri.T @ (Omega * delta_t + so3.I3)
    d_delta_p_d_pj = Ri.T @ GammaRij.T
    return d_delta_p_dRi, d_delta_p_d_vi, d_delta_p_d_pi, d_delta_p_d_pj

def test_se23(
        remove_rotation_of_earth:bool = False
):
    
    imu_data, pva_data, eTw = load_noiseless_data()

    # # Plot the true pva data
    # plt.figure()
    # plt.plot(pva_data[:, 0], pva_data[:, 1], label=r'$p_{w,x}$')
    # plt.plot(pva_data[:, 0], pva_data[:, 2], label=r'$p_{w,y}$')
    # plt.plot(pva_data[:, 0], pva_data[:, 3], label=r'$p_{w,z}$')
    # plt.legend()
    # plt.title('True Position')
    # plt.ylabel('m')
    # plt.grid()

    # plt.figure()
    # plt.plot(pva_data[:, 0], pva_data[:, 4], label=r'$v_{w,x}$')
    # plt.plot(pva_data[:, 0], pva_data[:, 5], label=r'$v_{w,y}$')
    # plt.plot(pva_data[:, 0], pva_data[:, 6], label=r'$v_{w,z}$')
    # plt.legend()    
    # plt.title('True Velocity')
    # plt.ylabel('m/s')
    # plt.grid()

    # plt.figure()
    # plt.plot(pva_data[:, 0], pva_data[:, 7], label=r'$\phi$')
    # plt.plot(pva_data[:, 0], pva_data[:, 8], label=r'$\theta$')
    # plt.plot(pva_data[:, 0], pva_data[:, 9], label=r'$\psi$')
    # plt.legend()
    # plt.title('True Atttitude')
    # plt.ylabel('degrees')
    # plt.grid()

    # Preintegrate the IMU data
    pim_times = imu_data[:, 0][::100]
    accel_biases = np.zeros((len(imu_data), 3))
    gyro_biases = np.zeros((len(imu_data), 3))
    pim_deltaR, pim_deltav, pim_deltap = preintegrate_imu(pim_times, imu_data, accel_biases, gyro_biases)

    # Calculate the true delta_R, delta_v, and delta_p
    deltaRijs, deltavijs, deltapijs = calc_true_deltas_se23(pim_times, pva_data, eTw, remove_rotation_of_earth=remove_rotation_of_earth)

        

    # Compare the preintegrated measurements to the true pva
    delta_R_diffs = np.array([so3.logmap(pim_deltaR[i].T @ deltaRijs[i]) for i in range(len(pim_deltaR))])
    delta_v_diffs = pim_deltav - deltavijs
    delta_p_diffs = pim_deltap - deltapijs

    # Plot the differences over time
    plt.figure()
    plt.plot(pim_times[1:], delta_R_diffs[:, 0], label=r'$\Delta$R diffs [x]')
    plt.plot(pim_times[1:], delta_R_diffs[:, 1], label=r'$\Delta$R diffs [y]')
    plt.plot(pim_times[1:], delta_R_diffs[:, 2], label=r'$\Delta$R diffs [z]')
    plt.legend()
    plt.title(r'$\Delta$R diffs with $SE_2(3)$')
    plt.grid()

    plt.figure()
    plt.plot(pim_times[1:], delta_v_diffs[:, 0], label=r'$\Delta$v diffs [x]')
    plt.plot(pim_times[1:], delta_v_diffs[:, 1], label=r'$\Delta$v diffs [y]')
    plt.plot(pim_times[1:], delta_v_diffs[:, 2], label=r'$\Delta$v diffs [z]')
    plt.legend()
    plt.title(r'$\Delta$v diffs with $SE_2(3)$')
    plt.grid()

    plt.figure()
    plt.plot(pim_times[1:], delta_p_diffs[:, 0], label=r'$\Delta$p diffs [x]')
    plt.plot(pim_times[1:], delta_p_diffs[:, 1], label=r'$\Delta$p diffs [y]')
    plt.plot(pim_times[1:], delta_p_diffs[:, 2], label=r'$\Delta$p diffs [z]')
    plt.legend()
    plt.title(r'$\Delta$p diffs with $SE_2(3)$')
    plt.grid()

    # # Plot the true deltas and the preintegrated deltas over time
    # pim_logmapped = np.array([so3.logmap(pim_deltaR[i]) for i in range(len(pim_deltaR))])
    # delta_R_logmapped = np.array([so3.logmap(deltaRijs[i]) for i in range(len(deltaRijs))])
    # plt.figure()
    # plt.plot(pim_times[1:], pim_logmapped[:, 0], label=r'$\Delta$R PIM [x]')
    # plt.plot(pim_times[1:], pim_logmapped[:, 1], label=r'$\Delta$R PIM [y]')
    # plt.plot(pim_times[1:], pim_logmapped[:, 2], label=r'$\Delta$R PIM [z]')
    # plt.plot(pim_times[1:], delta_R_logmapped[:, 0], label=r'$\Delta$R true [x]',linestyle='--')
    # plt.plot(pim_times[1:], delta_R_logmapped[:, 1], label=r'$\Delta$R true [y]',linestyle='--')
    # plt.plot(pim_times[1:], delta_R_logmapped[:, 2], label=r'$\Delta$R true [z]',linestyle='--')
    # plt.legend()
    # plt.title(r'$\Delta$R with $SE_2(3)$')
    # plt.grid()

    # plt.figure()
    # plt.plot(pim_times[1:], pim_deltav[:, 0], label=r'$\Delta$v PIM [x]')
    # plt.plot(pim_times[1:], pim_deltav[:, 1], label=r'$\Delta$v PIM [y]')
    # plt.plot(pim_times[1:], pim_deltav[:, 2], label=r'$\Delta$v PIM [z]')
    # plt.plot(pim_times[1:], deltavijs[:, 0], label=r'$\Delta$v true [x]',linestyle='--')
    # plt.plot(pim_times[1:], deltavijs[:, 1], label=r'$\Delta$v true [y]',linestyle='--')
    # plt.plot(pim_times[1:], deltavijs[:, 2], label=r'$\Delta$v true [z]',linestyle='--')
    # plt.legend()
    # plt.title(r'$\Delta$v with $SE_2(3)$')
    # plt.grid()

    # plt.figure()
    # plt.plot(pim_times[1:], pim_deltap[:, 0], label=r'$\Delta$p PIM [x]')
    # plt.plot(pim_times[1:], pim_deltap[:, 1], label=r'$\Delta$p PIM [y]')
    # plt.plot(pim_times[1:], pim_deltap[:, 2], label=r'$\Delta$p PIM [z]')
    # plt.plot(pim_times[1:], deltapijs[:, 0], label=r'$\Delta$p true [x]',linestyle='--')
    # plt.plot(pim_times[1:], deltapijs[:, 1], label=r'$\Delta$p true [y]',linestyle='--')
    # plt.plot(pim_times[1:], deltapijs[:, 2], label=r'$\Delta$p true [z]',linestyle='--')
    # plt.legend()
    # plt.title(r'$\Delta$p with $SE_2(3)$')
    # plt.grid()

def test_se23_bias_update():
    
    imu_data, pva_data, eTw = load_noiseless_data()

    # Preintegrate the IMU data
    pim_times = imu_data[:, 0][::100]
    accel_biases = np.zeros((len(imu_data), 3))
    gyro_biases = np.zeros((len(imu_data), 3))
    pim_deltaR, pim_deltav, pim_deltap = preintegrate_imu(pim_times, imu_data, accel_biases, gyro_biases)

    upsilons = _Upsilon_from_measurements(np.diff(imu_data[:,0]), imu_data[1:,4:7], imu_data[1:, 1:4])
    Fs = _F_se23_pim(np.diff(imu_data[:,0]), imu_data[1:, 1:4])
    Gs = _G_se23_pim(np.diff(imu_data[:,0]), imu_data[1:, 4:7])

    # # Recalculate with some bias on gyro_x - passes
    # gyro_biases[:, 0] = 0.001    
    # pim_deltaR_bias, pim_deltav_bias, pim_deltap_bias = preintegrate_imu(pim_times, imu_data, accel_biases, gyro_biases)



    # bias_update_jacobian_0 = calc_bias_update_jacobian(upsilons[:100], Fs[:100], Gs[:100])

    bias_update_jacobians = np.array([calc_bias_update_jacobian(upsilons[i*100:(i+1)*100], Fs[i*100:(i+1)*100], Gs[i*100:(i+1)*100],pim_deltaR[i]) for i in range(len(pim_times)-1)])

 
    # Now checkout bias on gyro_y - passes
    gyro_biases[:, 1] = 0.001
    accel_biases *= 0
    # Reintegrate the IMU data with the new biases
    pim_deltaR_bias, pim_deltav_bias, pim_deltap_bias = preintegrate_imu(pim_times, imu_data, accel_biases, gyro_biases)
    should_be_jac_R = np.array([so3.logmap(pim_deltaR[i].T @ pim_deltaR_bias[i]) for i in range(len(pim_deltaR))])/np.linalg.norm(gyro_biases[0])
    change_pim_deltav = pim_deltav_bias - pim_deltav
    change_pim_deltap = pim_deltap_bias - pim_deltap

    # Checkout bias on accel_z
    gyro_biases *= 0
    accel_biases[:, 2] = 0.001
    # Reintegrate the IMU data with the new biases
    pim_deltaR_bias, pim_deltav_bias, pim_deltap_bias = preintegrate_imu(pim_times, imu_data, accel_biases, gyro_biases)
    should_be_jac_v = (pim_deltav_bias - pim_deltav)/np.linalg.norm(accel_biases[0])
    should_be_jac_p = (pim_deltap_bias - pim_deltap)/np.linalg.norm(accel_biases[0])
    should_be_jac_R = np.array([so3.logmap(pim_deltaR[i].T @ pim_deltaR_bias[i]) for i in range(len(pim_deltaR))])/np.linalg.norm(accel_biases[0])

    stop_here =True

def test_delta_jacobians():
    
    imu_data, pva_data, eTw = load_noiseless_data()

    # Preintegrate the IMU data
    pim_times = imu_data[:, 0][::100]
    accel_biases = np.zeros((len(imu_data), 3))
    gyro_biases = np.zeros((len(imu_data), 3))
    pim_deltaR, pim_deltav, pim_deltap = preintegrate_imu(pim_times, imu_data, accel_biases, gyro_biases)

    deltaRijs, deltavijs, deltapijs = calc_true_deltas_se23(pim_times, pva_data, eTw)

    rounded_pim_times = np.round(pim_times, 2)
    rounded_pva_times = np.round(pva_data[:, 0], 2)
    true_pva_pim_times = pva_data[np.isin(rounded_pva_times, rounded_pim_times)]
    times = true_pva_pim_times[:, 0]
    ps = true_pva_pim_times[:, 1:4]
    vs = true_pva_pim_times[:, 4:7]
    rpys = true_pva_pim_times[:, 7:10]
    Rs = np.array([r3f.rpy_to_dcm(rpy*DEG_TO_RADIANS).T for rpy in rpys])

    # Calculate the Jacobians

def test_se23_residual_jacobians():
    
    imu_data, pva_data, eTw = load_noiseless_data()

    # Preintegrate the IMU data
    pim_times = imu_data[:, 0][::100]
    accel_biases = np.zeros((len(imu_data), 3))
    gyro_biases = np.zeros((len(imu_data), 3))
    pim_deltaR, pim_deltav, pim_deltap = preintegrate_imu(pim_times, imu_data, accel_biases, gyro_biases)
    upsilon_hats = np.array([_Upsilon_se23_pim(pim_deltaR[i], pim_deltav[i], pim_deltap[i]) for i in range(len(pim_deltaR))])

    upsilon_meass = _Upsilon_from_measurements(np.diff(imu_data[:,0]), imu_data[1:,4:7], imu_data[1:, 1:4])
    Fs = _F_se23_pim(np.diff(imu_data[:,0]), imu_data[1:, 1:4])
    Gs = _G_se23_pim(np.diff(imu_data[:,0]), imu_data[1:, 4:7])


    deltaRijs, deltavijs, deltapijs = calc_true_deltas_se23(pim_times, pva_data, eTw, remove_rotation_of_earth=True)
    upsilons = np.array([_Upsilon_se23_pim(deltaRijs[i], deltavijs[i], deltapijs[i]) for i in range(len(deltaRijs))])
   
    rounded_pim_times = np.round(pim_times, 2)
    rounded_pva_times = np.round(pva_data[:, 0], 2)
    true_pva_pim_times = pva_data[np.isin(rounded_pva_times, rounded_pim_times)]
    times = true_pva_pim_times[:, 0]
    ps = true_pva_pim_times[:, 1:4]
    vs = true_pva_pim_times[:, 4:7]
    rpys = true_pva_pim_times[:, 7:10]
    Rs = np.array([r3f.rpy_to_dcm(rpy*DEG_TO_RADIANS).T for rpy in rpys])
    pes = np.array([eTw[:3,:3]@p+eTw[:3,3] for p in ps])
    grav_es = np.array([grav_somigliana_pe(pe) for pe in pes])
    gravs = np.array([eTw[:3, :3].T @ grav_e for grav_e in grav_es])
    omega = eTw[:3,:3].T@wei
    Omega = so3.skew(omega)

    # Calculate the Jacobians - step 0
    GammaR01 = _Gamma_R_se23_pim(pim_times[1]-pim_times[0], omega)
    Gammav01 = _Gamma_v_se23_pim(pim_times[1]-pim_times[0], omega, gravs[0])
    Gammap01 = _Gamma_p_se23_pim(pim_times[1]-pim_times[0], omega, gravs[0])
    d_deltaR_dRi, d_deltaR_dRj = deltaR_lie_jacobian_rotating_earth(Rs[0], Rs[1], GammaR01)
    d_deltav_dRi, d_deltav_d_vi, d_deltav_d_vj, d_deltav_d_pi, d_deltav_d_pj = delta_v_lie_jacobian_rotating_earth(
        Rs[0], vs[0], vs[1], ps[0], ps[1], GammaR01, Gammav01, Omega
    )
    d_deltap_dRi, d_deltap_d_vi, d_deltap_d_pi, d_deltap_d_pj = delta_p_lie_jacobian_rotating_earth(
        Rs[0], vs[0], ps[0], ps[1], GammaR01, Gammap01, Omega, pim_times[1]-pim_times[0]
    )

    bias_update_jacobian = calc_bias_update_jacobian(upsilon_meass[:100], Fs[:100], Gs[:100], pim_deltaR[0])

    residual = se23.diamond_minus_se23(upsilons[0], upsilon_hats[0])

    # Numerical Jacobian - Ri, Rj, vi, pi, vj, pj. pass. 
    update_vec = np.array([0,0,0.001])
    updated_R1 = Rs[1]@ so3.expmap(update_vec)
    updated_R1_rpy = r3f.dcm_to_rpy(updated_R1.T)*RADIANS_TO_DEG
    updated_pva = true_pva_pim_times.copy()
    updated_pva[1, 7:10] = updated_R1_rpy
    updated_p0 = ps[0] + update_vec
    updated_pva = true_pva_pim_times.copy()
    updated_pva[0, 1:4] = updated_p0
    updated_deltaRijs, updated_deltavijs, updated_deltapijs = calc_true_deltas_se23(pim_times, updated_pva, eTw, remove_rotation_of_earth=True)
    updated_residual = se23.diamond_minus_se23(
        _Upsilon_se23_pim(updated_deltaRijs[0], updated_deltavijs[0], updated_deltapijs[0]),
        upsilon_hats[0]
    )

    # Evaluate Jacobian for bias updates - pass.
    bias_update_vec = np.zeros(6)
    bias_update_vec[0] = 0.001
    new_upsilon_hat = se23.diamond_plus_se23(
        upsilon_hats[0],
        bias_update_jacobian[:,0]*0.001
    )
    updated_residual = se23.diamond_minus_se23(
        upsilons[0],
        new_upsilon_hat
    )
    numerical_deriv = (updated_residual - residual) / np.linalg.norm(bias_update_vec)
    d_residual_d_bias = -se23.adjoint_se23(spla.inv(upsilon_hats[0])) @ se23.adjoint_se23(upsilon_hats[0]) @ bias_update_jacobian

    gyro_biases[:, 0] = 0.001
    pim_deltaR_bias, pim_deltav_bias, pim_deltap_bias = preintegrate_imu(pim_times, imu_data, accel_biases, gyro_biases)
    upsilon_hats_bias_0 = _Upsilon_se23_pim(pim_deltaR_bias[0], pim_deltav_bias[0], pim_deltap_bias[0])

    stop_here = True


def preintegrate_imu_with_covariance_and_bias_update_jacobians(
        pim_times: np.ndarray, 
        imu_data: np.ndarray, 
        accel_biases: np.ndarray,
        gyro_biases: np.ndarray,
        accel_white_noise: np.ndarray = np.eye(3),
        gyro_white_noise: np.ndarray = np.eye(3)
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Returns:
    - tuple with:
        - Ups_hats : np.ndarray of shape (n, 5, 5) where n is the number of intervals.
        - Sigmas : np.ndarray of shape (n, 9, 9) where n is the number of intervals.
        - bias_jacobians: np.ndarray of shape (n, 9, 6) where n is the number of intervals.
        - Gamma_Rijs: np.ndarray of shape (n, 3, 3) where n is the number of intervals.
        - Gamma_vijs: np.ndarray of shape (n, 3, 3) where n is the number of intervals.
        - Gamma_pijs: np.ndarray of shape (n, 3, 3) where n is the number of intervals.

    """
    # from matplotlib import pyplot as plt
    # imu_data_filtered = filter_imu_data(data=imu_data, cutoff_hz=10)
    # plt.figure() # Plot accel msmts over time for each axis on 3 subplots
    # plt.subplot(3,1,1)
    # plt.plot(imu_data[:, 0], imu_data[:, 1], label='Accel X')
    # plt.plot(imu_data[:, 0], imu_data_filtered[:, 1], label='Accel X Filtered', linestyle='--')
    # plt.legend()
    # plt.title('Accel X')
    # plt.subplot(3,1,2)
    # plt.plot(imu_data[:, 0], imu_data[:, 2], label='Accel Y')
    # plt.plot(imu_data[:, 0], imu_data_filtered[:, 2], label='Accel Y Filtered', linestyle='--')
    # plt.legend()
    # plt.title('Accel Y')
    # plt.subplot(3,1,3)
    # plt.plot(imu_data[:, 0], imu_data[:, 3], label='Accel Z')
    # plt.plot(imu_data[:, 0], imu_data_filtered[:, 3], label='Accel Z Filtered', linestyle='--')
    # plt.legend()
    # plt.title('Accel Z')
    # plt.xlabel('Time (s)')
    # plt.tight_layout()

    # plt.figure() # Plot gyro msmts over time for each axis on 3 subplots
    # plt.subplot(3,1,1)
    # plt.plot(imu_data[:, 0], imu_data[:, 4], label='Gyro X')
    # plt.plot(imu_data[:, 0], imu_data_filtered[:, 4], label='Gyro X Filtered', linestyle='--')
    # plt.legend()
    # plt.title('Gyro X')
    # plt.subplot(3,1,2)
    # plt.plot(imu_data[:, 0], imu_data[:, 5], label='Gyro Y')
    # plt.plot(imu_data[:, 0], imu_data_filtered[:, 5], label='Gyro Y Filtered', linestyle='--')
    # plt.legend()
    # plt.title('Gyro Y')
    # plt.subplot(3,1,3)
    # plt.plot(imu_data[:, 0], imu_data[:, 6], label='Gyro Z')
    # plt.plot(imu_data[:, 0], imu_data_filtered[:, 6], label='Gyro Z Filtered', linestyle='--')
    # plt.legend()
    # plt.title('Gyro Z')
    # plt.xlabel('Time (s)')
    # plt.tight_layout()

    # Preintegrate the IMU data
    # imu_data = imu_data_filtered.copy()
    msmt_times = imu_data[:, 0]
    accel_msmts = imu_data[:, 1:4]
    gyro_msmts = imu_data[:, 4:7]
    n_pim_intervals = len(pim_times) - 1
    pim_deltaR = np.zeros((n_pim_intervals, 3, 3))
    pim_deltav = np.zeros((n_pim_intervals, 3))
    pim_deltap = np.zeros((n_pim_intervals, 3))

    pim_sigmas = np.zeros((n_pim_intervals, 9, 9))
    pim_bias_jacobians = np.zeros((n_pim_intervals, 9, 6))
    for pim_interval in range(n_pim_intervals):
        i_idx = np.searchsorted(msmt_times, pim_times[pim_interval])
        j_idx = np.searchsorted(msmt_times, pim_times[pim_interval+1])

        delta_R_ik = np.eye(3)
        delta_v_ik = np.zeros(3)
        delta_p_ik = np.zeros(3)

        accel_msmts_interval_corrected = accel_msmts[i_idx:j_idx+1] - accel_biases[pim_interval]
        gyro_msmts_interval_corrected = gyro_msmts[i_idx:j_idx+1] - gyro_biases[pim_interval]
        dt_interval = np.ones_like(accel_msmts_interval_corrected[:,0])*np.median(np.diff(msmt_times[i_idx:j_idx+1]))
        upsilons = _Upsilon_from_measurements(dt_interval, gyro_msmts_interval_corrected, accel_msmts_interval_corrected)
        Fs = _F_se23_pim(dt_interval, accel_msmts_interval_corrected)
        Gs = _G_se23_pim(dt_interval, gyro_msmts_interval_corrected)

        for k in range(i_idx+1, min(j_idx+1,len(msmt_times))):
            deltaT = msmt_times[k] - msmt_times[k-1]
            delta_p_ik = delta_p_ik + delta_v_ik*deltaT + 0.5*delta_R_ik@(accel_msmts[k] - accel_biases[pim_interval])*(deltaT**2) 
            delta_v_ik = delta_v_ik + delta_R_ik @ (accel_msmts[k] - accel_biases[pim_interval])*deltaT 
            delta_R_ik = delta_R_ik @ so3.expmap((gyro_msmts[k] - gyro_biases[pim_interval]) * deltaT)

        bias_jacobian = calc_bias_update_jacobian(
            upsilons,
            Fs,
            Gs,
            delta_R_ik
        )
        pim_bias_jacobians[pim_interval] = bias_jacobian

        sigma = _Sigma_ij_se23_pim(
            delta_ts=dt_interval,
            omegas = gyro_msmts_interval_corrected,
            accels = accel_msmts_interval_corrected,
            gyro_white_noise= gyro_white_noise,
            accel_white_noise= accel_white_noise,
            inv_upsilons = np.array([se23.inv_se23(upsilon) for upsilon in upsilons])
        )

        pim_sigmas[pim_interval] = sigma

        # Save the final changes over the interval
        deltaR_ij = delta_R_ik
        deltav_ij = delta_v_ik
        deltap_ij = delta_p_ik

        pim_deltaR[pim_interval] = deltaR_ij
        pim_deltav[pim_interval] = deltav_ij
        pim_deltap[pim_interval] = deltap_ij

    Ups_hats = np.array([se23._se23_from_components(pim_deltaR[i], pim_deltav[i], pim_deltap[i]) for i in range(n_pim_intervals)])
    
    return Ups_hats, pim_sigmas, pim_bias_jacobians

@dataclass
class IMUMeasurement:

    timestamp: float
    accel: np.ndarray
    gyro: np.ndarray

@dataclass
class PIMData:

    UpsilonMsmt: np.ndarray
    GammaR: np.ndarray
    GammaV: np.ndarray
    GammaP: np.ndarray
    omega: np.ndarray
    delta_t: float
    accel_bias_jacobian: np.ndarray
    gyro_bias_jacobian: np.ndarray
    error_covariance: np.ndarray

@dataclass
class PreintegratedIMUFactor:

    state_keys: List[str]
    error_func: Callable[...,Any]
    jacobian_func: Callable[...,Any]
    error_weight: np.ndarray
    accel_bias_jacobian: np.ndarray
    gyro_bias_jacobian: np.ndarray


def preintegrate_imu_one_interval(
        imu_measurements: List[IMUMeasurement],
        accel_bias: np.ndarray,
        gyro_bias: np.ndarray,
        accel_white_noise: np.ndarray,
        gyro_white_noise: np.ndarray,
        omega: np.ndarray,
        gravity: np.ndarray
) -> PIMData:
    """Produce the PIMData given measurements and necessary data.

    Inputs:
    imu_measurements: List[IMUMeasurement], List of IMU measurements over the interval to be preintegrated.
    accel_bias: np.ndarray of shape (3,), the accelerometer bias to be used for correcting the measurements.
    gyro_bias: np.ndarray of shape (3,), the gyroscope bias to be used for correcting the measurements.
    accel_white_noise: np.ndarray of shape (3,3), the covariance of the accelerometer white noise.
    gyro_white_noise: np.ndarray of shape (3,3), the covariance of the gyroscope white noise.
    omega: np.ndarray of shape (3), the angular velocity of the Earth in the desired world frame (same as gravity)
    gravity: np.ndarray of shape (3,), the gravity vector in the desired world frame (same as omega)

    Outputs:
    PIMData dataclass 
    """
    # Step 1: Preintegrate the IMU measurements over the interval to get deltas
    imu_measurements.sort(key=lambda x: x.timestamp) # sort by time
    msmt_times = np.array([msmt.timestamp for msmt in imu_measurements])
    accel_msmts = np.array([msmt.accel for msmt in imu_measurements])
    gyro_msmts = np.array([msmt.gyro for msmt in imu_measurements])

    accel_msmts_corrected = accel_msmts - accel_bias
    gyro_msmts_corrected = gyro_msmts - gyro_bias

    delta_ts = np.diff(msmt_times)
    upsilons = _Upsilon_from_measurements(delta_ts, gyro_msmts_corrected[1:], accel_msmts_corrected[1:])
    Fs = _F_se23_pim(delta_ts, accel_msmts_corrected[1:])
    Gs = _G_se23_pim(delta_ts, gyro_msmts_corrected[1:])

    delta_p_ik = np.zeros(3)
    delta_v_ik = np.zeros(3)
    delta_R_ik = np.eye(3)


    for k in range(1, len(msmt_times)):
        deltaT = delta_ts[k-1]
        delta_p_ik = delta_p_ik + delta_v_ik*deltaT + 0.5*delta_R_ik@(accel_msmts_corrected[k])*(deltaT**2)
        delta_v_ik = delta_v_ik + delta_R_ik @ (accel_msmts_corrected[k])*deltaT
        delta_R_ik = delta_R_ik @ so3.expmap((gyro_msmts_corrected[k]) * deltaT)

    # Step 2: Calculate the Jacobians of the preintegrated measurements with respect to the biases
    full_bias_jacobian = calc_bias_update_jacobian(
        upsilons,
        Fs,
        Gs,
        delta_R_ik
    )

    gyro_bias_jacobian = full_bias_jacobian[:, :3]
    accel_bias_jacobian = full_bias_jacobian[:, 3:]

    # Step 3: Calculate the covariance of the preintegrated measurements
    sigma = _Sigma_ij_se23_pim(
        delta_ts=delta_ts,
        omegas = gyro_msmts_corrected[1:],
        accels = accel_msmts_corrected[1:],
        gyro_white_noise= gyro_white_noise,
        accel_white_noise= accel_white_noise,
        inv_upsilons = np.array([se23.inv_se23(upsilon) for upsilon in upsilons])
    )
    interval_delta_t = np.sum(delta_ts)

    return PIMData(
        UpsilonMsmt = se23._se23_from_components(delta_R_ik, delta_v_ik, delta_p_ik),
        GammaR = _Gamma_R_se23_pim(interval_delta_t, omega), # Placeholder, should be calculated properly
        GammaV = _Gamma_v_se23_pim(interval_delta_t, omega, gravity), # Placeholder, should be calculated properly
        GammaP = _Gamma_p_se23_pim(interval_delta_t, omega, gravity), # Placeholder, should be calculated properly
        omega = omega,
        delta_t = interval_delta_t,
        accel_bias_jacobian = accel_bias_jacobian,
        gyro_bias_jacobian = gyro_bias_jacobian,
        error_covariance = sigma
    )

#@njit # This is njit friendly.
def pim_se23_error(state_tuple, pim_data: PIMData):
    """This calculates the error for a PIM factor given the current state estimates
    and the PIMData.
    This calculates Upsilon using equations 86-88 from Associating Uncertainty to Extended Poses.

    Inputs:
    state_tuple: Tuple containing the estimated values in the order 
    (Ri, vi, pi, Rj, vj, pj, ba, bg) where:
        Ri: np.ndarray of shape (3,3), the estimated rotation matrix at time i.
        vi: np.ndarray of shape (3,), the estimated velocity at time i.
        pi: np.ndarray of shape (3,), the estimated position at time i.
        Rj: np.ndarray of shape (3,3), the estimated rotation matrix at time j.
        vj: np.ndarray of shape (3,), the estimated velocity at time j.
        pj: np.ndarray of shape (3,), the estimated position at time j.
        ba: np.ndarray of shape (3,), the estimated accelerometer bias.
        bg: np.ndarray of shape (3,), the estimated gyroscope bias.

    Outputs:
    - residual: np.ndarray of shape (9,), the error between the preintegrated measurements and the expected measurements given the current state estimates. 
    """
    Ri, vi, pi, Rj, vj, pj, _, _ = state_tuple
    Omega = so3.skew(pim_data.omega)

    # Calculate the expected preintegrated measurement (Upsilon) given the current state estimates and the PIMData
    ups = se23._se23_from_components(
        (pim_data.GammaR @ Ri).T @ Rj,
        Ri.T @ (pim_data.GammaR.T @ (vj + Omega@pj - pim_data.GammaV) - vi - Omega@pi),
        Ri.T @ (pim_data.GammaR.T @ (pj- pim_data.GammaP) - (vi + Omega@pi)*pim_data.delta_t - pi)
    )

    return se23.diamond_minus_se23(ups, pim_data.ups_hat)

#@njit # This is njit friendly.
def pim_se23_jacobians(state_tuple, pim_data: PIMData):
    """This calculates the Jacobians of the residual with respect to the state variables and the bias updates.
    
    Inputs:
    state_tuple: Tuple containing the estimated values in the order
    (Ri, vi, pi, Rj, vj, pj, ba, bg) where:
        Ri: np.ndarray of shape (3,3), the estimated rotation matrix at time i.
        vi: np.ndarray of shape (3,), the estimated velocity at time i.
        pi: np.ndarray of shape (3,), the estimated position at time i.
        Rj: np.ndarray of shape (3,3), the estimated rotation matrix at time j.
        vj: np.ndarray of shape (3,), the estimated velocity at time j.
        pj: np.ndarray of shape (3,), the estimated position at time j.
        ba: np.ndarray of shape (3,), the estimated accelerometer bias.
        bg: np.ndarray of shape (3,), the estimated gyroscope bias.
    pim_data: PIMData dataclass containing the necessary, pseudo-constant data

    
    Outputs:
    - d_residual_d_Ri: np.ndarray of shape (9,3), the Jacobian of the residual with respect to Ri.
    - d_residual_d_vi: np.ndarray of shape (9,3), the Jacobian of the residual with respect to vi.
    - d_residual_d_pi: np.ndarray of shape (9,3), the Jacobian of the residual with respect to pi.
    - d_residual_d_Rj: np.ndarray of shape (9,3), the Jacobian of the residual with respect to Rj.
    - d_residual_d_vj: np.ndarray of shape (9,3), the Jacobian of the residual with respect to vj.
    - d_residual_d_pj: np.ndarray of shape (9,3), the Jacobian of the residual with respect to pj.
    - d_residual_d_ba: np.ndarray of shape (9,3), the Jacobian of the residual with respect to the accelerometer bias.
    - d_residual_d_bg: np.ndarray of shape (9,3), the Jacobian of the residual with respect to the gyroscope bias.  
    """
    # Read in the data from the inputs
    Ri, vi, pi, Rj, vj, pj, _, _ = state_tuple
    bias_update_jacobian = np.hstack((pim_data.gyro_bias_jacobian, pim_data.accel_bias_jacobian))
    Omega = so3.skew(pim_data.omega)

    # Precompute sub-Jacobians for use in the chain-rule for the full Jacobian
    d_deltaR_d_Ri, d_deltaR_d_Rj = deltaR_lie_jacobian_rotating_earth(Ri, Rj, pim_data.GammaR)
    d_deltav_d_Ri, d_deltav_d_vi, d_deltav_d_vj, d_deltav_d_pi, d_deltav_d_pj = \
        delta_v_lie_jacobian_rotating_earth(Ri, vi, vj, pi, pj, pim_data.GammaR, pim_data.GammaV, Omega)
    d_deltap_d_Ri, d_deltap_d_vi, d_deltap_d_pi, d_deltap_d_pj = \
        delta_p_lie_jacobian_rotating_earth(Ri, vi, pi, pj, pim_data.GammaR,  pim_data.GammaP, Omega, pim_data.delta_t)
    
    # Construct the Jacobian of the residual with respect to the state variables
    # at time i (first time)
    d_ups_d_update0 = np.zeros((9, 9), dtype=np.float64)
    d_ups_d_update0[:3, :3] = d_deltaR_d_Ri
    d_ups_d_update0[3:6, :3] = d_deltav_d_Ri
    d_ups_d_update0[6:9, :3] = d_deltap_d_Ri
    d_ups_d_update0[3:6, 3:6] = d_deltav_d_vi
    d_ups_d_update0[6:9, 3:6] = d_deltap_d_vi
    d_ups_d_update0[3:6, 6:9] = d_deltav_d_pi
    d_ups_d_update0[6:9, 6:9] = d_deltap_d_pi
    d_residual_d_T0_update = d_ups_d_update0

    # Construct the Jacobian of the residual with respect to the state variables
    # at time j (second time)
    d_ups_d_update1 = np.zeros((9, 9), dtype=np.float64)
    d_ups_d_update1[:3, :3] = d_deltaR_d_Rj
    d_ups_d_update1[3:6, 3:6] = d_deltav_d_vj
    d_ups_d_update1[3:6, 6:9] = d_deltav_d_pj
    d_ups_d_update1[6:9, 6:9] = d_deltap_d_pj
    d_residual_d_T1_update = d_ups_d_update1

    # Construct the Jacobian of the residual with respect to the bias updates
    d_residual_d_bias_update = np.zeros((9,6), dtype=np.float64)
    d_residual_d_bias_update[:3,:] = (Ri.T@pim_data.GammaR.T@Rj).T@(-pim_data.ups_hat[:3,:3])@bias_update_jacobian[:3,:]
    d_residual_d_bias_update[3:6,:] = -bias_update_jacobian[3:6,:]
    d_residual_d_bias_update[6:9,:] = -bias_update_jacobian[6:9,:]
    d_residual_d_bias_update = d_residual_d_bias_update
    
    # Parse the Jacobians into the appropriate variables
    d_residual_d_Ri = d_residual_d_T0_update[:,:3]
    d_residual_d_vi = d_residual_d_T0_update[:,3:6]
    d_residual_d_pi = d_residual_d_T0_update[:,6:9]
    d_residual_d_Rj = d_residual_d_T1_update[:,:3]
    d_residual_d_vj = d_residual_d_T1_update[:,3:6]
    d_residual_d_pj = d_residual_d_T1_update[:,6:9]    
    d_residual_d_ba = d_residual_d_bias_update[:,3:]
    d_residual_d_bg = d_residual_d_bias_update[:,:3]
    
    return d_residual_d_Ri, d_residual_d_vi, d_residual_d_pi, d_residual_d_Rj, d_residual_d_vj, d_residual_d_pj, d_residual_d_ba, d_residual_d_bg

def build_pim_factor(
        state_keys: List[str],
        imu_measurements: List[IMUMeasurement],
        accel_bias: np.ndarray,
        gyro_bias: np.ndarray,
        accel_white_noise: np.ndarray,
        gyro_white_noise: np.ndarray,
        position_for_gravity_in_ecef: np.ndarray,
        rotation_from_local_ned_to_ecef: np.ndarray = np.eye(3),
)-> PreintegratedIMUFactor:
    """This function builds a PreintegratedIMUFactor given the necessary inputs.
    Inputs:
    state_keys: List[str], the keys for the states that this factor will connect
        in the factor graph. Should be in the order [Ri, vi, pi, Rj, vj, pj, ba, bg].
    imu_measurements: List[IMUMeasurement], the IMU measurements to be preintegrated.
    accel_bias: np.ndarray of shape (3,), the initial accelerometer bias to be 
        used for correcting the measurements.
    gyro_bias: np.ndarray of shape (3,), the initial gyroscope bias to be used 
        for correcting the measurements.
    accel_white_noise: np.ndarray of shape (3,3), the covariance of the 
        accelerometer white noise.
    gyro_white_noise: np.ndarray of shape (3,3), the covariance of the gyroscope 
        white noise.
    rotation_from_local_ned_to_ecef: np.ndarray of shape (3,3), the rotation matrix
        that rotates vectors from the local NED frame to ECEF coordinates. This is used
        to calculate the gravity vector and the Earth's rotation in the local frame.
        Defaults to np.eye(3), which corresponds to using ECEF and not a local frame.
    position_for_gravity_in_ecef: np.ndarray of shape (3,), the position in ECEF coordinates
        where the gravity vector should be calculated for. This is used to calculate 
        the gravity vector.
    
    Outputs:
    PreintegratedIMUFactor dataclass
    """
    # Calculate the gravity vector and Earth's rotation in the local frame
    eRw = rotation_from_local_ned_to_ecef
    grav_e = grav_somigliana_pe(position_for_gravity_in_ecef)
    grav = eRw.T @ grav_e
    omega = eRw.T @ wei

    # Preintegrate the IMU measurements to get the PIMData
    pim_data = preintegrate_imu_one_interval(
        imu_measurements=imu_measurements,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
        accel_white_noise=accel_white_noise,
        gyro_white_noise=gyro_white_noise,
        omega=omega,
        gravity=grav
    )

    # Build the PreintegratedIMUFactor dataclass
    pim_factor = PreintegratedIMUFactor(
        state_keys = state_keys,
        error_func = partial(pim_se23_error, pim_data=pim_data),
        jacobian_func = partial(pim_se23_jacobians, pim_data=pim_data),
        error_weight = weight_from_covariance(pim_data.error_covariance),
        accel_bias_jacobian = pim_data.accel_bias_jacobian,
        gyro_bias_jacobian = pim_data.gyro_bias_jacobian
    )

    return pim_factor


if __name__ == '__main__':
    # test_so3r3(remove_rotation_of_earth=True)
    # test_se23()
    # test_se23(remove_rotation_of_earth=True)
    test_se23_bias_update()
    test_se23_residual_jacobians()
    # plt.show()