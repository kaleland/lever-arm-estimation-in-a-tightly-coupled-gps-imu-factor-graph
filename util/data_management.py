
from util.constants import wei 
from util import config, se3, gnss, pim
import numpy as np

def import_gps_data(gnss_data, Tew, base_pr_sigma: float, base_phase_sigma: float, graph_start_time_gnss: int = 0):
    """Returns:
    A dictionary containing the GNSS data formatted for use in a factor graph.
        - clock_bias_keys: Set of clock bias keys for the graph
        - t_gnss: Unique GNSS times
        - pr_msmts: Pseudorange measurements in the format [time, pseudorange, sigma, PRN, satXYZ_w]
        - tdcp_msmts: Time-differenced carrier phase measurements in the format [t0, t1, phase_diff, sigma, PRN, satXYZ_w_t0, satXYZ_w_t1]
        - attitude_keys: Set of attitude keys for the graph
        - position_keys: Set of position keys for the graph.

    """
    gnss_data_times = gnss_data[:,0]
    gnss_data_times = gnss_data_times - graph_start_time_gnss
    prn = gnss_data[:,1]
    pr_msmts = gnss_data[:,2]
    phase_msmts = gnss_data[:,3]
    satXYZ_e = gnss_data[:,5:8]

    Twe = se3.inverse_T(Tew)
    satXYZ_w = np.array([
        (Twe@np.concatenate((satXYZ_e[i],[1])))[:3] for i in range(satXYZ_e.shape[0])
    ])

    cnr = gnss_data[:,8]
    elevation = np.deg2rad(gnss_data[:,10])

    pr_sigma = gnss.elDepWeight(elevation, base_pr_sigma)
    phase_sigma = gnss.elDepWeight(elevation, base_phase_sigma)

    pr_msmts = np.vstack([
        gnss_data_times,
        pr_msmts,
        pr_sigma,
        prn,
        satXYZ_w.T
    ]).T
    n_pr_msmts = pr_msmts.shape[0]

    last_phase_map = {} # Map storing last phase msmt for each PRN
    tdcp_msmts = []
    for phase_idx in range(phase_msmts.shape[0]):
        if prn[phase_idx] in last_phase_map:
            last_phase_data = last_phase_map[prn[phase_idx]]
            # if True: # For robust (non l2 graph)
            if gnss_data_times[phase_idx] - last_phase_data['time'] < 1.9: # Don't include TDCP measurements where there is 2 seconds or more between carrier-phase mesaurements
                tdcp_msmts.append([
                    last_phase_data['time'],
                    gnss_data_times[phase_idx],
                    phase_msmts[phase_idx] - last_phase_data['phase'],
                    phase_sigma[phase_idx] + last_phase_data['phase_sigma'], #np.sqrt(phase_sigma[phase_idx]**2 + last_phase_data['phase_sigma']**2),
                    prn[phase_idx],
                    last_phase_data['satXYZ_w'][0],
                    last_phase_data['satXYZ_w'][1],
                    last_phase_data['satXYZ_w'][2],
                    satXYZ_w[phase_idx][0],
                    satXYZ_w[phase_idx][1],
                    satXYZ_w[phase_idx][2],
                ])
        last_phase_map[prn[phase_idx]] = {
            'phase': phase_msmts[phase_idx],
            'time': gnss_data_times[phase_idx],
            'phase_sigma': phase_sigma[phase_idx],
            'satXYZ_w': satXYZ_w[phase_idx]
        }
    tdcp_msmts = np.array(tdcp_msmts)
    # Sort by t0, then t1, then PRN
    tdcp_msmts = tdcp_msmts[np.lexsort((tdcp_msmts[:,4],tdcp_msmts[:,1],tdcp_msmts[:,0]))]
    tdcp_msmts = tdcp_msmts.copy()
    n_tdcp_msmts = tdcp_msmts.shape[0]

    t_gnss = np.unique(gnss_data_times)

    # start_time_int = graph_start_time_gnss
    # t_gnss_offset = t_gnss - start_time_int
    clock_bias_keys = set(f'c{int(t)}' for t in t_gnss)
    attitude_keys = set(f'R{int(t)}' for t in t_gnss)
    position_keys = set(f'p{int(t)}' for t in t_gnss)

    return {
        'clock_bias_keys': clock_bias_keys,
        't_gnss': t_gnss,
        'pr_msmts': pr_msmts,
        'tdcp_msmts': tdcp_msmts,
        'attitude_keys': attitude_keys,
        'position_keys': position_keys,
    }

def import_real_gps_data(gnss_data, Tew, base_pr_sigma: float, base_phase_sigma: float, graph_start_time_gnss: int = 0):
    """Returns:
    A dictionary containing the GNSS data formatted for use in a factor graph.
        - t_gnss: Unique GNSS times
        - pr_msmts: Pseudorange measurements in the format [time, pseudorange, sigma, PRN, satXYZ_w]
        - tdcp_msmts: Time-differenced carrier phase measurements in the format [t0, t1, phase_diff, sigma, PRN, satXYZ_w_t0, satXYZ_w_t1].

    """
    gnss_data_times = gnss_data[:,0]
    gnss_data_times = gnss_data_times - graph_start_time_gnss
    prn = gnss_data[:,1]
    pr_msmts = gnss_data[:,2]
    phase_msmts = gnss_data[:,3]
    satXYZ_e = gnss_data[:,5:8]

    Twe = se3.inverse_T(Tew)
    satXYZ_w = np.array([
        (Twe@np.concatenate((satXYZ_e[i],[1])))[:3] for i in range(satXYZ_e.shape[0])
    ])

    elevation = np.deg2rad(gnss_data[:,10])

    pr_sigma = gnss.elDepWeight(elevation, base_pr_sigma)
    phase_sigma = gnss.elDepWeight(elevation, base_phase_sigma)

    pr_msmts = np.vstack([
        gnss_data_times,
        pr_msmts,
        pr_sigma,
        prn,
        satXYZ_w.T
    ]).T
    n_pr_msmts = pr_msmts.shape[0]

    last_phase_map = {} # Map storing last phase msmt for each PRN
    tdcp_msmts = []
    for phase_idx in range(phase_msmts.shape[0]):
        if prn[phase_idx] in last_phase_map:
            last_phase_data = last_phase_map[prn[phase_idx]]
            # if True: # For robust (non l2 graph)
            if gnss_data_times[phase_idx] - last_phase_data['time'] < 1.9: # Don't include TDCP measurements where there is 2 seconds or more between carrier-phase mesaurements
                tdcp_msmts.append([
                    last_phase_data['time'],
                    gnss_data_times[phase_idx],
                    phase_msmts[phase_idx] - last_phase_data['phase'],
                    phase_sigma[phase_idx] + last_phase_data['phase_sigma'], # np.sqrt(phase_sigma[phase_idx]**2 + last_phase_data['phase_sigma']**2),
                    prn[phase_idx],
                    last_phase_data['satXYZ_w'][0],
                    last_phase_data['satXYZ_w'][1],
                    last_phase_data['satXYZ_w'][2],
                    satXYZ_w[phase_idx][0],
                    satXYZ_w[phase_idx][1],
                    satXYZ_w[phase_idx][2],
                ])
        last_phase_map[prn[phase_idx]] = {
            'phase': phase_msmts[phase_idx],
            'time': gnss_data_times[phase_idx],
            'phase_sigma': phase_sigma[phase_idx],
            'satXYZ_w': satXYZ_w[phase_idx]
        }
    tdcp_msmts = np.array(tdcp_msmts)
    # Sort by t0, then t1, then PRN
    tdcp_msmts = tdcp_msmts[np.lexsort((tdcp_msmts[:,4],tdcp_msmts[:,1],tdcp_msmts[:,0]))]
    tdcp_msmts = tdcp_msmts.copy()
    n_tdcp_msmts = tdcp_msmts.shape[0]

    t_gnss = np.sort(np.unique(gnss_data_times))

    return {
        't_gnss': t_gnss,
        'pr_msmts': pr_msmts,
        'tdcp_msmts': tdcp_msmts,
    }
