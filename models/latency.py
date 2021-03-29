from numba import prange, njit

import numpy as np


@njit(parallel=True)
def actual_begin_end(pass_inds, tof_s_list, margin_dB_list):
    b_s_array = np.empty(len(pass_inds), np.float64)
    e_s_array = np.empty(len(pass_inds), np.float64)

    for i in prange(len(pass_inds)):
        p = pass_inds[i]
        link_tof_s = tof_s_list[p][np.nonzero(margin_dB_list[i] >= 0)]
        b_s_array[i] = np.NaN if len(link_tof_s) == 0 else link_tof_s[0]
        e_s_array[i] = np.NaN if len(link_tof_s) == 0 else link_tof_s[-1]

    return b_s_array[~np.isnan(b_s_array)], e_s_array[~np.isnan(b_s_array)]


@njit(parallel=True)
def compute_latency_passes(tof_s_list, margin_dB_list, t_sim_s):

    b_s_array, e_s_array = actual_begin_end(tof_s_list, margin_dB_list)

    e_s_array = np.append([0.], e_s_array)
    b_s_array = np.append(b_s_array, t_sim_s)

    diff_s = b_s_array - e_s_array
    latency = np.sum(0.5 * (diff_s)**2)

    return latency / t_sim_s

@njit(parallel=True)
def compute_max_latency_passes(pass_inds, tof_s_list, margin_dB_list, t_sim_s):

    b_s_array, e_s_array = actual_begin_end(pass_inds, tof_s_list, margin_dB_list)

    e_s_array = np.append([0.], e_s_array)
    b_s_array = np.append(b_s_array, t_sim_s)

    diff_s = b_s_array - e_s_array

    return np.max(diff_s)

@njit(parallel=True)
def compute_max_latency_passes(b_s_array, e_s_array, t_sim_s):

    e_s_array = np.append([0.], e_s_array)
    b_s_array = np.append(b_s_array, t_sim_s)

    diff_s = b_s_array - e_s_array

    return np.max(diff_s)