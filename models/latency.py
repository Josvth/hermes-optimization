from numba import prange, njit

import numpy as np


#@njit(parallel=True)
def actual_begin_end(tof_s_list, margin_dB_list):
    b_s_array = np.empty(len(tof_s_list), np.float64)
    e_s_array = np.empty(len(tof_s_list), np.float64)

    for i in prange(len(b_s_array)):
        b_s_array[i] = tof_s_list[i][np.nonzero(margin_dB_list[i] >= 0)[0]]
        e_s_array[i] = tof_s_list[i][np.nonzero(margin_dB_list[i] >= 0)[-1]]

    return b_s_array, e_s_array


#@njit(parallel=True)
def compute_latency_passes(tof_s_list, margin_dB_list, t_sim_s):

    b_s_array, e_s_array = actual_begin_end(tof_s_list, margin_dB_list)

    e_s_array = np.insert(e_s_array, 0, 0.)
    b_s_array = np.append(b_s_array, t_sim_s)

    latency = 0.
    for i in prange(len(b_s_array)):
        diff_s = b_s_array[i] - e_s_array[i]
        latency += 0.5 * (diff_s)**2

    return latency / t_sim_s
