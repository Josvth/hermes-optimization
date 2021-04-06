
import numpy as np
from numba import njit

import energy
import latency
import pointing
import vcm

@njit
def compute_fg(N_passes, O_matrix, t_sim_s, tof_s_list, fspl_dB_list, theta_rad_list, phi_rad_list,
               x_pass, Ptx_dBm_array, Gtx_dBi, GT_dBK, B_Hz, alpha,
               EsN0_req_dB_array, eta_bitsym_array, eta_maee_array, margin_dB,
               min_throughput, max_throughput, max_latency, max_energy, max_pointing, max_rate_rads
               ):

    # Calculate overlap constraint
    x_pass_tile = x_pass.repeat(N_passes).reshape((-1, N_passes))
    overlap = O_matrix * x_pass_tile * x_pass_tile.T
    #g_overlap = np.sum(overlap > 0) / int((N_passes * (N_passes + 1)) / 2)
    g_overlap = np.sum(overlap > 0)

    # Compute objectives
    f_throughput = 0  # Maximum throughput objective
    f_energy = 0  # Minimum energy objective
    f_pointing = 0  # Minimum pointing objective
    f_latency = 0  # Minimize latency objective

    rates_rads_array = np.zeros(N_passes)

    if g_overlap == 0 and np.sum(x_pass) > 0:
        pass_inds = np.nonzero(x_pass)[0]

        # Throughput
        b_s_array, e_s_array, linktime_s_array, f_throughput, vcm_array = vcm.compute_passes_throughput(
            pass_inds, tof_s_list, fspl_dB_list,
            Ptx_dBm_array, Gtx_dBi, GT_dBK, B_Hz, alpha,
            EsN0_req_dB_array, eta_bitsym_array, margin_dB)

        # Energy
        eta_maee_array = eta_maee_array[vcm_array]
        f_energy = energy.compute_passes_energy_maee(linktime_s_array, Ptx_dBm_array, eta_maee_array)

        # Pointing
        f_pointing, rates_rads_array[x_pass] = pointing.compute_pointing_passes(
            pass_inds, tof_s_list, theta_rad_list, phi_rad_list, Gtx_dBi)

        # Latency
        if b_s_array.size > 0:
            f_latency = latency.compute_max_latency_passes(b_s_array, e_s_array, t_sim_s)

    # Throughput minimum/maximum
    g_throughput_minimum = 0
    g_throughput_maximum = 0
    if min_throughput > 0:
        g_throughput_minimum = np.maximum((min_throughput - f_throughput) / min_throughput, 0)
    if max_throughput > 0:
        g_throughput_maximum = np.maximum((f_throughput - max_throughput) / max_throughput, 0)

    # Maximum latency
    g_latency_maximum = 0
    if max_latency > 0:
        g_latency_maximum = np.maximum((f_latency - max_latency) / max_latency, 0)

    # Maximum energy
    g_energy_maximum = 0
    if max_energy > 0:
        g_energy_maximum = np.maximum((f_energy - max_energy) / max_energy, 0)

    # Maximum pointing
    g_pointing_maximum = 0
    if max_pointing > 0:
        g_pointing_maximum = np.maximum((f_pointing - max_pointing) / max_pointing, 0)

    ff = np.array([-1 * f_throughput, f_latency, f_energy, f_pointing])
    gg = np.array([g_overlap, g_throughput_minimum, g_throughput_maximum, g_latency_maximum, g_energy_maximum, g_pointing_maximum])

    return ff, gg
