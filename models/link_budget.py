import numpy as np
from numba import jit, njit, prange


@njit
def compute_fspl(range_m, fc_Hz):
    fspl_dB = 20 * np.log10(range_m) + 20 * np.log10(fc_Hz) - 147.55

    return fspl_dB


def compute_passes_fspl(range_m_list, fc_Hz):
    return [compute_fspl(range_m_list[i], fc_Hz) for i in range(len(range_m_list))]


@njit(parallel=True)
def compute_snr(fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz):
    kB_dB = 10 * np.log10(B_Hz * 1.380649e-23)  # k*B in dB
    SNR_dB = Ptx_dBm - 30 + Gtx_dBi - fspl_dB + GT_dBK - kB_dB  # Compute SNR

    return SNR_dB

@njit(parallel=True)
def compute_throughput_from_margin(tof_s, margin_dB, B_Hz, alpha, eta_bitsym):
    """
    Computes link time and throughput where margin is positive

    :param ndarray tof_s: array with time-of-flight in seconds
    :param ndarray margin_dB: array with margin at time-of-flight in dB
    :param float B_Hz: channel bandwidth in Hz
    :param float alpha: roll-off factor
    :param float eta_bitsym: spectral efficiency in bits/Hz
    :return: link_time, throughput_bits
    """

    positive_margin = margin_dB >= 0.0

    dt = np.diff(tof_s)  # Deltas between each time step in s
    link_time_s = np.sum(dt * positive_margin[1:])  # Total time the link is established in s

    Rs_syms = B_Hz / (1 + alpha)  # Symbol rate in symbols per second
    Rb_bits = Rs_syms * eta_bitsym  # Data rate in bits per second

    throughput_bits = link_time_s * Rb_bits  # Throughput in bits per second

    return link_time_s, throughput_bits


@njit
def compute_throughput(tof_s, fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz, alpha, EsN0_req_dB, eta_bitsym, margin_dB):
    SNR_dB = compute_snr(fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz)  # SNR at the reciever in dB
    EsN0_dB = SNR_dB - 10 * np.log10(1 / (1 + alpha))  # Es/N0 at the receiver in dB

    return compute_throughput_from_margin(tof_s, EsN0_dB - (EsN0_req_dB + margin_dB), B_Hz, alpha, eta_bitsym)

# @njit(parallel=True)
def compute_passes_throughput(tof_s_list, fspl_dB_list, Ptx_dBm_list, Gtx_dBi, GT_dBK, B_Hz,
                              alpha, EsN0_req_dB, eta_bitsym, margin_dB):
    """

    :param tof_s_list: List of with a numpy array for each pass with the time-of-flight
    :param fspl_dB_list: List of with a numpy array for each pass with the free-space-path loss
    :param Ptx_dBm_list: List with transmit powers for each pass
    :param Gtx_dBi: Narray or list of Narray with antenna gains
    :param GT_dBK: G/T for all passes
    :param B_Hz: Bandwidth for all passes
    :param alpha: Roll-off factor for all passes
    :param EsN0_req_dB: Array of required Eb/N0 for each pass
    :param eta_bitsym: Array of spectral efficiency for each pass
    :param margin_dB: Required link-margin
    :return:
    """
    linktime_s_array = np.empty(len(tof_s_list), np.float64)
    throughput_bits_array = np.empty(len(tof_s_list), np.float64)

    for i in prange(len(throughput_bits_array)):
        linktime_s_array[i], throughput_bits_array[i] = compute_throughput(tof_s_list[i], fspl_dB_list[i],
                                                                           Ptx_dBm_list[i], Gtx_dBi, GT_dBK,
                                                                           B_Hz, alpha,
                                                                           EsN0_req_dB,
                                                                           eta_bitsym, margin_dB)

    return linktime_s_array, np.sum(throughput_bits_array)

@jit(nopython=True)
def calculate_isend(SNR_dB, alpha):
    return SNR_dB - 10 * np.log10(1 - alpha)


@jit(nopython=True)
def calculate_snr(fspl_dB, Ptx_dBm, Gtx_dB, GT_dBK, bandwidth_Hz):
    kB_dB = 10 * np.log10(bandwidth_Hz * 1.380649e-23)  # k*B in dB
    SNR_dB = Ptx_dBm + Gtx_dB - fspl_dB + GT_dBK - kB_dB - 30.

    return SNR_dB
