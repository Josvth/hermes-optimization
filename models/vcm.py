import numpy as np
from numba import prange, njit
from numba.typed import List

from models import link_budget


@njit(parallel=True)
def compute_throughput_vcm(tof_s, fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz, alpha,
                           EsN0_req_dB_array, eta_bitsym_array, min_margin_dB):
    SNR_dB = link_budget.compute_snr(fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz)  # SNR at the reciever in dB
    EsN0_dB = SNR_dB - 10 * np.log10(1 / (1 + alpha))  # Es/N0 at the receiver in dB

    # Selection of modcod with smallest positive margin
    margins_dB = np.min(EsN0_dB) - (EsN0_req_dB_array + min_margin_dB)  # Array with minimum margins for each modcod

    margins_dB = np.maximum(margins_dB, 0)  # Make sure the margin is positive

    modcod_sel = int(np.argmin(margins_dB))  # Select modcod with lowest margin

    # Compute throughput with selected modcod
    margin_dB = EsN0_dB - (EsN0_req_dB_array[modcod_sel] + min_margin_dB)

    link_time_s, throughput_bits = link_budget.compute_throughput_from_margin(tof_s, margin_dB, B_Hz, alpha,
                                                                              eta_bitsym_array[modcod_sel])

    b_s = np.NaN
    e_s = np.NaN

    if link_time_s > 0.0:
        b_s = tof_s[np.nonzero(margin_dB >= 0)][0]
        e_s = tof_s[np.nonzero(margin_dB >= 0)][-1]

    return b_s, e_s, margin_dB, link_time_s, throughput_bits, modcod_sel


@njit(parallel=True)
def compute_passes_throughput(pass_inds, tof_s_list, fspl_dB_list, Ptx_dBm_array, Gtx_dBi, GT_dBK, B_Hz,
                              alpha, EsN0_req_dB_array, eta_bitsym_array, min_margin_dB):
    """
    :param numpy.ndarray pass_inds: Array of pass indices to consider in the computation
    :param numba.typed.List tof_s_list: List of ndarrays with time-of-flights per pass
    :param numba.typed.List fspl_dB_list: List of ndarrays with free-space-path-loss per pass
    :param numpy.ndarray Ptx_dBm_array: Array with transmitted power for SELECTED passes (ToDo: change this to be down selected by the pass_inds in this function)
    :param float Gtx_dBi: Antenna gain in dBi
    :param float GT_dBK: Receiver G/T in dBK
    :param float B_Hz: Bandwidth in Hz
    :param float alpha: Roll-off factor
    :param numpy.ndarray EsN0_req_dB_array: Array with required EbN0 for selectable MODCODs
    :param numpy.ndarray eta_bitsym_array: Array with spectral efficiencies for selectable MODCODs
    :param float min_margin_dB: Minimum margin in dB to apply to the link-budget
    :return: tuple(numpy.ndarray, numpy.ndarray, np.ndarray, float, numpy.ndarray)
    """
    b_s_array = np.empty(len(pass_inds), np.float64)
    e_s_array = np.empty(len(pass_inds), np.float64)

    linktime_s_array = np.empty(len(pass_inds), np.float64)
    throughput_bits_array = np.empty(len(pass_inds), np.float64)
    vcm_array = np.empty(len(pass_inds), np.int32)

    for i in prange(len(pass_inds)):
        p = pass_inds[i]

        b_s_array[i], e_s_array[i], _, linktime_s_array[i], throughput_bits_array[i], vcm_array[
            i] = compute_throughput_vcm(
            tof_s_list[p], fspl_dB_list[p],
            Ptx_dBm_array[i], Gtx_dBi, GT_dBK,
            B_Hz, alpha,
            EsN0_req_dB_array,
            eta_bitsym_array, min_margin_dB)

    # ToDo: remove the sum around throughput for more functionality
    return b_s_array[~np.isnan(b_s_array)], e_s_array[~np.isnan(b_s_array)], linktime_s_array, np.sum(
        throughput_bits_array), vcm_array


def _make_compute_passes_throughput():
    # This function makes a compiled function that can be passed to other @njit functions
    return compute_passes_throughput
