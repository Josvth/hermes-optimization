import numpy as np
from numba import prange, njit

from models import link_budget


@njit
def compute_throughput_vcm(tof_s, fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz, alpha,
                           EsN0_req_dB_array, eta_bitsym_array, min_margin_dB):
    SNR_dB = link_budget.compute_snr(fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz)  # SNR at the reciever in dB
    EsN0_dB = SNR_dB - 10 * np.log10(1 / (1 + alpha))  # Es/N0 at the receiver in dB

    # Selection of modcod with smallest positive margin
    margins_dB = np.min(EsN0_dB) - (EsN0_req_dB_array + min_margin_dB)  # Array with minimum margins for each modcod

    margins_dB = np.maximum(margins_dB, 0)  # Make sure the margin is positive

    modcod_sel = np.argmin(margins_dB)  # Select modcod with lowest margin

    # Compute throughput with selected modcod
    margin_dB = EsN0_dB - (EsN0_req_dB_array[modcod_sel] + min_margin_dB)

    link_time_s, throughput_bits = link_budget.compute_throughput_from_margin(tof_s, margin_dB, B_Hz, alpha,
                                                                              eta_bitsym_array[modcod_sel])

    return link_time_s, throughput_bits, modcod_sel


#@njit(parallel=True)
def compute_passes_throughput(tof_s_list, fspl_dB_list, Ptx_dBm_array, Gtx_dBi, GT_dBK, B_Hz,
                              alpha, EsN0_req_dB_array, eta_bitsym_array, min_margin_dB):
    linktime_s_array = np.empty(len(tof_s_list))
    throughput_bits_array = np.empty(len(tof_s_list))
    vcm_array = np.empty(len(tof_s_list))

    for i in prange(len(throughput_bits_array)):
        linktime_s_array[i], throughput_bits_array[i], vcm_array[i] = compute_throughput_vcm(
            tof_s_list[i], fspl_dB_list[i],
            Ptx_dBm_array[i], Gtx_dBi, GT_dBK,
            B_Hz, alpha,
            EsN0_req_dB_array,
            eta_bitsym_array, min_margin_dB)

    return linktime_s_array, np.sum(throughput_bits_array), vcm_array
