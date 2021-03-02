from numba import njit, prange
import numpy as np

@njit
def compute_overlap_matrix(start_tofs, end_tofs):
    N = len(start_tofs)

    B = start_tofs.repeat(N).reshape((-1, N))
    E = end_tofs.repeat(N).reshape((-1, N)).T

    O = np.tril(E - B, -1)

    return O


@njit
def compute_contact_time(pass_select, begin_tofs, end_tofs):
    contact_time = np.sum(end_tofs[pass_select == 1] - begin_tofs[pass_select == 1])

    return contact_time


## Link budget stuff
@njit
def compute_fspl(range_m, fc_Hz):
    fspl_dB = 20 * np.log10(range_m) + 20 * np.log10(fc_Hz) - 147.55

    return fspl_dB


def compute_passes_fspl(range_m_list, fc_Hz):
    return [compute_fspl(range_m_list[i], fc_Hz) for i in range(len(range_m_list))]


@njit
def compute_snr(fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz):
    kB_dB = 10 * np.log10(B_Hz * 1.380649e-23)  # k*B in dB
    SNR_dB = Ptx_dBm - 30 + Gtx_dBi - fspl_dB + GT_dBK - kB_dB  # Compute SNR

    return SNR_dB


## Throughput functions
@njit
def compute_throughput(tof_s, fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz, alpha, EsN0_req_dB, eta_bitsym):
    SNR_dB = compute_snr(fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz)  # SNR at the reciever in dB
    EsN0_dB = SNR_dB * (1 + alpha)  # Es/N0 at the receiver in dB

    positive_margin = EsN0_dB >= EsN0_req_dB  # True values where there is a positive link margin

    dt = np.diff(tof_s)  # Deltas between each time step in s
    link_time = np.sum(dt * positive_margin[1:])  # Total time the link is established in s

    Rs_syms = B_Hz / (1 + alpha)  # Symbol rate in symbols per second
    Rb_bits = Rs_syms * eta_bitsym  # Data rate in bits per second

    throughput_bits = link_time * Rb_bits  # Throughput in bits per second

    return throughput_bits


#@njit(parallel=True)
def compute_passes_throughput(tof_s_list, fspl_dB_list, Ptx_dBm_list, Gtx_dBi, GT_dBK, B_Hz_list,
                              alpha_list, EsN0_req_dB_list, eta_bitsym_list):
    throughput_bits_list = np.zeros(len(tof_s_list))

    for i in prange(len(throughput_bits_list)):
        throughput_bits_list[i] = compute_throughput(tof_s_list[i], fspl_dB_list[i], Ptx_dBm_list[i], Gtx_dBi, GT_dBK,
                                                     B_Hz_list[i], alpha_list[i], EsN0_req_dB_list[i],
                                                     eta_bitsym_list[i])

    return np.sum(throughput_bits_list)


## Energy functions
@njit
def compute_energy_simplified(tof_s, Ptx_dBm):
    dt = np.diff(tof_s)
    energy_J = 10 ** ((Ptx_dBm - 30) / 10) * np.sum(dt)  # Energy in J

    return energy_J


#@njit(parallel=True)
def compute_passes_energy_simplified(tof_s_list, Ptx_dBm_list):
    energy_J_list = np.zeros(len(tof_s_list))

    for i in prange(len(energy_J_list)):
        energy_J_list[i] = compute_energy_simplified(tof_s_list[i], Ptx_dBm_list[i])

    return np.sum(energy_J_list)
