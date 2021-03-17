from numba import njit, prange
import numpy as np
from numba.typed import List


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
    link_time = np.sum(dt * positive_margin[1:])  # Total time the link is established in s

    Rs_syms = B_Hz / (1 + alpha)  # Symbol rate in symbols per second
    Rb_bits = Rs_syms * eta_bitsym  # Data rate in bits per second

    throughput_bits = link_time * Rb_bits  # Throughput in bits per second

    return link_time, throughput_bits

@njit
def compute_throughput(tof_s, fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz, alpha, EsN0_req_dB, eta_bitsym, margin_dB):
    SNR_dB = compute_snr(fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz)  # SNR at the reciever in dB
    EsN0_dB = SNR_dB - 10 * np.log10(1 / (1 + alpha))  # Es/N0 at the receiver in dB

    return compute_throughput_from_margin(tof_s, EsN0_dB - (EsN0_req_dB + margin_dB), B_Hz, alpha, eta_bitsym)

@njit
def compute_throughput_max_vcm(tof_s, fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz, alpha,
                               EsN0_req_dB_array, eta_bitsym_array, min_margin_dB):
    SNR_dB = compute_snr(fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz)  # SNR at the reciever in dB
    EsN0_dB = SNR_dB - 10 * np.log10(1 / (1 + alpha))  # Es/N0 at the receiver in dB

    # Selection of modcod with smallest positive margin
    margins_dB = np.min(EsN0_dB) - (EsN0_req_dB_array + min_margin_dB)  # Array with minimum margins for each modcod

    margins_dB = np.maximum(margins_dB, 0)  # Make sure the margin is positive

    modcod_sel = np.argmin(margins_dB)  # Select modcod with lowest margin

    # Compute throughput with selected modcod
    margin_dB = EsN0_dB - (EsN0_req_dB_array[modcod_sel] + min_margin_dB)

    return compute_throughput_from_margin(tof_s, margin_dB, B_Hz, alpha, eta_bitsym_array[modcod_sel]), modcod_sel


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
    linktime_s_array = np.zeros(len(tof_s_list))
    throughput_bits_array = np.zeros(len(tof_s_list))

    for i in prange(len(throughput_bits_array)):
        linktime_s_array[i], throughput_bits_array[i] = compute_throughput(tof_s_list[i], fspl_dB_list[i],
                                                                           Ptx_dBm_list[i], Gtx_dBi, GT_dBK,
                                                                           B_Hz, alpha,
                                                                           EsN0_req_dB,
                                                                           eta_bitsym, margin_dB)

    return linktime_s_array, np.sum(throughput_bits_array)


# @njit(parallel=True)
def compute_passes_throughput_opt_vcm(tof_s_list, fspl_dB_list, Ptx_dBm_array, Gtx_dBi, GT_dBK, B_Hz,
                                      alpha, EsN0_req_dB_array, eta_bitsym_array, min_margin_dB):
    linktime_s_array = np.zeros(len(tof_s_list))
    throughput_bits_array = np.zeros(len(tof_s_list))
    vcm_array = np.zeros(len(tof_s_list), dtype=np.int64)

    for i in prange(len(throughput_bits_array)):
        linktime_s_array[i], throughput_bits_array[i], vcm_array[i] = compute_throughput_max_vcm(
            tof_s_list[i], fspl_dB_list[i],
            Ptx_dBm_array[i], Gtx_dBi, GT_dBK,
            B_Hz, alpha,
            EsN0_req_dB_array,
            eta_bitsym_array, min_margin_dB)

    return linktime_s_array, np.sum(throughput_bits_array), vcm_array

def compute_passes_throughput_multi_carrier(tof_s_list, fspl_dB_list, Ptx_dBm_array, Gtx_dBi, GT_dBK, B_Hz,
                                      alpha, EsN0_req_dB_2darray, eta_bitsym_2darray, min_margin_dB):
    # Determine number of sub-carriers
    if B_Hz <= 100e6:
        carriers = 1
    if B_Hz <= 200e6:
        carriers = 2
    if B_Hz <= 300e6:
        carriers = 3

    EsN0_req_dB_array = EsN0_req_dB_2darray[carriers, :]
    eta_bitsym_array = eta_bitsym_2darray[carriers, :]

    return compute_passes_throughput_opt_vcm(tof_s_list,fspl_dB_list,Ptx_dBm_array, Gtx_dBi, GT_dBK, B_Hz,
                                             alpha, EsN0_req_dB_array, eta_bitsym_array, min_margin_dB)

## Visbility functions
@njit
def compute_hpbw(Gtx_dBi):
    theta_hpbw_rad = np.sqrt(4 * np.pi / (10 ** (0.1 * Gtx_dBi)))  # Inverse Krauss' formula
    return theta_hpbw_rad


@njit
def compute_elevation_angles(r_ab):
    xy = r_ab[:, 0] ** 2 + r_ab[:, 1] ** 2
    theta = np.arctan2(np.sqrt(xy), r_ab[:, 2])
    return theta


def compute_passes_elevation_angles(r_ab_m_list):
    return [compute_elevation_angles(r_ab_m_list[i]) for i in range(len(r_ab_m_list))]


def compute_gain_values(theta_rad, Gtx0_dBi):
    theta_hpbw_rad = compute_hpbw(Gtx0_dBi)
    Gtx_dBi = (theta_rad <= 0.5 * theta_hpbw_rad) * Gtx0_dBi
    return Gtx_dBi


# @njit(parallel=True)
def compute_passes_throughput_visibility(tof_s_list, theta_rad_list, fspl_dB_list, Ptx_dBm_array, Gtx0_dBi, GT_dBK,
                                         B_Hz_array,
                                         alpha_array, EsN0_req_dB_array, eta_bitsym_array, margin_dB):
    Gtx_dBi_list = List(tof_s_list)

    for i in prange(len(Gtx_dBi_list)):
        Gtx_dBi_list[i] = compute_gain_values(theta_rad_list[i], Gtx0_dBi[i])

    for i, l in enumerate([tof_s_list, theta_rad_list, fspl_dB_list, Gtx_dBi_list]):
        if l._list_type is None:
            print(['tof_s_list', 'theta_rad_list', 'fspl_dB_list', 'Gtx_dBi_list'][i])

    return compute_passes_throughput(tof_s_list, fspl_dB_list, Ptx_dBm_array, Gtx_dBi_list, GT_dBK, B_Hz_array,
                                     alpha_array, EsN0_req_dB_array, eta_bitsym_array, margin_dB)


## Energy functions
@njit
def compute_energy(linktime_s, Ptx_dBm, eta):
    energy_J = (1 / eta) * 10 ** ((Ptx_dBm - 30) / 10) * linktime_s  # Energy in J
    return energy_J


# @njit(parallel=True)
def compute_passes_energy_simplified(tof_s_list, Ptx_dBm_list):
    energy_J_list = np.zeros(len(tof_s_list))

    for i in prange(len(energy_J_list)):
        dt = np.diff(tof_s_list[i])
        linktime_s = np.sum(dt)
        energy_J_list[i] = compute_energy(linktime_s, Ptx_dBm_list[i], 1.0)

    return np.sum(energy_J_list)


def compute_passes_energy_simplified_visibility(linktime_s_list, Ptx_dBm_list):
    energy_J_list = np.zeros(len(linktime_s_list))

    for i in prange(len(energy_J_list)):
        energy_J_list[i] = compute_energy(linktime_s_list[i], Ptx_dBm_list[i], 1.0)

    return np.sum(energy_J_list)


# @njit
def compute_passes_energy_maee(linktime_s_list, Ptx_dBm_array, eta_maee_list):
    energy_J_list = np.zeros(len(linktime_s_list))

    for i in prange(len(energy_J_list)):
        energy_J_list[i] = compute_energy(linktime_s_list[i], Ptx_dBm_array[i], eta_maee_list[i])

    return np.sum(energy_J_list)
