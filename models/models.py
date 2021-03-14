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
def compute_throughput(tof_s, fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz, alpha, EsN0_req_dB, eta_bitsym, margin_dB):
    SNR_dB = compute_snr(fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz)  # SNR at the reciever in dB
    EsN0_dB = SNR_dB - 10 * np.log10(1 / (1 + alpha))  # Es/N0 at the receiver in dB

    positive_margin = EsN0_dB >= EsN0_req_dB + margin_dB  # True values where there is a positive link margin

    dt = np.diff(tof_s)  # Deltas between each time step in s
    link_time = np.sum(dt * positive_margin[1:])  # Total time the link is established in s

    Rs_syms = B_Hz / (1 + alpha)  # Symbol rate in symbols per second
    Rb_bits = Rs_syms * eta_bitsym  # Data rate in bits per second

    throughput_bits = link_time * Rb_bits  # Throughput in bits per second

    return link_time, throughput_bits


@njit
def compute_throughput_min_power(tof_s, fspl_dB, Ptx_dBm_max, Gtx_dBi, GT_dBK, B_Hz, alpha, EsN0_req_dB, eta_bitsym,
                                 margin_dB):
    SNR_dB = compute_snr(fspl_dB, 0.0, Gtx_dBi, GT_dBK, B_Hz)  # SNR at the reciever in dB
    EsN0_dB = SNR_dB - 10 * np.log10(1 / (1 + alpha))  # Es/N0 at the receiver in dB

    margin_no_Ptx = EsN0_dB - (EsN0_req_dB + margin_dB)  # Margin before adding transmitter power

    req_Ptx_dBm = -1 * np.minimum(margin_no_Ptx, 0)  # Required Ptx to close the link

    Ptx_dBm = np.minimum(np.max(req_Ptx_dBm), Ptx_dBm_max)  # The used transmission power

    margin = margin_no_Ptx + Ptx_dBm  # Margins after adding transmitter power

    positive_margin = margin >= 0  # True values where there is a positive link margin

    dt = np.diff(tof_s)  # Deltas between each time step in s
    link_time = np.sum(dt * positive_margin[1:])  # Total time the link is established in s

    Rs_syms = B_Hz / (1 + alpha)  # Symbol rate in symbols per second
    Rb_bits = Rs_syms * eta_bitsym  # Data rate in bits per second

    throughput_bits = link_time * Rb_bits  # Throughput in bits per second

    return link_time, throughput_bits, Ptx_dBm


@njit
def compute_throughput_max_vcm(tof_s, fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz, alpha, EsN0_req_dB, eta_bitsym,
                               min_margin_dB):
    SNR_dB = compute_snr(fspl_dB, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz)  # SNR at the reciever in dB
    EsN0_dB = SNR_dB - 10 * np.log10(1 / (1 + alpha))  # Es/N0 at the receiver in dB

    # Selection of modcod with smallest positive margin
    margins_dB = np.min(EsN0_dB) - (EsN0_req_dB + min_margin_dB)  # Array with minimum margins for each modcod

    margins_dB = np.maximum(margins_dB, 0)  # Make sure the margin is positive

    modcod_sel = np.argmin(margins_dB)  # Select modcod with lowest margin

    # Compute throughput with selected modcod
    margin_dB = EsN0_dB - (EsN0_req_dB[modcod_sel] + min_margin_dB)

    positive_margin = margin_dB >= 0  # True values where there is a positive link margin

    dt = np.diff(tof_s)  # Deltas between each time step in s
    link_time = np.sum(dt * positive_margin[1:])  # Total time the link is established in s

    Rs_syms = B_Hz / (1 + alpha)  # Symbol rate in symbols per second
    Rb_bits = Rs_syms * eta_bitsym[modcod_sel]  # Data rate in bits per second

    throughput_bits = link_time * Rb_bits  # Throughput in bits per second

    return link_time, throughput_bits, modcod_sel


@njit(parallel=True)
def compute_passes_throughput(tof_s_list, fspl_dB_list, Ptx_dBm, Gtx_dBi_list, GT_dBK, B_Hz,
                              alpha, EsN0_req_dB_array, eta_bitsym_array, margin_dB):
    """

    :param tof_s_list: List of with a numpy array for each pass with the time-of-flight
    :param fspl_dB_list: List of with a numpy array for each pass with the free-space-path loss
    :param Ptx_dBm: Transmit power for all passes
    :param Gtx_dBi_list: Narray or list of Narray with antenna gains
    :param GT_dBK: G/T for all passes
    :param B_Hz: Bandwidth for all passes
    :param alpha: Roll-off factor for all passes
    :param EsN0_req_dB_array: Array of required Eb/N0 for each pass
    :param eta_bitsym_array: Array of spectral efficiency for each pass
    :param margin_dB: Required link-margin
    :return:
    """
    linktime_s_array = np.zeros(len(tof_s_list))
    throughput_bits_array = np.zeros(len(tof_s_list))

    for i in prange(len(throughput_bits_array)):
        linktime_s_array[i], throughput_bits_array[i] = compute_throughput(tof_s_list[i], fspl_dB_list[i],
                                                                           Ptx_dBm, Gtx_dBi_list[i], GT_dBK,
                                                                           B_Hz, alpha,
                                                                           EsN0_req_dB_array[i],
                                                                           eta_bitsym_array[i], margin_dB)

    return linktime_s_array, np.sum(throughput_bits_array)


@njit(parallel=True)
def compute_passes_throughput_min_power(tof_s_list, fspl_dB_list, Ptx_dBm_max, Gtx_dBi, GT_dBK, B_Hz,
                                        alpha, EsN0_req_dB_array, eta_bitsym_array, margin_dB):
    linktime_s_array = np.zeros(len(tof_s_list))
    throughput_bits_array = np.zeros(len(tof_s_list))
    Ptx_dBm_array = np.zeros(len(tof_s_list))

    for i in prange(len(throughput_bits_array)):
        linktime_s_array[i], throughput_bits_array[i], Ptx_dBm_array[i] = compute_throughput_min_power(
            tof_s_list[i], fspl_dB_list[i],
            Ptx_dBm_max, Gtx_dBi, GT_dBK,
            B_Hz, alpha,
            EsN0_req_dB_array[i],
            eta_bitsym_array[i], margin_dB)

    return linktime_s_array, np.sum(throughput_bits_array), Ptx_dBm_array


@njit(parallel=True)
def compute_passes_throughput_max_vcm(tof_s_list, fspl_dB_list, Ptx_dBm, Gtx_dBi, GT_dBK, B_Hz,
                                      alpha, max_vcm, EsN0_req_dB_array, eta_bitsym_array, min_margin_dB):
    linktime_s_array = np.zeros(len(tof_s_list))
    throughput_bits_array = np.zeros(len(tof_s_list))
    vcm_array = np.zeros(len(tof_s_list), dtype=np.int64)

    for i in prange(len(throughput_bits_array)):
        linktime_s_array[i], throughput_bits_array[i], vcm_array[i] = compute_throughput_max_vcm(
            tof_s_list[i], fspl_dB_list[i],
            Ptx_dBm, Gtx_dBi, GT_dBK,
            B_Hz, alpha,
            EsN0_req_dB_array[:max_vcm + 1],
            eta_bitsym_array[:max_vcm + 1], min_margin_dB)

    return linktime_s_array, np.sum(throughput_bits_array), vcm_array


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


@njit
def compute_passes_energy_maee(linktime_s_list, Ptx_dBm_array, eta_maee_list):
    energy_J_list = np.zeros(len(linktime_s_list))

    for i in prange(len(energy_J_list)):
        energy_J_list[i] = compute_energy(linktime_s_list[i], Ptx_dBm_array[i], eta_maee_list[i])

    return np.sum(energy_J_list)
