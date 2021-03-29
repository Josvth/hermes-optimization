from numba import njit, prange
import numpy as np
from numba.typed import List

from models.link_budget import compute_snr

## Contact model
# see models.contact

## Link budget model
# see models.link_budget

## VCM model
# see models.vcm

## Visbility model
@njit(parallel=True)
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
# see models.energy