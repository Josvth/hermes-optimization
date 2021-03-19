import numpy as np
from numba import njit

@njit
def compute_hpbw(Gtx_dBi):
    theta_hpbw_rad = np.sqrt(4 * np.pi / (10 ** (0.1 * Gtx_dBi)))  # Inverse Krauss' formula
    return theta_hpbw_rad

@njit
def compute_pass_angles(r_ab):
    xy = r_ab[:, 0] ** 2 + r_ab[:, 1] ** 2
    theta = np.arctan2(np.sqrt(xy), r_ab[:, 2])
    phi = np.arctan2(r_ab[:,1], r_ab[:,0])
    return theta, phi

def compute_passes_angles(r_ab_m_list):

    theta_rad_list = [None] * len(r_ab_m_list)
    phi_rad_list = [None] * len(r_ab_m_list)

    for i in range(len(r_ab_m_list)):
        theta_rad_list[i], phi_rad_list[i] = compute_pass_angles(r_ab_m_list[i])

    return theta_rad_list, phi_rad_list

def compute_gain_values(theta_rad, Gtx0_dBi):
    theta_hpbw_rad = compute_hpbw(Gtx0_dBi)
    Gtx_dBi = (theta_rad <= 0.5 * theta_hpbw_rad) * Gtx0_dBi
    return Gtx_dBi
