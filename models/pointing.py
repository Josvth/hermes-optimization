from numba import prange, njit

from models import visibility
import numpy as np


@njit
def compute_antenna_pointing(theta_rad_array, Gtx_dBi):
    theta_hpbw_rad = visibility.compute_hpbw(Gtx_dBi)
    return np.maximum(theta_rad_array - 0.5 * theta_hpbw_rad, 0)


@njit
def compute_pointing(tof_s_array, theta_rad_array, phi_rad_array, Gtx_dBi):
    theta_pointing_rad_array = compute_antenna_pointing(theta_rad_array, Gtx_dBi)

    dt = np.diff(tof_s_array)
    f_pointing = np.sum((theta_pointing_rad_array[1:] > 0) * dt)

    dtheta = np.diff(theta_rad_array)
    dphi = np.diff(phi_rad_array)

    dphi[np.abs(dphi) >= np.pi / 2] = 0.  # Catches any flip-overs/gimbal lock

    theta_rate_max_rad = np.max(np.abs(dtheta / dt))
    phi_rate_max_rad = np.max(np.abs(dphi / dt))

    return f_pointing, np.maximum(theta_rate_max_rad, phi_rate_max_rad)


@njit(parallel=True)
def compute_pointing_passes(tof_s_list, theta_rad_list, phi_rad_list, Gtx_dBi):
    f_pointing_array = np.zeros(len(tof_s_list))
    rate_max_rad_array = np.zeros(len(tof_s_list))

    for i in prange(len(f_pointing_array)):
        f_pointing_array[i], rate_max_rad_array[i] = compute_pointing(
            tof_s_list[i], theta_rad_list[i], phi_rad_list[i], Gtx_dBi)

    return np.sum(f_pointing_array), rate_max_rad_array
