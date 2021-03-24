import numpy as np
from numba import njit, prange

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


@njit(parallel=True)
def compute_passes_energy_maee(linktime_s_array, Ptx_dBm_array, eta_maee_array):
    energy_J = 0.0

    for i in prange(linktime_s_array.shape[0]):
        energy_J += compute_energy(linktime_s_array[i], Ptx_dBm_array[i], eta_maee_array[i])

    return energy_J
