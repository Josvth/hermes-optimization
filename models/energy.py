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


@njit
def compute_passes_energy_maee(linktime_s_array, Ptx_dBm_array, eta_maee_array):
    energy_J = np.sum(compute_energy(linktime_s_array, Ptx_dBm_array, eta_maee_array))
    return energy_J

# Extended models
@njit
def modulator_power(Rb_bits):
    Pmod = 0.015 * Rb_bits + 14.967 # Modulator power as function of data rate
    return Pmod

@njit
def compute_modulator_energy(linktime_s, Rb_bits):
    energy_mod_J = modulator_power(Rb_bits) * linktime_s
    return energy_mod_J

@njit
def compute_passes_energy_extended(linktime_s_array, Ptx_dBm_array, eta_maee_array, B_Hz, alpha, eta_bitsym_array):

    energy_J = compute_passes_energy_maee(linktime_s_array, Ptx_dBm_array, eta_maee_array)

    Rs_syms = B_Hz / (1 + alpha)  # Symbol rate in symbols per second
    Rb_bits_array = Rs_syms * eta_bitsym_array  # Data rate in bits per second

    energy_J = energy_J + np.sum(compute_modulator_energy(linktime_s_array, Rb_bits_array))

    return energy_J

def _make_compute_passes_energy_extended():
    return compute_passes_energy_extended