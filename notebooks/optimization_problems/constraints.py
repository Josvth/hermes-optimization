import numpy as np


class Requirements(object):
    # Throughput
    min_throughput = 5e9
    max_throughput = -1

    # Energy
    max_energy = -1

    # Pointing rate
    max_rate_rads = 1e18


def apply_constraints(requirements, f_throughput, f_energy=0, f_pointing=0, theta_rate_rads_max=0, phi_rate_rads_max=0):
    # Throughput minimum/maximum
    g_throughput_minimum = 0
    g_throughput_maximum = 0
    if requirements.min_throughput > 0:
        g_throughput_minimum = requirements.min_throughput - f_throughput
    if requirements.max_throughput > 0:
        g_throughput_maximum = f_throughput - requirements.max_throughput

    g_energy_maximum = 0
    if requirements.max_energy > 0:
        g_energy_maximum = f_energy - requirements.max_throughput

    # Rotational rates
    # g_theta_rate_max = 0
    # g_phi_rate_max = 0
    # if requirements.max_rate_rads > 0:
    #     g_theta_rate_max = theta_rate_rads_max - requirements.max_rate_rads
    #     g_phi_rate_max = phi_rate_rads_max - requirements.max_rate_rads

    return np.array([g_throughput_minimum, g_throughput_maximum, g_energy_maximum])
