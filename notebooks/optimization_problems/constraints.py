import numpy as np
from numba import njit


class Requirements(object):
    # Throughput
    min_throughput = 5e9
    max_throughput = -1

    # Latency
    max_latency = -1

    # Energy
    max_energy = -1

    # Pointing
    max_pointing = -1
    max_rate_rads = -1

def compute_constraints(requirements, f_throughput, f_latency=0, f_energy=0, f_pointing=0, rates_rads_array=0):

    # Throughput minimum/maximum
    g_throughput_minimum = 0
    g_throughput_maximum = 0
    if requirements.min_throughput > 0:
        g_throughput_minimum = requirements.min_throughput - f_throughput
    if requirements.max_throughput > 0:
        g_throughput_maximum = f_throughput - requirements.max_throughput

    # Maximum latency
    g_latency_maximum = 0
    if requirements.max_latency > 0:
        g_latency_maximum = f_latency - requirements.max_latency

    # Maximum energy
    g_energy_maximum = 0
    if requirements.max_energy > 0:
        g_energy_maximum = f_energy - requirements.max_energy

    # Maximum pointing
    g_pointing_maximum = 0
    if requirements.max_pointing > 0:
        g_pointing_maximum = f_pointing - requirements.max_pointing

    # Rotational rates
    g_rates_maximum = 0
    if requirements.max_rate_rads > 0:
        g_rates_maximum = rates_rads_array - requirements.max_rate_rads

    g_constraints = np.append([g_throughput_minimum, g_throughput_maximum, g_latency_maximum,
                               g_energy_maximum, g_pointing_maximum], g_rates_maximum)
    return g_constraints
