import numpy as np
from numba import jit


@jit(nopython=True)
def calculate_isend(SNR_dB, alpha):
    return SNR_dB - 10 * np.log10(1 - alpha)


@jit(nopython=True)
def calculate_snr(fspl_dB, Ptx_dBm, Gtx_dB, GT_dBK, bandwidth_Hz):
    kB_dB = 10 * np.log10(bandwidth_Hz * 1.380649e-23)  # k*B in dB
    SNR_dB = Ptx_dBm + Gtx_dB - fspl_dB + GT_dBK - kB_dB - 30.

    return SNR_dB
