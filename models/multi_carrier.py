from models import vcm
from numba import njit

@njit
def get_sub_carriers(B_Hz):
    # Determine number of sub-carriers
    if B_Hz <= 100e6:
        return 1
    if B_Hz <= 200e6:
        return 2
    if B_Hz <= 300e6:
        return 3
    return 0