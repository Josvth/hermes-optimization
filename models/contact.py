from numba import njit, prange
import numpy as np

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
