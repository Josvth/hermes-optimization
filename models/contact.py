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

@njit
def compute_overlap(x_pass, O_matrix):
    N_passes = len(x_pass)
    x_pass_tile = x_pass.repeat(N_passes).reshape((-1, N_passes))
    overlap = O_matrix * x_pass_tile * x_pass_tile.T
    N_overlap = np.sum(overlap > 0)

    return N_overlap

@njit
def down_select_passes(x_pass, O_matrix):

    M = ~(O_matrix > 0)

    for i in range(len(x_pass)):
        if x_pass[i]:               # If the pass is selected
            mask = M[:,i]
            x_pass = x_pass & mask  # Mask all following overlapping passes

    return x_pass