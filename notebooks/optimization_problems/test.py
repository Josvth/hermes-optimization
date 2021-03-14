from itertools import compress

from numba.typed import List

from models.models import compute_passes_throughput_visibility, compute_passes_energy_maee
from notebooks.optimization_problems.design_vector import explode_design_vector


import numpy as np

def my_eval(param):
    x = param['x']
    N_passes = param['N_passes']
    x_indices = param['x_indices']
    O_matrix = param['O_matrix']
    tof_s_list = param['tof_s_list']
    fspl_dB_list = param['fspl_dB_list']
    theta_rad_list = param['theta_rad_list']
    sys_param = param['sys_param']
    reqs = param['reqs']

    # Explode design vector
    design_vector = explode_design_vector(x, N_passes, x_indices)

    # Calculate overlap constraint
    overlap = O_matrix * design_vector['pass']
    overlap = overlap[np.tril(np.ones((N_passes, N_passes)), -1) == 1]
    g_overlap = overlap.flatten()  # Non-overlapping constraint

    # Compute throughput of selected passes
    sel_pass = design_vector['pass'] > 0
    tof_s_list = List(compress(tof_s_list, sel_pass))  # List of tofs of the selected passes
    fspl_dB_list = List(compress(fspl_dB_list, sel_pass))
    Ptx_dBm_list = List(compress(design_vector['power'], sel_pass))
    Gtx_dBi_array = np.array(list(compress(design_vector['antenna'].repeat(N_passes), sel_pass)))
    B_Hz_array = np.array(
        list(compress(map(sys_param.B_Hz_list.__getitem__, design_vector['bandwidth']), sel_pass)))
    alpha_array = np.array(list(
        compress(map(sys_param.alpha_list.__getitem__, design_vector['rolloff'].repeat(self.N_passes)),
                 sel_pass)))
    EsN0_req_dB_array = np.array(
        list(compress(map(sys_param.EsN0_req_dB_list.__getitem__, design_vector['modcod']), sel_pass)))
    eta_bitsym_array = np.array(
        list(compress(map(sys_param.eta_bitsym_list.__getitem__, design_vector['modcod']), sel_pass)))
    eta_maee_array = np.array(
        list(compress(map(sys_param.eta_maee_list.__getitem__, design_vector['modcod']), sel_pass)))

    theta_rad_list = List(compress(theta_rad_list, sel_pass))

    linktime_s_list, f_throughput = compute_passes_throughput_visibility(tof_s_list, theta_rad_list,
                                                                         fspl_dB_list,
                                                                         Ptx_dBm_list, Gtx_dBi_array,
                                                                         self.sys_param.GT_dBK, B_Hz_array,
                                                                         alpha_array, EsN0_req_dB_array,
                                                                         eta_bitsym_array,
                                                                         self.sys_param.margin_dB)

    f_energy = compute_passes_energy_maee(linktime_s_list, Ptx_dBm_list, eta_maee_array)

    # Compute minimum throughput constraint
    g_minimum = reqs.min_throughput - f_throughput

    out = {}

    out["F"] = [-1 * f_throughput, f_energy]
    out["G"] = np.concatenate([g_overlap, np.array([g_minimum])])

    return out