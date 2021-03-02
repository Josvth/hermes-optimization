from itertools import compress

import numpy as np
from numba.typed import List
from pymoo.model.problem import Problem

from hermes.postprocessing import generate_passes_df_reduced, generate_grouped_passed_df, generate_pass_range_list, \
    generate_pass_tof_list

from models.models import compute_overlap_matrix, compute_contact_time, compute_passes_fspl, compute_passes_throughput, compute_passes_energy_simplified
from notebooks.optimization_problems.design_vector import design_vector_indices, design_vector_bounds, \
    explode_design_vector


class LinkBudgetProblem(Problem):

    def __init__(self, instances_df, system_parameters, single_power=True, *args, **kwargs):
        #self.instances_df = instances_df
        self.sys_param = system_parameters
        self.N_passes = int(np.max(instances_df.index.get_level_values('p').unique().values))
        self.single_power = single_power

        grouped_passes_df = generate_grouped_passed_df(instances_df)

        # Setup variables needed for optimization
        self.b_tofs = grouped_passes_df.first().tof.values
        self.e_tofs = grouped_passes_df.last().tof.values

        # Compute overlap matrix
        self.O_matrix = compute_overlap_matrix(self.b_tofs, self.e_tofs)

        # Compute free space path loss
        self.tof_s_list = generate_pass_tof_list(grouped_passes_df)
        self.range_m_list = generate_pass_range_list(grouped_passes_df)
        self.fspl_dB_list = compute_passes_fspl(self.range_m_list, self.sys_param.fc_Hz)

        # Get design vector indices and bounds
        self.x_length, self.x_indices = design_vector_indices(self.N_passes)
        self.xl, self.xu = design_vector_bounds(self.x_length, self.x_indices, self.sys_param)

        super().__init__(n_var=self.x_length,
                         n_obj=2,
                         n_constr=int((self.N_passes * (self.N_passes + 1)) / 2) + 1,
                         xl=self.xl,
                         xu=self.xu,
                         elementwise_evaluation=True,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        # Explode design vector
        design_vector = explode_design_vector(x, self.N_passes, self.x_indices)

        # Calculate overlap constraint
        overlap = self.O_matrix * design_vector['pass']
        overlap = overlap[np.tril(np.ones((self.N_passes, self.N_passes)), -1) == 1]
        g_overlap = overlap.flatten()  # Non-overlapping constraint

        # Compute throughput of selected passes
        sel_pass = design_vector['pass'] > 0
        tof_s_list = List(compress(self.tof_s_list, sel_pass))  # List of tofs of the selected passes
        fspl_dB_list = List(
            compress(self.fspl_dB_list, sel_pass))  # List of free-space path losses of the selected passes
        Ptx_dBm_list = design_vector['power']
        Gtx_dBi = design_vector['antenna'][0]
        B_Hz_list = List(map(self.sys_param.B_Hz_list.__getitem__, design_vector['bandwidth']))
        alpha_list = List(map(self.sys_param.alpha_list.__getitem__, design_vector['rolloff']))
        EsN0_req_dB_list = List(map(self.sys_param.EsN0_req_dB_list.__getitem__, design_vector['modcod']))
        eta_bitsym_list = List(map(self.sys_param.eta_bitsym_list.__getitem__, design_vector['modcod']))

        # Option to simplify to one power selection for all passes
        if self.single_power:
            Ptx_dBm_list = [Ptx_dBm_list[0]] * len(Ptx_dBm_list)

        f_throughput = compute_passes_throughput(tof_s_list, fspl_dB_list,
                                                 Ptx_dBm_list, Gtx_dBi, self.sys_param.GT_dBK, B_Hz_list,
                                                 alpha_list, EsN0_req_dB_list, eta_bitsym_list)
        f_energy = compute_passes_energy_simplified(tof_s_list, Ptx_dBm_list)

        # Compute minimum throughput constraint
        g_minimum = (f_throughput == 0) * 1.0

        out["F"] = [-1 * f_throughput, f_energy]
        out["G"] = np.concatenate([g_overlap, np.array([g_minimum])])
