from itertools import compress

import numpy as np
from numba.typed import List
from pymoo.model.problem import Problem

from hermes.postprocessing import generate_passes_df_reduced, generate_grouped_passed_df, generate_pass_range_list, \
    generate_pass_tof_list

from models import contact, link_budget, energy

from notebooks.optimization_problems.design_vector import design_vector_indices, design_vector_bounds, \
    explode_design_vector


class LinkBudgetProblem(Problem):

    def __init__(self, instances_df, system_parameters, single_power=True, *args, **kwargs):
        # self.instances_df = instances_df
        self.sys_param = system_parameters
        self.N_passes = int(np.max(instances_df.index.get_level_values('p').unique().values))
        self.single_power = single_power

        grouped_passes_df = generate_grouped_passed_df(instances_df)

        # Setup variables needed for optimization
        self.b_tofs = grouped_passes_df.first().tof.values
        self.e_tofs = grouped_passes_df.last().tof.values

        # Compute overlap matrix
        self.O_matrix = contact.compute_overlap_matrix(self.b_tofs, self.e_tofs)

        # Compute free space path loss
        self.tof_s_list = generate_pass_tof_list(grouped_passes_df)
        range_m_list = generate_pass_range_list(grouped_passes_df)
        self.fspl_dB_list = link_budget.compute_passes_fspl(range_m_list, self.sys_param.fc_Hz)

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
        N_sel_pass = np.sum(sel_pass)

        # Defaults
        f_throughput = 0
        f_energy = 0

        # Compute throughput of selected passes
        sel_pass = design_vector['pass'] > 0
        if np.sum(sel_pass) > 0:

            tof_s_list = List(compress(self.tof_s_list, sel_pass))  # List of tofs of the selected passes
            fspl_dB_list = List(compress(self.fspl_dB_list, sel_pass))
            Ptx_dBm_array = design_vector['power'][sel_pass]
            Gtx_dBi = design_vector['antenna'][0]
            B_Hz = self.sys_param.B_Hz_array[design_vector['bandwidth'][0]]
            alpha = self.sys_param.alpha_array[design_vector['rolloff'][0]]
            EsN0_req_dB_array = self.sys_param.EsN0_req_dB_array[0]
            eta_bitsym_array = self.sys_param.eta_bitsym_array[0]

            if self.single_power:
                Ptx_dBm_array[:] = design_vector['power'][0]

            linktime_s_array, f_throughput = link_budget.compute_passes_throughput(tof_s_list, fspl_dB_list,
                                                                       Ptx_dBm_array, Gtx_dBi,
                                                                       self.sys_param.GT_dBK, B_Hz,
                                                                       alpha,
                                                                       EsN0_req_dB_array,
                                                                       eta_bitsym_array,

                                                                       self.sys_param.margin_dB)

            # Ptx_dBm_array = np.array([Ptx_dBm] * np.sum(sel_pass))
            f_energy = energy.compute_passes_energy_simplified(tof_s_list, Ptx_dBm_array)

        # Compute minimum throughput constraint
        g_minimum = (f_throughput == 0) * 1.0

        out["F"] = (-1 * f_throughput, f_energy)
        out["G"] = np.concatenate((g_overlap, np.array([g_minimum])))
