from itertools import compress

import numpy as np
from numba.typed import List
from pymoo.model.problem import Problem

from hermes.postprocessing import generate_grouped_passed_df, generate_pass_range_list, \
    generate_pass_tof_list, generate_pass_r_ab_list

from models import contact, link_budget, vcm, models, energy, multi_carrier, pointing, visibility, latency

from notebooks.optimization_problems.constraints import Requirements, compute_constraints
from notebooks.optimization_problems.design_vector import design_vector_indices, design_vector_bounds, \
    explode_design_vector


class FullProblem(Problem):

    def __init__(self, instances_df, system_parameters, requirements=Requirements(), *args, **kwargs):
        # self.instances_df = instances_df
        self.sys_param = system_parameters
        self.reqs = requirements

        self.N_passes = int(np.max(instances_df.index.get_level_values('p').unique().values))

        grouped_passes_df = generate_grouped_passed_df(instances_df)

        # Setup variables needed for optimization
        self.b_tofs = grouped_passes_df.first().tof.values
        self.e_tofs = grouped_passes_df.last().tof.values
        self.t_end_s = grouped_passes_df.last().tof.values[-1]

        # Compute overlap matrix
        self.O_matrix = contact.compute_overlap_matrix(self.b_tofs, self.e_tofs)

        # Compute variables needed during pass
        self.tof_s_list = generate_pass_tof_list(grouped_passes_df)

        range_m_list = generate_pass_range_list(grouped_passes_df)
        self.fspl_dB_list = link_budget.compute_passes_fspl(range_m_list, self.sys_param.fc_Hz)

        r_ab_m_list = generate_pass_r_ab_list(grouped_passes_df)
        self.theta_rad_list, self.phi_rad_list = visibility.compute_passes_angles(r_ab_m_list)

        # Get design vector indices and bounds
        self.x_length, self.x_indices = design_vector_indices(self.N_passes)
        self.xl, self.xu = design_vector_bounds(self.x_length, self.x_indices, self.sys_param)

        super().__init__(n_var=self.x_length,
                         n_obj=4,
                         n_constr=int((self.N_passes * (self.N_passes + 1)) / 2) + 5 + self.N_passes,
                         xl=self.xl,
                         xu=self.xu,
                         elementwise_evaluation=True,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        # Defaults
        f_throughput = 0  # Maximum throughput objective
        f_energy = 0  # Minimum energy objective
        f_pointing = 0  # Minimum pointing objective
        f_latency = 0  # Minimize latency objective

        rates_rads_array = np.zeros(self.N_passes)

        # Explode design vector
        design_vector = explode_design_vector(x, self.N_passes, self.x_indices)

        # Calculate overlap constraint
        overlap = self.O_matrix * design_vector['pass'].astype('bool')
        overlap = overlap[np.tril(np.ones((self.N_passes, self.N_passes)), -1) == 1]
        g_overlap = overlap.flatten()  # Non-overlapping constraint

        # Compute throughput of selected passes
        sel_pass = design_vector['pass'].astype('bool') > 0
        num_pass = int(np.sum(sel_pass))
        if num_pass > 0:
            tof_s_list = List(compress(self.tof_s_list, sel_pass))  # List of tofs of the selected passes
            fspl_dB_list = List(compress(self.fspl_dB_list, sel_pass))
            theta_rad_list = List(compress(self.theta_rad_list, sel_pass))
            phi_rad_list = List(compress(self.phi_rad_list, sel_pass))

            Ptx_dBm_array = design_vector['power'][sel_pass].astype('float64')
            Gtx_dBi = design_vector['antenna'][0]
            B_Hz = self.sys_param.B_Hz_array[design_vector['bandwidth'][0]]
            alpha = self.sys_param.alpha_array[0]
            EsN0_req_dB_array = self.sys_param.EsN0_req_dB_array
            carriers = multi_carrier.get_sub_carriers(B_Hz)
            eta_bitsym_array = np.squeeze(self.sys_param.eta_bitsym_array[:, carriers - 1])
            eta_maee_array = np.squeeze(self.sys_param.eta_maee_array[:, carriers - 1])

            # Throughput
            margin_dB_list, linktime_s_array, f_throughput, vcm_array = vcm.compute_passes_throughput(tof_s_list,
                                                                                                      fspl_dB_list,
                                                                                                      Ptx_dBm_array,
                                                                                                      Gtx_dBi,
                                                                                                      self.sys_param.GT_dBK,
                                                                                                      B_Hz,
                                                                                                      alpha,
                                                                                                      EsN0_req_dB_array,
                                                                                                      eta_bitsym_array,
                                                                                                      self.sys_param.margin_dB)
            # Energy
            eta_maee_array = eta_maee_array[vcm_array]
            f_energy = energy.compute_passes_energy_maee(linktime_s_array, Ptx_dBm_array, eta_maee_array)

            # Pointing
            f_pointing, rates_rads_array[sel_pass] = pointing.compute_pointing_passes(
                tof_s_list, theta_rad_list, phi_rad_list, Gtx_dBi)

            # Latency
            f_latency = latency.compute_latency_passes(tof_s_list, margin_dB_list, self.t_end_s)

        # Handle constraints
        gg_constraint = compute_constraints(self.reqs, f_throughput, f_latency=f_latency, f_energy=f_energy,
                                            f_pointing=f_pointing, rates_rads_array=rates_rads_array)

        out["F"] = [-1*f_throughput, f_latency, f_energy, f_pointing]
        out["G"] = np.concatenate([g_overlap, gg_constraint])
