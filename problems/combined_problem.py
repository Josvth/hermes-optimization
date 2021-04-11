import numpy as np
from numba.typed import List
from pymoo.model.problem import Problem

import combined
from hermes.postprocessing import generate_grouped_passed_df, generate_pass_range_list, \
    generate_pass_tof_list, generate_pass_r_ab_list
from models import contact, link_budget, multi_carrier, visibility
from notebooks.optimization_problems.constraints import Requirements
from notebooks.optimization_problems.design_vector import design_vector_indices, design_vector_bounds, \
    explode_design_vector


class CombinedProblem(Problem):

    def __init__(self, instances_df, system_parameters, requirements=Requirements(), f_mask=[0, 1, 2, 3], *args, **kwargs):
        # self.instances_df = instances_df
        self.sys_param = system_parameters
        self.reqs = requirements

        self.N_passes = int(np.max(instances_df.index.get_level_values('p').unique().values))

        grouped_passes_df = generate_grouped_passed_df(instances_df)

        # Setup variables needed for optimization
        self.b_tofs = grouped_passes_df.first().tof.values
        self.e_tofs = grouped_passes_df.last().tof.values
        self.t_sim_s = grouped_passes_df.last().tof.values[-1]

        # Compute overlap matrix
        self.O_matrix = contact.compute_overlap_matrix(self.b_tofs, self.e_tofs)

        # Compute variables needed during pass
        self.tof_s_list = generate_pass_tof_list(grouped_passes_df)
        self.margin_dB_list = self.tof_s_list

        range_m_list = generate_pass_range_list(grouped_passes_df)
        self.fspl_dB_list = link_budget.compute_passes_fspl(range_m_list, self.sys_param.fc_Hz)

        r_ab_m_list = generate_pass_r_ab_list(grouped_passes_df)
        self.theta_rad_list, self.phi_rad_list = visibility.compute_passes_angles(r_ab_m_list)

        # Get design vector indices and bounds
        self.x_length, self.x_indices = design_vector_indices(self.N_passes)
        self.xl, self.xu = design_vector_bounds(self.x_length, self.x_indices, self.sys_param)

        self.f_mask = f_mask

        super().__init__(n_var=self.x_length,
                         n_obj=len(f_mask),
                         n_constr=1 + 5 + self.N_passes,
                         xl=self.xl,
                         xu=self.xu,
                         elementwise_evaluation=True,
                         **kwargs)

    def evaluate_unmasked(self, x):

        # Explode design vector
        design_vector = explode_design_vector(x, self.N_passes, self.x_indices)

        x_pass = design_vector['pass'].astype('bool')
        Ptx_dBm_array = design_vector['power'][x_pass].astype('float64')
        Gtx_dBi = design_vector['antenna'][0]
        B_Hz = self.sys_param.B_Hz_array[int(design_vector['bandwidth'][0])]
        alpha = self.sys_param.alpha_array[0]
        carriers = multi_carrier.get_sub_carriers(B_Hz)
        EsN0_req_dB_array = np.squeeze(self.sys_param.EsN0_req_dB_array[:, carriers - 1])
        eta_bitsym_array = np.squeeze(self.sys_param.eta_bitsym_array[:, carriers - 1])
        eta_maee_array = np.squeeze(self.sys_param.eta_maee_array[:, carriers - 1])

        ff, gg = combined.compute_fg(
            self.N_passes, self.O_matrix, self.t_sim_s,
            List(self.tof_s_list), List(self.fspl_dB_list), List(self.theta_rad_list), List(self.phi_rad_list),
            x_pass, Ptx_dBm_array, Gtx_dBi, self.sys_param.GT_dBK, B_Hz, alpha,
            EsN0_req_dB_array, eta_bitsym_array, eta_maee_array, self.sys_param.margin_dB,
            self.reqs.min_throughput, self.reqs.max_throughput, self.reqs.max_latency, self.reqs.max_energy,
            self.reqs.max_pointing, self.reqs.max_rate_rads)

        return ff, gg

    def _evaluate(self, x, out, *args, **kwargs):

        ff, gg = self.evaluate_unmasked(x)

        out["F"] = ff[self.f_mask]
        out["G"] = gg

class CombinedProblemDownSelect(CombinedProblem):

    def __init__(self, instances_df, system_parameters, requirements=Requirements(), f_mask=[0, 1, 2, 3], *args,
                 **kwargs):
        super().__init__(instances_df, system_parameters, requirements, f_mask, *args, **kwargs)
        self.n_constr = self.n_constr - 1

    def evaluate_unmasked(self, x):

        # Explode design vector
        design_vector = explode_design_vector(x, self.N_passes, self.x_indices)

        x_pass = design_vector['pass'].astype('bool')
        Ptx_dBm_array = design_vector['power'].astype('float64')
        Gtx_dBi = design_vector['antenna'][0]
        B_Hz = self.sys_param.B_Hz_array[int(design_vector['bandwidth'][0])]
        alpha = self.sys_param.alpha_array[0]
        carriers = multi_carrier.get_sub_carriers(B_Hz)
        EsN0_req_dB_array = np.squeeze(self.sys_param.EsN0_req_dB_array[:, carriers - 1])
        eta_bitsym_array = np.squeeze(self.sys_param.eta_bitsym_array[:, carriers - 1])
        eta_maee_array = np.squeeze(self.sys_param.eta_maee_array[:, carriers - 1])

        ff, gg = combined.compute_fg_no_down_select(
            self.N_passes, self.O_matrix, self.t_sim_s,
            List(self.tof_s_list), List(self.fspl_dB_list), List(self.theta_rad_list), List(self.phi_rad_list),
            x_pass, Ptx_dBm_array, Gtx_dBi, self.sys_param.GT_dBK, B_Hz, alpha,
            EsN0_req_dB_array, eta_bitsym_array, eta_maee_array, self.sys_param.margin_dB,
            self.reqs.min_throughput, self.reqs.max_throughput, self.reqs.max_latency, self.reqs.max_energy,
            self.reqs.max_pointing, self.reqs.max_rate_rads)

        return ff, gg