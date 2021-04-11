import numpy as np
from pymoo.model.problem import Problem

import contact
from hermes.postprocessing import generate_passes_df_reduced
from notebooks.optimization_problems.design_vector import explode_design_vector, design_vector_indices


class InitialContactProblem(Problem):

    def __init__(self, instances_df, **kwargs):

        self.instances_df = instances_df
        self.N_passes = int(np.max(instances_df.index.get_level_values('p').unique().values))

        # Setup variables needed for optimization
        self.passes_df = generate_passes_df_reduced(instances_df)
        self.b_tofs = self.passes_df['start_tof'].values
        self.e_tofs = self.passes_df['end_tof'].values

        # Compute overlap matrix
        self.O_matrix = contact.compute_overlap_matrix(self.b_tofs, self.e_tofs)

        self.x_length = self.N_passes
        self.x_indices = {'pass': list(range(self.N_passes))}

        super().__init__(n_var=self.N_passes,
                         n_obj=2,
                         n_constr=1,
                         xl=0,
                         xu=1,
                         elementwise_evaluation=True,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        design_vector = explode_design_vector(x, self.N_passes)

        x_pass = design_vector['pass'].astype('bool')

        f_passes = np.sum(x_pass)
        f_contact = contact.compute_contact_time(x_pass, self.b_tofs, self.e_tofs) # Contact time objective

        g_overlap = contact.compute_overlap(x_pass, self.O_matrix)

        out["F"] = [-1 * f_contact, -1 * f_passes]
        out["G"] = g_overlap

