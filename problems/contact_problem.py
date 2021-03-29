import numpy as np
from pymoo.model.problem import Problem

import contact
from hermes.postprocessing import generate_passes_df_reduced
from notebooks.optimization_problems.design_vector import explode_design_vector


class ContactProblem(Problem):

    def __init__(self, instances_df, **kwargs):

        self.instances_df = instances_df
        self.N_passes = int(np.max(instances_df.index.get_level_values('p').unique().values))

        # Setup variables needed for optimization
        self.passes_df = generate_passes_df_reduced(instances_df)
        self.b_tofs = self.passes_df['start_tof'].values
        self.e_tofs = self.passes_df['end_tof'].values

        # Compute overlap matrix
        self.O_matrix = contact.compute_overlap_matrix(self.b_tofs, self.e_tofs)

        super().__init__(n_var=self.N_passes,
                         n_obj=1,
                         n_constr=int((self.N_passes*(self.N_passes + 1)) / 2),
                         xl=0,
                         xu=1,
                         elementwise_evaluation=True,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        design_vector = explode_design_vector(x, self.N_passes)

        x_pass = design_vector['pass'].astype('bool')
        f1 = -1 * contact.compute_contact_time(x_pass, self.b_tofs, self.e_tofs) # Contact time objective

        x_pass_tile = x_pass.repeat(self.N_passes).reshape((-1, self.N_passes))
        overlap = self.O_matrix * x_pass_tile * x_pass_tile.T
        g_overlap = np.sum(overlap > 0)

        out["F"] = f1
        out["G"] = g_overlap

