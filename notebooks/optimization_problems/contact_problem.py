import numpy as np
from pymoo.model.problem import Problem

from hermes.postprocessing import generate_passes_df_reduced
from models.models import compute_overlap_matrix, compute_contact_time
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
        self.O_matrix = compute_overlap_matrix(self.b_tofs, self.e_tofs)

        super().__init__(n_var=self.N_passes,
                         n_obj=1,
                         n_constr=int((self.N_passes*(self.N_passes + 1)) / 2),
                         xl=0,
                         xu=1,
                         elementwise_evaluation=True,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):

        design_vector = explode_design_vector(x, self.N_passes)
        f1 = -1 * compute_contact_time(design_vector['pass'], self.b_tofs, self.e_tofs) # Contact time objective

        overlap = self.O_matrix * design_vector['pass']
        overlap = overlap[np.tril(np.ones((self.N_passes, self.N_passes)), -1) == 1]
        g1 = overlap.flatten()  # Non-overlapping constraint

        out["F"] = f1
        out["G"] = g1

