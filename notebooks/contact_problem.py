import numpy as np
from pymoo.model.problem import Problem

from hermes.postprocessing import generate_passes_df_reduced
from models.models import compute_overlap_matrix, compute_contact_time, explode_design_vector

class ContactProblem(Problem):

    def __init__(self, instances_df, **kwargs):

        self.instances_df = instances_df
        self.N_passes = int(np.max(instances_df.index.get_level_values('p').unique().values))

        # Setup variables needed for optimization
        self.passes_df = generate_passes_df_reduced(instances_df)
        self.b_tofs = self.passes_df['start_tof'].values
        self.e_tofs = self.passes_df['end_tof'].values

        # Compute overlap matrix
        #self.O_matrix = compute_overlap_matrix(self.b_tofs, self.e_tofs)

        B = np.tile(self.b_tofs, (self.N_passes, 1)).T
        E = np.tile(self.e_tofs, (self.N_passes, 1))

        self.O_matrix = np.tril(E - B, -1)

        # super().__init__(n_var=10, n_obj=1, n_constr=0, xl=-5, xu=5,
        #                  elementwise_evaluation=True, **kwargs)

        super().__init__(n_var=self.N_passes,
                         n_obj=1,
                         n_constr=self.N_passes*self.N_passes,
                         xl=0,
                         xu=1,
                         elementwise_evaluation=True,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        #out["F"] = (x ** 2).sum()

        design_vector = explode_design_vector(x, self.N_passes)
        f1 = -1 * compute_contact_time(design_vector['pass'], self.b_tofs, self.e_tofs)

        overlap = self.O_matrix * design_vector['pass']
        #overlap = overlap[np.tril(np.ones((self.N_passes, self.N_passes)), -1) == 1]
        g1 = overlap.flatten()

        out["F"] = f1
        out["G"] = g1

