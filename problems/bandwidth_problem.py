from combined_problem import CombinedProblem

import numpy as np
from notebooks.optimization_problems.constraints import Requirements

class BandwidthProblem(CombinedProblem):
    def __init__(self, instances_df, system_parameters, requirements=Requirements(), *args, **kwargs):
        super().__init__(instances_df, system_parameters, requirements, f_mask=np.array([0, 2]), *args, **kwargs)
