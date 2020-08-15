import pandas as pd
from pymoo.util.termination.default import MultiObjectiveDefaultTermination

from pymoo.algorithms.nsga2 import NSGA2
from notebooks.problems import VCMProblem, VCMProblemPool

import numpy as np

from pymoo.optimize import minimize

from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

if __name__ == '__main__':

    telesat_passes_df = pd.read_pickle('Telesat_0_7day.pkl')

    N_passes = 20
    telesat_passes_df = telesat_passes_df.head(N_passes)

    # Load modcods
    dvbs2_df = pd.read_csv('modcods/csv/dvbs2.csv', index_col='ID')

    fc = 30e9
    GT_dBk = 13.2
    Ptx_dBm = np.arange(20, 40, 0.1)

    design_vars = {
        'Ptx_dBm_set': Ptx_dBm,   # 0 to 10 W
        'bandwidth_Hz_set': [20e6],
        'Gtx_dBi_set': [20]
    }

    problem = VCMProblemPool(telesat_passes_df, dvbs2_df, fc, GT_dBk, design_vars)

    sampling, crossover, mutation = problem.generate_default_scm()

    termination = MultiObjectiveDefaultTermination(
        #x_tol=1e-8,
        #cv_tol=1e-6,
        f_tol=0.001, #f_tol=0.005,
        nth_gen=5,
        n_last=30,
        n_max_gen=1000, #n_max_gen=1000,
        n_max_evals=100000 #n_max_evals=100000
    )

    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=300,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    res = minimize(problem,
                   algorithm,
                   termination=termination,
                   seed=1,
                   save_history=False,
                   verbose=True
                   )
