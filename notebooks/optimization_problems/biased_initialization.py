from pymoo.algorithms.nsga3 import NSGA3
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination

import util
from contact_problem import ContactProblemDownSelect
from optimization_problems.design_vector import design_vector_default_scm
import numpy as np

def max_contact_biased(instances_df, sys_param, case, pop_size, n_partitions=100, plot = True):

    init_problem = ContactProblemDownSelect(instances_df, sys_param)
    sampling, crossover, mutation = design_vector_default_scm(init_problem.x_length, init_problem.x_indices)

    # NGSA-III optimization
    from pymoo.factory import get_reference_directions
    init_ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=n_partitions)
    init_pop_size = int(np.ceil(len(init_ref_dirs) / 100) * 100)
    init_n_offsprings = int(np.ceil(len(init_ref_dirs) / 100) * 100 / 4)

    print("init_ref_dirs: %d" % len(init_ref_dirs))
    print("init_pop_size: %d" % init_pop_size)
    print("init_n_offsprings: %d" % init_n_offsprings)

    algorithm = NSGA3(
        pop_size=init_pop_size,
        n_offsprings=init_n_offsprings,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        ref_dirs=init_ref_dirs,
        eliminate_duplicates=False,
    )

    termination = MultiObjectiveSpaceToleranceTermination(tol=0.00001,
                                                          n_last=30,
                                                          nth_gen=5,
                                                          n_max_gen=None,
                                                          n_max_evals=None)

    print("Running initial conditions optimization...")
    init_res = minimize(init_problem,
                        algorithm,
                        termination,
                        seed=1,
                        # verbose=True,
                        copy_algorithm=False,
                        )

    x_pass, _, _, _ = util.get_selection(init_problem, init_res)

    if plot:
        print("Plotting %d results" % len(init_res.X))

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(init_res.F[:, 1], init_res.F[:, 0] * -1, marker='.')
        ax.set_xlabel('Passes used [.]')
        ax.set_ylabel('Maximum contact time [s]')

        util.plot_used_passes(case, instances_df, x_pass[np.argmin(init_res.F[:, 0]), :])

    print("Generating initial population of size %d..." % pop_size)
    init_pop = init_res.X  # Load design variables
    init_pop[:, init_problem.x_indices['pass']] = x_pass  # Filter down-selected passes
    repeats = int(np.ceil(pop_size / len(init_res.X)))
    init_pop = np.tile(init_pop, (repeats, 1))
    init_pop = init_pop[0:pop_size, :]  # Reduce to population size
    sample_pop = sampling._do(init_problem, pop_size)  # Sample a random population
    sample_pop[:, init_problem.x_indices['pass']] = init_pop[:, init_problem.x_indices['pass']]  # Set pass selection

    return sample_pop