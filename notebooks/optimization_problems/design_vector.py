import numpy as np
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling
from pymoo.operators.crossover.half_uniform_crossover import HalfUniformCrossover
from pymoo.util.normalization import denormalize


def design_vector_indices(N):
    def _add_vars(indices, total_count, var_name, var_count):
        indices[var_name] = list(range(total_count, total_count + var_count))
        total_count = total_count + var_count
        return indices, total_count

    indices = dict()
    total_count = 0

    indices, total_count = _add_vars(indices, total_count, 'pass', N)
    indices, total_count = _add_vars(indices, total_count, 'power', N)
    indices, total_count = _add_vars(indices, total_count, 'antenna', 1)
    indices, total_count = _add_vars(indices, total_count, 'bandwidth', 1)
    #indices, total_count = _add_vars(indices, total_count, 'rolloff', 1)
    #indices, total_count = _add_vars(indices, total_count, 'modcod', 1)

    return total_count, indices


def design_vector_bounds(var_count, indices, system_parameters):
    # Generates the integer bounds for the design vector
    xl = [0] * var_count
    xu = [0] * var_count

    mapping_xl = dict()
    mapping_xl['pass'] = 0
    mapping_xl['power'] = min(system_parameters.Ptx_dBm_bounds)
    mapping_xl['antenna'] = min(system_parameters.Gtx_dBi_bounds)
    mapping_xl['bandwidth'] = 0
    #mapping_xl['rolloff'] = 0
    #mapping_xl['modcod'] = 0

    mapping_xu = dict()
    mapping_xu['pass'] = 1
    mapping_xu['power'] = max(system_parameters.Ptx_dBm_bounds)
    mapping_xu['antenna'] = max(system_parameters.Gtx_dBi_bounds)
    mapping_xu['bandwidth'] = len(system_parameters.B_Hz_array) - 1
    #mapping_xu['rolloff'] = 0 #len(system_parameters.alpha_list) - 1
    #mapping_xu['modcod'] = len(system_parameters.EsN0_req_dB_array) - 1

    for k, v in indices.items():
        for i in v:
            xl[i] = mapping_xl[k]
            xu[i] = mapping_xu[k]

    return xl, xu

def design_vector_default_scm(var_count, indices, real_power = False):
    from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, \
        MixedVariableMutation
    from pymoo.factory import get_sampling, get_crossover, get_mutation

    mapping_mask = dict()
    mapping_mask['pass'] = "bin"
    mapping_mask['power'] = "real"
    mapping_mask['antenna'] = "real"
    mapping_mask['bandwidth'] = "int"
    #mapping_mask['rolloff'] = "int"
    #mapping_mask['modcod'] = "int"

    mask = [None] * var_count
    for k, v in indices.items():
        for i in v:
            mask[i] = mapping_mask[k]

    sampling = MixedVariableSampling(mask, {
        "bin": get_sampling("bin_random"),
        "int": get_sampling("int_random"),
        "real": get_sampling("real_random")
    })

    crossover = MixedVariableCrossover(mask, {
        "bin": get_crossover("bin_hux"),
        #"bin": get_crossover("bin_k_point", n_points=2),
        "int": get_crossover("int_sbx", prob=1.0, eta=3.0),
        "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    })

    mutation = MixedVariableMutation(mask, {
        "bin": get_mutation("bin_bitflip", prob=0.01),
        "int": get_mutation("int_pm", eta=3.0),
        "real": get_mutation("real_pm", eta=3.0),
    })

    return sampling, crossover, mutation

class NoCrossover(Crossover):

    def __init__(self, **kwargs):
        super().__init__(2, 2, **kwargs)

    def _do(self, problem, X, **kwargs):
        return X

def design_vector_no_crossover_scm(var_count, indices):
    from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, \
        MixedVariableMutation
    from pymoo.factory import get_sampling, get_crossover, get_mutation

    mapping_mask = dict()
    mapping_mask['pass'] = "bin-no-cross"
    mapping_mask['power'] = "real-no-cross"
    mapping_mask['antenna'] = "real"
    mapping_mask['bandwidth'] = "int"
    #mapping_mask['rolloff'] = "int"
    #mapping_mask['modcod'] = "int"

    mask = [None] * var_count
    for k, v in indices.items():
        for i in v:
            mask[i] = mapping_mask[k]

    sampling = MixedVariableSampling(mask, {
        "bin-no-cross": get_sampling("bin_random"),
        "real-no-cross": get_sampling("real_random"),
        "int": get_sampling("int_random"),
        "real": get_sampling("real_random")
    })

    crossover = MixedVariableCrossover(mask, {
        "bin-no-cross": NoCrossover(),
        "real-no-cross": NoCrossover(),
        "int": get_crossover("int_sbx", prob=1.0, eta=3.0),
        "real": get_crossover("real_sbx", prob=1.0, eta=3.0)
    })

    mutation = MixedVariableMutation(mask, {
        "bin-no-cross": get_mutation("bin_bitflip", prob=0.01),
        "real-no-cross": get_mutation("real_pm", eta=3.0),
        "int": get_mutation("int_pm", eta=3.0),
        "real": get_mutation("real_pm", eta=3.0),
    })

    return sampling, crossover, mutation

def design_vector_no_crossover_mut_scm(var_count, indices):
    from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, \
        MixedVariableMutation
    from pymoo.factory import get_sampling, get_crossover, get_mutation

    mapping_mask = dict()
    mapping_mask['pass'] = "bin-no-cross"
    mapping_mask['power'] = "real-no-cross"
    mapping_mask['antenna'] = "real"
    mapping_mask['bandwidth'] = "int"
    #mapping_mask['rolloff'] = "int"
    #mapping_mask['modcod'] = "int"

    mask = [None] * var_count
    for k, v in indices.items():
        for i in v:
            mask[i] = mapping_mask[k]

    sampling = MixedVariableSampling(mask, {
        "bin-no-cross": get_sampling("bin_random"),
        "real-no-cross": get_sampling("real_random"),
        "int": get_sampling("int_random"),
        "real": get_sampling("real_random")
    })

    crossover = MixedVariableCrossover(mask, {
        "bin-no-cross": NoCrossover(),
        "real-no-cross": NoCrossover(),
        "int": get_crossover("int_sbx", prob=1.0, eta=3.0),
        "real": get_crossover("real_sbx", prob=1.0, eta=3.0)
    })

    mutation = MixedVariableMutation(mask, {
        "bin-no-cross": get_mutation("bin_bitflip", prob=0.001),
        "real-no-cross": get_mutation("real_pm", eta=3.0, prob=0.01),
        "int": get_mutation("int_pm", eta=3.0, prob=0.01),
        "real": get_mutation("real_pm", eta=3.0, prob=0.01),
    })

    return sampling, crossover, mutation

class PassPowerSampling(Sampling):

    def __init__(self, pass_sampling, power_sampling, **kwargs):
        super().__init__()
        self.pass_sampling = pass_sampling
        self.power_sampling = power_sampling

    def _do(self, problem, X, **kwargs):
        pass
        import warnings
        warnings.error("NOT IMPLEMENTED YET")
        return X

class PassPowerCrossover(Crossover):

    def __init__(self, pass_crossover=HalfUniformCrossover(), **kwargs):
        super().__init__(2, 2, **kwargs)
        self.pass_crossover = pass_crossover

    def _do(self, problem, X, **kwargs):

        n_parents, n_individuals, n_vars = X.shape

        X_pass = X[:, :, :int(n_vars/2)].astype('bool')
        X_power = X[:, :, int(n_vars/2):].astype('float')

        _X_pass = self.pass_crossover._do(problem, X_pass)

        only_a = X_pass[0, ...] & ~X_pass[1, ...] & _X_pass
        only_b = ~X_pass[0, ...] & X_pass[1, ...] & _X_pass
        both = X_pass[0, ...] & X_pass[1, ...] & _X_pass

        _X_power = X_power
        _X_power[only_a] = X_power[only_a]
        _X_power[only_b] = X_power[only_b]
        _X_power[both] = 10*np.log10((10**((X_power[both] - 30)/10) + 10**((X_power[both] - 30)/10))/2) + 30

        _X = np.concatenate((_X_pass, _X_power), axis=2)

        return _X

class PassPowerMutation(Mutation):

    def __init__(self, pass_mutation, power_mutation, **kwargs):
        super().__init__()
        self.pass_mutation = pass_mutation
        self.power_mutation = power_mutation

    def _do(self, problem, X, **kwargs):

        n_individuals, n_vars = X.shape

        X_pass = X[:, :int(n_vars / 2)]
        X_power = X[:, int(n_vars / 2):]

        # Change the problem variables to spoof the mutation function
        _n_var, _xl, _xu = problem.n_var, problem.xl, problem.xu

        problem.n_var, problem.xl, problem.xu = n_vars, _xl[:int(n_vars / 2)], _xu[:int(n_vars / 2)]
        _X_pass = self.pass_mutation._do(problem, X_pass)

        problem.n_var, problem.xl, problem.xu = n_vars, _xl[int(n_vars / 2):], _xu[int(n_vars / 2):]
        _X_power = self.power_mutation._do(problem, X_power)

        # reset the original bounds of the problem
        problem.n_var = _n_var
        problem.xl = _xl
        problem.xu = _xu

        _X = np.concatenate((_X_pass, _X_power), axis=1)

        return  _X

def design_vector_passpower_scm(var_count, indices, real_power=False):
    from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, \
        MixedVariableMutation
    from pymoo.factory import get_sampling, get_crossover, get_mutation

    mapping_mask = dict()
    mapping_mask['pass'] = "passpower"
    mapping_mask['power'] = "passpower"
    mapping_mask['antenna'] = "real"
    mapping_mask['bandwidth'] = "int"
    # mapping_mask['rolloff'] = "int"
    # mapping_mask['modcod'] = "int"

    mask = [None] * var_count
    for k, v in indices.items():
        for i in v:
            mask[i] = mapping_mask[k]

    sampling = MixedVariableSampling(mask, {
        "passpower": PassPowerSampling(get_sampling("bin_random"),get_sampling("real_random")),
        "int": get_sampling("int_random"),
        "real": get_sampling("real_random")
    })

    crossover = MixedVariableCrossover(mask, {
        "passpower": PassPowerCrossover(),
        "int": get_crossover("int_sbx", prob=1.0, eta=3.0),
        "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    })

    mutation = MixedVariableMutation(mask, {
        "passpower": PassPowerMutation(get_mutation("bin_bitflip", prob=0.01), get_mutation("real_pm", eta=3.0)),
        "int": get_mutation("int_pm", eta=3.0),
        "real": get_mutation("real_pm", eta=3.0),
    })

    return sampling, crossover, mutation

class SystemParameters:
    # Spectral
    fc_Hz = 20e9  # Fixed carrier frequency in [Hz]
    B_Hz_array = np.array([20e6]) # Bandwidth in [Hz]

    # Waveform
    alpha_array = np.array([0.35]) # Roll-off factor for all passes
    EsN0_req_dB_array = np.array([10.69]) # List of required EsN0 for the selectable modulation per pass in [dB]
    eta_bitsym_array = np.array([2.6460120]) # Spectral efficiency in for the selectable modulation per pass in [bits/symbol]
    eta_maee_array = np.array([1.0])

    # Transmitter
    Ptx_dBm_bounds = (10, 40) # Bounds of selectable power per pass in [dBm]
    Gtx_dBi_bounds = (20, 20) # Bounds of selectable antenna gains in [dBi]

    # Receiver
    GT_dBK = 13.2  # Receiver G/T figure in [dB/K]

    # Margin
    margin_dB = 0.0 # Link margin in [dB]

def explode_design_vector(x, N, indices=None):
    design_vector = dict()

    if indices is None:
        _, indices = design_vector_indices(N)

    for k, v in indices.items():
        if len(x) >= v[-1]:
            design_vector[k] = x[v]

    return design_vector


class InitialSampling(Sampling):
    """
    Randomly sample points in the real space by considering the lower and upper bounds of the problem.
    """

    def __init__(self, var_type=np.float) -> None:
        super().__init__()
        self.var_type = var_type

    def _do(self, problem, n_samples, **kwargs):

        # Generate random for all
        val = np.random.random((n_samples, problem.n_var))
        val = denormalize(val, problem.xl, problem.xu)

        # Then specifically select 1 pass in each solution
        N_pass = int((problem.n_var - 2) / 2)
        _, ind = design_vector_indices(N_pass)
        for i in range(n_samples):
            val[i, 0:N_pass] = False
            val[i, i % N_pass] = True

        return val