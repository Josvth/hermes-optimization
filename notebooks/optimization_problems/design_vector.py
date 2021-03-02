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
    indices, total_count = _add_vars(indices, total_count, 'bandwidth', N)
    indices, total_count = _add_vars(indices, total_count, 'rolloff', N)
    indices, total_count = _add_vars(indices, total_count, 'modcod', N)

    return total_count, indices


def design_vector_bounds(var_count, indices, system_parameters, real_power = False):
    # Generates the integer bounds for the design vector
    xl = [0] * var_count
    xu = [0] * var_count

    mapping_xl = dict()
    mapping_xl['pass'] = 0
    mapping_xl['power'] = min(system_parameters.Ptx_dBm_list) if real_power else 0
    mapping_xl['antenna'] = 0
    mapping_xl['bandwidth'] = 0
    mapping_xl['rolloff'] = 0
    mapping_xl['modcod'] = 0

    mapping_xu = dict()
    mapping_xu['pass'] = 1
    mapping_xu['power'] = max(system_parameters.Ptx_dBm_list) if real_power else len(system_parameters.Ptx_dBm_list) - 1
    mapping_xu['antenna'] = len(system_parameters.Gtx_dBi_list) - 1
    mapping_xu['bandwidth'] = len(system_parameters.B_Hz_list) - 1
    mapping_xu['rolloff'] = len(system_parameters.alpha_list) - 1
    mapping_xu['modcod'] = len(system_parameters.EsN0_req_dB_list) - 1

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
    mapping_mask['power'] = "real" if real_power else "int"
    mapping_mask['antenna'] = "int"
    mapping_mask['bandwidth'] = "int"
    mapping_mask['rolloff'] = "int"
    mapping_mask['modcod'] = "int"

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
        "int": get_crossover("int_sbx", prob=1.0, eta=3.0),
        "real": get_crossover("real_sbx", prob=0.9, eta=15),
    })

    mutation = MixedVariableMutation(mask, {
        "bin": get_mutation("bin_bitflip"),
        "int": get_mutation("int_pm", eta=3.0),
        "real": get_mutation("real_pm", eta=20),
    })

    return sampling, crossover, mutation

class SystemParameters:
    # Spectral
    fc_Hz = 30e9  # Fixed carrier frequency in [Hz]
    B_Hz_list = [20e6]  # Bandwidth in Hz

    # Waveform
    alpha_list = [0.35]  # Roll-off factor
    EsN0_req_dB_list = [10.69]  # List of required EsN0 for the selectable modulation per pass in [dB]
    eta_bitsym_list = [2.6460120]  # Spectral efficiency in for the selectable modulation per pass in [bits/symbol]

    # Transmitter
    Ptx_dBm_list = list(range(10, 40, 5))  # List of selectable powers per pass in [dBm]
    Gtx_dBi_list = [20]  # List of selectable antenna gains in [dBi]

    # Receiver
    GT_dBK = 13.2  # Receiver G/T figure in [dB/K]


def explode_design_vector(x, N, indices=None):
    design_vector = dict()

    if indices is None:
        _, indices = design_vector_indices(N)

    for k, v in indices.items():
        if len(x) >= v[-1]:
            design_vector[k] = x[v]

    return design_vector