from pymoo.model.problem import Problem
import numpy as np
from numba import jit

from models.link_budget import calculate_snr, calculate_isend


class LinkBudgetProblem(Problem):

    def __init__(self, passes_df, N_passes, fc, Ptx_dBm, Gtx_dB, GT, B, isend_req, Rs, eta):
        self.passes_df = passes_df.head(N_passes)  # Pandas dataframe containing all pass information
        self.N_passes = N_passes  # Number of passes to consider in the optimization
        self.fc = fc  # Carrier frequency in [Hz]
        self.Ptx_dBm = Ptx_dBm  # Numpy array of transmission powers in [dBm]
        self.Gtx_dB = Gtx_dB  # Transmitter antenna gain in [dBi]
        self.GT = GT  # Receiver G/T figure in [dB/K]
        self.B = B  # Transmission bandwidth in [Hz]
        self.isend_req = isend_req  # Es/N0 requirement in [dB]
        self.Rs = Rs  # Symbol rate in [syms/s]
        self.eta = eta  # Spectral efficiency in [bits/symbol]
        self.O = self.generate_non_overlapping_matrix()  # Generates the non overlapping matrix
        super().__init__(
            n_var=N_passes + 1,
            n_obj=2,
            n_constr=N_passes * N_passes,
            xl=[0] * (N_passes + 1), xu=[1] * N_passes + [len(Ptx_dBm) - 1])

    def _evaluate(self, xx, out, *args, **kwargs):

        # --- snipp! ---#
        f1 = np.zeros((np.size(xx, 0), 1))  # Results of objective function 1
        f2 = np.zeros((np.size(xx, 0), 1))  # Results of objective function 2
        g1 = np.zeros((np.size(xx, 0), self.N_passes * self.N_passes))

        # Non-vectorized for loop - SLOW!
        for i in range(np.size(xx, 0)):
            x = xx[i, :]  # obtain one sample in the sample set to evaluate

            contact_time, overlap, power_consumption, total_link_time, _ = self.evaluate_passes(x)

            # Pass results to output matrices
            f1[i] = -1 * total_link_time * self.Rs * self.eta  # Througput in bits per second
            f2[i] = power_consumption
            g1[i, :] = overlap.flatten()

        out["F"] = np.column_stack([f1, f2])
        out["G"] = g1

    def evaluate_passes(self, x):
        xpass = x[0:-1]  # Binary variables that select passes
        xpower = x[-1]  # Integer variable that selects power

        # Calculate contact time
        b_tofs = self.passes_df.StartTof[xpass].values
        e_tofs = self.passes_df.StopTof[xpass].values
        contact_time = np.sum(e_tofs + 1 - b_tofs)

        # Calculate overlap constraints
        overlap = self.O * xpass

        # Calculate power consumption
        power_consumption = contact_time * 10 ** ((self.Ptx_dBm[xpower] - 30) / 10)

        # Link budget calculation for each pass
        isend_dB_list = [None] * len(self.passes_df)

        total_link_time = 0
        for i, pass_df in self.passes_df.loc[xpass].iterrows():
            d = np.linalg.norm(pass_df.sff, axis=1)  # Range at each tof during pass
            Ptx_dBm = self.Ptx_dBm[xpower]  # Transmit power
            Gtx_dB = self.Gtx_dB  # Transmitter gain
            fspl_dB = 20 * np.log10(d) + 20 * np.log10(self.fc) - 147.55  # Free space path loss
            GT_dB = self.GT  # G/T figure
            kB_dB = 10 * np.log10(self.B * 1.380649e-23)  # k*B in dB

            # Received SNR on each tof during pass
            SNR_dB = Ptx_dBm + Gtx_dB - fspl_dB + GT_dB - kB_dB - 30.
            isend_dB_list[i] = SNR_dB - 10 * np.log10(self.Rs / self.B)  # ISEND on each tof during pass

            # Link time where the ISEND is higher than the requirement
            link_time = np.sum(isend_dB_list[i] >= self.isend_req)

            total_link_time = total_link_time + link_time

        return contact_time, overlap, power_consumption, total_link_time, isend_dB_list

    def generate_non_overlapping_matrix(self):

        # Construct the non-overlapping pass matrix
        b_tofs = self.passes_df.StartTof.values  # vector with pass begin tofs
        e_tofs = self.passes_df.StopTof.values  # vector with pass end tofs

        b_mat = np.tile(b_tofs, (len(self.passes_df), 1)).T
        e_mat = np.tile(e_tofs, (len(self.passes_df), 1)) + 1

        ovlp_mat = np.tril(e_mat - b_mat, -1)  # Only lower triangular of the matrix is used
        ovlp_mat = (ovlp_mat > 0) * 1  # The matrix is normalized to give all constraints equal importance

        return ovlp_mat


class LinkBudgetProblemNonZero(Problem):

    def __init__(self, passes_df, N_passes, fc, Ptx_dBm, Gtx_dB, GT, B, isend_req, Rs, eta):
        self.passes_df = passes_df.head(N_passes)  # Pandas dataframe containing all pass information
        self.N_passes = N_passes  # Number of passes to consider in the optimization
        self.fc = fc  # Carrier frequency in [Hz]
        self.Ptx_dBm = Ptx_dBm  # Numpy array of transmission powers in [dBm]
        self.Gtx_dB = Gtx_dB  # Transmitter antenna gain in [dBi]
        self.GT = GT  # Receiver G/T figure in [dB/K]
        self.B = B  # Transmission bandwidth in [Hz]
        self.isend_req = isend_req  # Es/N0 requirement in [dB]
        self.Rs = Rs  # Symbol rate in [syms/s]
        self.eta = eta  # Spectral efficiency in [bits/symbol]
        self.O = self.generate_non_overlapping_matrix()  # Generates the non overlapping matrix
        super().__init__(
            n_var=N_passes + 1,
            n_obj=2,
            n_constr=N_passes * N_passes + 1,
            xl=[0] * (N_passes + 1), xu=[1] * N_passes + [len(Ptx_dBm) - 1])

    def _evaluate(self, xx, out, *args, **kwargs):

        # --- snipp! ---#
        f1 = np.zeros((np.size(xx, 0), 1))  # Results of objective function 1
        f2 = np.zeros((np.size(xx, 0), 1))  # Results of objective function 2
        g1 = np.zeros((np.size(xx, 0), self.N_passes * self.N_passes))
        g2 = np.zeros((np.size(xx, 0), 1))

        # Non-vectorized for loop - SLOW!
        for i in range(np.size(xx, 0)):
            x = xx[i, :]  # obtain one sample in the sample set to evaluate

            contact_time, overlap, power_consumption, total_link_time, _ = self.evaluate_passes(x)

            # Pass results to output matrices
            f1[i] = -1 * total_link_time * self.Rs * self.eta  # Througput in bits per second
            f2[i] = power_consumption
            g1[i, :] = overlap.flatten()
            g2[i] = total_link_time <= 0

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

    def evaluate_passes(self, x):
        xpass = x[0:-1]  # Binary variables that select passes
        xpower = x[-1]  # Integer variable that selects power

        # Calculate contact time
        b_tofs = self.passes_df.StartTof[xpass].values
        e_tofs = self.passes_df.StopTof[xpass].values
        contact_time = np.sum(e_tofs + 1 - b_tofs)

        # Calculate overlap constraints
        overlap = self.O * xpass

        # Calculate power consumption
        power_consumption = contact_time * 10 ** ((self.Ptx_dBm[xpower] - 30) / 10)

        # Link budget calculation for each pass
        isend_dB_list = [None] * len(self.passes_df)

        total_link_time = 0
        for i, pass_df in self.passes_df.loc[xpass].iterrows():
            d = np.linalg.norm(pass_df.sff, axis=1)  # Range at each tof during pass
            Ptx_dBm = self.Ptx_dBm[xpower]  # Transmit power
            Gtx_dB = self.Gtx_dB  # Transmitter gain
            fspl_dB = 20 * np.log10(d) + 20 * np.log10(self.fc) - 147.55  # Free space path loss
            GT_dB = self.GT  # G/T figure
            kB_dB = 10 * np.log10(self.B * 1.380649e-23)  # k*B in dB

            # Received SNR on each tof during pass
            SNR_dB = Ptx_dBm + Gtx_dB - fspl_dB + GT_dB - kB_dB - 30.
            isend_dB_list[i] = SNR_dB - 10 * np.log10(self.Rs / self.B)  # ISEND on each tof during pass

             # Link time where the ISEND is higher than the requirement
            link_time = np.sum(isend_dB_list[i] >= self.isend_req)

            total_link_time = total_link_time + link_time

        return contact_time, overlap, power_consumption, total_link_time, isend_dB_list

    def generate_non_overlapping_matrix(self):

        # Construct the non-overlapping pass matrix
        b_tofs = self.passes_df.StartTof.values  # vector with pass begin tofs
        e_tofs = self.passes_df.StopTof.values  # vector with pass end tofs

        b_mat = np.tile(b_tofs, (len(self.passes_df), 1)).T
        e_mat = np.tile(e_tofs, (len(self.passes_df), 1)) + 1

        ovlp_mat = np.tril(e_mat - b_mat, -1)  # Only lower triangular of the matrix is used
        ovlp_mat = (ovlp_mat > 0) * 1  # The matrix is normalized to give all constraints equal importance

        return ovlp_mat


import multiprocessing
class VCMProblem(Problem):
    # Class variables
    modcods_df = None
    fc = None  # Carrier frequency in [Hz]
    passes_df = None
    N = None
    overlap = None

    design_vars = None

    def __init__(self, passes_df, modcods_df, fc, GT_dBK, design_vars):

        self.modcods_df = modcods_df

        self.fc = fc
        self.GT_dBK = GT_dBK
        self.passes_df = self.calculate_fspl(passes_df, fc)
        self.N = len(self.passes_df)
        self.overlap = self.generate_non_overlapping_matrix(passes_df)

        self.n_overlap_constr = (self.N * (self.N - 1)) >> 1

        ## Extract from pandas for fast access
        self.ttof = self.passes_df.ttof.to_list()
        self.ffspl_dB = self.passes_df.fspl_dB.tolist()

        self.eeta = self.modcods_df.eta.tolist()
        self.iisend_req_dB = self.modcods_df.isend.tolist()

        self.Ptx_dBm_set = design_vars['Ptx_dBm_set']
        self.Gtx_dBi_set = design_vars['Gtx_dBi_set']
        self.bandwidth_Hz_set = design_vars['bandwidth_Hz_set']

        # Design vector indices
        self.x_indices = dict()
        self.x_indices['pass'] = np.arange(self.N)
        self.x_indices['power'] = np.arange(self.N, 2*self.N)
        self.x_indices['modcod'] = np.arange(2*self.N, 3*self.N)
        self.x_indices['bandwidth'] = 3*self.N
        self.x_indices['antenna'] = 3*self.N + 1
        n_var = self.x_indices['antenna'] + 1

        xl = np.array([0] * n_var)
        xl[self.x_indices['pass']] = 0          # Pass selection
        xl[self.x_indices['power']] = 0         # Minimum power index per pass
        xl[self.x_indices['modcod']] = 0        # Minimum MODCOD index
        xl[self.x_indices['bandwidth']] = 0     # Minimum bandwidth index
        xl[self.x_indices['antenna']] = 0       # Minimum antenna gain index

        xu = np.array([0] * n_var)
        xu[self.x_indices['pass']] = 0          # Pass selection
        xu[self.x_indices['power']] = len(self.Ptx_dBm_set) - 1             # Maximum power index per pass
        xu[self.x_indices['modcod']] = len(self.modcods_df) - 1             # Maximum MODCOD index
        xu[self.x_indices['bandwidth']] = len(self.bandwidth_Hz_set) - 1    # Maximum bandwidth index
        xu[self.x_indices['antenna']] = len(self.Gtx_dBi_set) - 1           # Maximum antenna gain index

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=self.n_overlap_constr + 1,
            xl=xl,
            xu=xu)

    def generate_default_scm(self):

        from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, \
            MixedVariableMutation
        from pymoo.factory import get_sampling, get_crossover, get_mutation

        mask = []
        mask = mask + ["bin"] * self.N  # Selection per pass
        mask = mask + ["int"] * self.N  # Power index per pass
        mask = mask + ["int"] * self.N  # MODCOD index per pass
        mask = mask + ["int"]  # Bandwidth index
        mask = mask + ["int"]  # Antenna gain index

        sampling = MixedVariableSampling(mask, {
            "bin": get_sampling("bin_random"),
            "int": get_sampling("int_random")
        })

        crossover = MixedVariableCrossover(mask, {
            "bin": get_crossover("bin_hux"),
            "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
        })

        mutation = MixedVariableMutation(mask, {
            "bin": get_mutation("bin_bitflip"),
            "int": get_mutation("int_pm", eta=3.0)
        })

        return sampling, crossover, mutation

    def _evaluate(self, xx, out, *args, **kwargs):

        f1 = np.zeros((len(xx), 1))  # Results of objective function 1
        f2 = np.zeros((len(xx), 1))  # Results of objective function 2

        g1 = np.zeros((len(xx), self.n_overlap_constr))

        # Process design by design
        for i in range(len(xx)):

            x = xx[i, :]

            f_throughput, f_energy, g_overlap = self._evaluate_design(x)

            f1[i] = f_throughput
            f2[i] = f_energy
            g1[i,:] = g_overlap

        # Calculate throughput must be > 0 (inversed)
        g2 = (f1 >= 0) * 1

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

    def _evaluate_design(self, x):

        # Extract modcod settings for each pass
        f_throughput = 0
        f_energy = 0

        # Calculate overlap constraint
        overlap = self.overlap * x[0:self.N]
        overlap = overlap[np.tril(np.ones(overlap.shape), -1) == 1]
        g_overlap = overlap.flatten()

        # Process used passes
        for j in np.argwhere(x[self.x_indices['pass']]).flatten().tolist():

            # Setting indexes
            power_index = x[self.x_indices['power'][j]]
            modcod_index = x[self.x_indices['modcod'][j]]
            bandwidth_index = x[self.x_indices['bandwidth']]
            antenna_index = x[self.x_indices['antenna']]

            # Link parameters
            ttof = self.ttof[j]
            fspl_dB = self.ffspl_dB[j]  # Free space path loss in dB
            Ptx_dBm = self.Ptx_dBm_set[power_index]  # Transmit power [dBm]
            bandwidth_Hz = self.bandwidth_Hz_set[bandwidth_index]
            Gtx_dB = self.Gtx_dBi_set[antenna_index]
            GT_dBK = self.GT_dBK

            # Modcod
            alpha = 0.35
            eta = self.eeta[modcod_index]
            isend_req_dB = self.eeta[modcod_index]

            result = self._evaluate_pass(
                ttof=ttof,
                fspl_dB=fspl_dB,
                Ptx_dBm=Ptx_dBm,
                Gtx_dB=Gtx_dB,
                GT_dBK=GT_dBK,
                bandwidth_Hz=bandwidth_Hz,
                alpha=alpha,
                eta=eta,
                isend_req_dB=isend_req_dB,
                dt=1)

            f_throughput = f_throughput + result['Throughput']  # Throughput
            f_energy = f_energy + result['Energy']  # Energy

        return f_throughput, f_energy, g_overlap

    def _evaluate_pass(self, ttof, fspl_dB, Ptx_dBm, Gtx_dB, GT_dBK, bandwidth_Hz, alpha, eta, isend_req_dB, dt):

        Rs = bandwidth_Hz * (1 - alpha)
        Rb = Rs * eta

        # Received SNR on each tof during pass
        SNR_dB = calculate_snr(fspl_dB, Ptx_dBm, Gtx_dB, GT_dBK, bandwidth_Hz)
        ISEND_dB = calculate_isend(SNR_dB, alpha)

        # True values where there is a link
        link = (ISEND_dB >= isend_req_dB)

        link_time = np.sum(link) * dt  # Link time [s]
        throughput = link_time * Rb  # Throughput [bits/s]
        energy = 10 ** ((Ptx_dBm - 30) / 10) * (ttof[-1] - ttof[0] + dt)  # Energy [J]

        out = dict()

        # Objectives
        out['Throughput'] = -throughput
        out['Energy'] = energy

        # Meta
        return out

    def generate_non_overlapping_matrix(self, passes_df):

        # Construct the non-overlapping pass matrix
        b_tofs = passes_df.StartTof.values  # vector with pass begin tofs
        e_tofs = passes_df.StopTof.values  # vector with pass end tofs

        b_mat = np.tile(b_tofs, (len(passes_df), 1)).T
        e_mat = np.tile(e_tofs, (len(passes_df), 1)) + 1

        ovlp_mat = np.tril(e_mat - b_mat, -1)  # Only lower triangular of the matrix is used
        ovlp_mat = (ovlp_mat > 0) * 1  # The matrix is normalized to give all constraints equal importance

        return ovlp_mat

    def calculate_fspl(self, passes_df, fc):

        fspl_dB = [None] * len(passes_df)

        for i, row in passes_df.iterrows():
            fspl_dB[i] = 20 * np.log10(row.rrange) + 20 * np.log10(fc) - 147.55

        return passes_df.assign(fspl_dB=fspl_dB)



class VCMProblemPool(VCMProblem):

    def __init__(self, passes_df, modcods_df, fc, GT_dBK, design_vars):
        super().__init__(passes_df, modcods_df, fc, GT_dBK, design_vars)

    def _evaluate(self, xx, out, *args, **kwargs):

        f1 = np.zeros((len(xx), 1))  # Results of objective function 1
        f2 = np.zeros((len(xx), 1))  # Results of objective function 2

        g1 = np.zeros((len(xx), self.n_overlap_constr))

        with multiprocessing.Pool(5) as pool:
            a = pool.map(self._evaluate_design, np.split(xx, len(xx)))

        # # Process design by design
        # for i in range(len(xx)):
        #
        #     x = xx[i, :]
        #
        #     f_throughput, f_energy, g_overlap = self._evaluate_design(x)
        #
        #     f1[i] = f_throughput
        #     f2[i] = f_energy
        #     g1[i,:] = g_overlap

        # Calculate throughput must be > 0 (inversed)
        g2 = (f1 >= 0) * 1

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


class VCMProblemVarBW(Problem):
    # Class variables
    modcods_df = None
    fc = None  # Carrier frequency in [Hz]
    passes_df = None
    N = None
    overlap = None

    design_vars = None

    def __init__(self, passes_df, modcods_df, fc, GT_dBK, design_vars):

        self.modcods_df = modcods_df

        self.fc = fc
        self.GT_dBK = GT_dBK
        self.passes_df = self.calculate_fspl(passes_df, fc)
        self.N = len(self.passes_df)
        self.overlap = self.generate_non_overlapping_matrix(passes_df)

        self.n_overlap_constr = (self.N * (self.N - 1)) >> 1

        self.ttof = self.passes_df.ttof.to_list()
        self.fspl_dB = self.passes_df.fspl_dB.tolist()
        self.Ptx_dBm_set = design_vars['Ptx_dBm_set']
        self.Gtx_dBi_set = design_vars['Gtx_dBi_set']
        self.bandwidth_Hz_set = design_vars['bandwidth_Hz_set']

        # Design vector indices
        n_var = 4*self.N + 1
        self.x_indices = dict()
        self.x_indices['pass'] = slice(self.N)
        self.x_indices['power'] = slice(self.N, 2*self.N)
        self.x_indices['modcod'] = slice(2*self.N, 3*self.N)
        self.x_indices['bandwidth'] = slice(4*self.N, 5*self.N)
        self.x_indices['antenna'] = 4*self.N

        xl = np.array([0] * n_var)
        xl[self.x_indices['pass']] = 0          # Pass selection
        xl[self.x_indices['power']] = 0         # Minimum power index per pass
        xl[self.x_indices['modcod']] = 0        # Minimum MODCOD index
        xl[self.x_indices['bandwidth']] = 0     # Minimum bandwidth index
        xl[self.x_indices['antenna']] = 0       # Minimum antenna gain index

        xu = np.array([0] * n_var)
        xu[self.x_indices['pass']] = 0          # Pass selection
        xu[self.x_indices['power']] = len(self.Ptx_dBm_set) - 1             # Maximum power index per pass
        xu[self.x_indices['modcod']] = len(self.modcods_df) - 1             # Maximum MODCOD index
        xu[self.x_indices['bandwidth']] = len(self.bandwidth_Hz_set) - 1    # Maximum bandwidth index
        xu[self.x_indices['antenna']] = len(self.Gtx_dBi_set) - 1           # Maximum antenna gain index

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=self.n_overlap_constr + 1,
            xl=xl,
            xu=xu)

    def generate_default_scm(self):

        from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, \
            MixedVariableMutation
        from pymoo.factory import get_sampling, get_crossover, get_mutation

        mask = []
        mask = mask + ["bin"] * self.N  # Selection
        mask = mask + ["int"] * self.N  # Power index per pass
        mask = mask + ["int"] * self.N  # MODCOD index per pass
        mask = mask + ["int"] * self.N  # Bandwidth index per pass
        mask = mask + ["int"]  # Antenna gain index

        sampling = MixedVariableSampling(mask, {
            "bin": get_sampling("bin_random"),
            "int": get_sampling("int_random")
        })

        crossover = MixedVariableCrossover(mask, {
            "bin": get_crossover("bin_hux"),
            "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
        })

        mutation = MixedVariableMutation(mask, {
            "bin": get_mutation("bin_bitflip"),
            "int": get_mutation("int_pm", eta=3.0)
        })

        return sampling, crossover, mutation

    def _generate_pass_eval_tuples(self, xx):
        evals = [None] * self.N * len(xx)

        for i in range(len(xx)):
            for j in range(self.N):
                # These need to be unpacked to pure numba stuff
                pass_df = self.passes_df.iloc[j]
                modcod_df = xx[i, self.N + j]

                # These are ok
                Ptx_deciW = xx[i, j]
                bandwidth_Hz = self.bandwidth_Hz_set[xx[i, 2 * self.N + j]]
                Gtx_dB = self.Gtx_dBi_set[xx[i, -1]]

                evals[i * self.N + j] = (pass_df, modcod_df, Ptx_deciW, bandwidth_Hz, Gtx_dB)

        return evals

    def _evaluate(self, xx, out, *args, **kwargs):

        # --- snipp! ---#
        f1 = np.zeros((len(xx), 1))  # Results of objective function 1
        f2 = np.zeros((len(xx), 1))  # Results of objective function 2

        g1 = np.zeros((len(xx), self.n_overlap_constr))

        # Process sample by sample
        for i in range(len(xx)):

            x = xx[i, :]

            # Calculate overlap constraint
            overlap = self.overlap * x[0:self.N]
            overlap = overlap[np.tril(np.ones(overlap.shape), -1) == 1]
            g1[i, :] = overlap.flatten()


            # Process used passes
            for j in np.argwhere(x[0:self.N]).flatten().tolist():

                # Link parameters
                ttof = self.ttof[j]
                fspl_dB = self.fspl_dB[j]   # Free space path loss in dB
                Ptx_dBm = self.Ptx_dBm_set[x[self.N + j]]  # Transmit power [dBm]
                bandwidth_Hz = self.bandwidth_Hz_set[x[3 * self.N + j]]
                Gtx_dB = self.Gtx_dBi_set[x[-1]]
                GT_dBK = self.GT_dBK

                # Modcod
                alpha = 0.2
                eta = 2.6460120
                isend_req_dB = 10.69

                result = vcmproblem_evaluate_pass(
                    ttof=ttof,
                    fspl_dB=fspl_dB,
                    Ptx_dBm=Ptx_dBm,
                    Gtx_dB=Gtx_dB,
                    GT_dBK=GT_dBK,
                    bandwidth_Hz=bandwidth_Hz,
                    alpha=alpha,
                    eta=eta,
                    isend_req_dB=isend_req_dB,
                    dt=1)

                f1[i] = f1[i] + result['Throughput']  # Throughput
                f2[i] = f2[i] + result['Energy']  # Energy

        # Calculate throughput must be > 0 (inversed)
        g2 = (f1 >= 0) * 1

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

    def generate_non_overlapping_matrix(self, passes_df):

        # Construct the non-overlapping pass matrix
        b_tofs = passes_df.StartTof.values  # vector with pass begin tofs
        e_tofs = passes_df.StopTof.values  # vector with pass end tofs

        b_mat = np.tile(b_tofs, (len(passes_df), 1)).T
        e_mat = np.tile(e_tofs, (len(passes_df), 1)) + 1

        ovlp_mat = np.tril(e_mat - b_mat, -1)  # Only lower triangular of the matrix is used
        ovlp_mat = (ovlp_mat > 0) * 1  # The matrix is normalized to give all constraints equal importance

        return ovlp_mat

    def calculate_fspl(self, passes_df, fc):

        fspl_dB = [None] * len(passes_df)

        for i, row in passes_df.iterrows():
            fspl_dB[i] = 20 * np.log10(row.rrange) + 20 * np.log10(fc) - 147.55

        return passes_df.assign(fspl_dB=fspl_dB)

def vcmproblem_evaluate_pass(ttof, fspl_dB, Ptx_dBm, Gtx_dB, GT_dBK, bandwidth_Hz, alpha, eta, isend_req_dB, dt):

    kB_dB = 10 * np.log10(bandwidth_Hz * 1.380649e-23)  # k*B in dB
    Rs = bandwidth_Hz * (1 - alpha)
    Rb = Rs * eta

    # Received SNR on each tof during pass
    SNR_dB = Ptx_dBm + Gtx_dB - fspl_dB + GT_dBK - kB_dB - 30.
    isend_dB = SNR_dB - 10 * np.log10(1 - alpha)  # ISEND on each tof during pass

    # True values where there is a link
    link = (isend_dB >= isend_req_dB)

    link_time = np.sum(link) * dt  # Link time [s]
    throughput = link_time * Rb  # Throughput [bits/s]
    energy = 10 ** ((Ptx_dBm - 30) / 10) * (ttof[-1] - ttof[0] + dt)  # Energy [J]

    out = dict()

    # Objectives
    out['Throughput'] = -throughput
    out['Energy'] = energy

    # Meta
    return out


