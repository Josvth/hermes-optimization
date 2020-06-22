from pymoo.model.problem import Problem
import numpy as np


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
    fc = None           # Carrier frequency in [Hz]
    passes_df = None
    N = None
    overlap = None

    design_vars = None

    def __init__(self, passes_df, modcods_df, fc, GT_dBK, design_vars):

        self.modcods_df = modcods_df

        self.fc = fc
        self.GT_dBK = GT_dBK
        self.passes_df = self.calculate_fspl(passes_df, fc)
        self.passes_df_list = self.passes_df.iterrows()
        self.N = len(self.passes_df)
        self.overlap = self.generate_non_overlapping_matrix(passes_df)

        self.n_overlap_constr = (self.N * (self.N - 1)) >> 1

        self.Ptx_deciW_set = design_vars['Ptx_deciW_set']
        self.Gtx_dBi_set = design_vars['Gtx_dBi_set']
        self.bandwidth_Hz_set = design_vars['bandwidth_Hz_set']

        xl = []
        xl = xl + [self.Ptx_deciW_set[0]] * self.N        # Minimum power per pass
        xl = xl + [0] * self.N                     # Minimum MODCOD setting = 0
        xl = xl + [0] * self.N                     # Minimum bandwidth setting = 0
        xl = xl + [0]                              # Minimum antenna gain setting = 0

        xu = []
        xu = xu + [self.Ptx_deciW_set[-1]] * self.N               # Maximum power per pass
        xu = xu + [len(self.modcods_df) - 1] * self.N        # Maximum MODCOD setting = Nmodcod - 1
        xu = xu + [len(self.bandwidth_Hz_set) - 1] * self.N    # Maximum bandwidth setting = Nbandwidth - 1
        xu = xu + [len(self.Gtx_dBi_set) - 1]                # Maximum antenna gain setting = Nantenna - 1

        super().__init__(
            n_var=3*self.N + 1,
            n_obj=2,
            n_constr=self.n_overlap_constr + 1,
            xl=xl,
            xu=xu)

    def generate_default_scm(self):

        from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableCrossover, MixedVariableMutation
        from pymoo.factory import get_sampling, get_crossover, get_mutation

        mask = []
        mask = mask + ["int"] * self.N  # Power deciWatt per pass
        mask = mask + ["int"] * self.N  # MODCOD index per pass
        mask = mask + ["int"] * self.N  # Bandwidth index per pass
        mask = mask + ["int"]           # Antenna gain index

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

                evals[i*self.N + j] = (pass_df, modcod_df, Ptx_deciW, bandwidth_Hz, Gtx_dB)

        return evals

    def _evaluate(self, xx, out, *args, **kwargs):

        # --- snipp! ---#
        f1 = np.zeros((len(xx), 1))  # Results of objective function 1
        f2 = np.zeros((len(xx), 1))  # Results of objective function 2

        g1 = np.zeros((len(xx), self.n_overlap_constr))
        g2 = np.zeros((len(xx), 1))

        # Generate evaluations
        evals = self._generate_pass_eval_tuples(xx)

        # Parallelize population
        with multiprocessing.Pool(8) as p:
            results = p.starmap(vcmproblem_evaluate_pass, evals)

        # Unpack results
        for i in range(len(results)):
            n_pop = int(i / self.N) # population number of result
            n_pass = i % self.N     # Pass number of result

            f1[n_pop] = f1[n_pop] + results[i]['Throughput']   # Throughput
            f2[n_pop] = f2[n_pop] + results[i]['Energy']   # Energy

        # Calculate non-overlap constraint
        for i in range(len(xx)):
            overlap = self.overlap * (xx[i,0:self.N] > 1)
            overlap = overlap[np.tril(np.ones(overlap.shape), -1) == 1]
            g1[i,:] = overlap.flatten()

        # Calculate throughput must be > 0 (inversed)
        g2 = f1 > 0

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

    # def evaluate_passes(self, x):
    #     xpass = x[0:-1]  # Binary variables that select passes
    #     xpower = x[-1]  # Integer variable that selects power
    #
    #     # Calculate contact time
    #     b_tofs = self.passes_df.StartTof[xpass].values
    #     e_tofs = self.passes_df.StopTof[xpass].values
    #     contact_time = np.sum(e_tofs + 1 - b_tofs)
    #
    #     # Calculate overlap constraints
    #     overlap = self.O * xpass
    #
    #     # Calculate power consumption
    #     power_consumption = contact_time * 10 ** ((self.Ptx_dBm[xpower] - 30) / 10)
    #
    #     # Link budget calculation for each pass
    #     isend_dB_list = [None] * len(self.passes_df)
    #
    #     total_link_time = 0
    #     for i, pass_df in self.passes_df.loc[xpass].iterrows():
    #         d = np.linalg.norm(pass_df.sff, axis=1)  # Range at each tof during pass
    #         Ptx_dBm = self.Ptx_dBm[xpower]  # Transmit power
    #         Gtx_dB = self.Gtx_dB  # Transmitter gain
    #         fspl_dB = 20 * np.log10(d) + 20 * np.log10(self.fc) - 147.55  # Free space path loss
    #         GT_dB = self.GT  # G/T figure
    #         kB_dB = 10 * np.log10(self.B * 1.380649e-23)  # k*B in dB
    #
    #         # Received SNR on each tof during pass
    #         SNR_dB = Ptx_dBm + Gtx_dB - fspl_dB + GT_dB - kB_dB - 30.
    #         isend_dB_list[i] = SNR_dB - 10 * np.log10(self.Rs / self.B)  # ISEND on each tof during pass
    #
    #         # Link time where the ISEND is higher than the requirement
    #         link_time = np.sum(isend_dB_list[i] >= self.isend_req)
    #
    #         total_link_time = total_link_time + link_time
    #
    #     return contact_time, overlap, power_consumption, total_link_time, isend_dB_list

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
            fspl_dB[i] = 20*np.log10(row.rrange) + 20*np.log10(fc) + 147.55

        return passes_df.assign(fspl_dB = fspl_dB)



def vcmproblem_evaluate_pass(pass_df, modcod_df, Ptx, bandwidth, Gtx_dB):

    out = dict()

    out["Throughput"] = 0
    out["Energy"] = 0

    return out