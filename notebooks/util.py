import numpy as np
import matplotlib.pyplot as plt

import contact
import energy
import multi_carrier
from pyreport import PlotUtil


def print_targets(case):
    print("Targets per orbit")
    print("T_target: %.2f Gb (min: %.2f Gb, max: %.2f Gb)" %
          (case['T_bitorbit_target'] / 1e9, case['T_bitorbit_min'] / 1e9, case['T_bitorbit_max'] / 1e9))
    print("L_target: %.2f s" % (case['L_sorbit_target']))
    print("E_target: %.2f kJ (max: %.2f kJ)" % (case['E_Jorbit_target'] / 1e3, case['E_Jorbit_max'] / 1e3))
    print("P_target: %.2f s" % (case['P_sorbit_target']))

def get_selection(problem, res):
    x_pass = res.X[:, problem.x_indices['pass']].astype('bool')
    for i in range(x_pass.shape[0]):
        x_pass[i, :] = contact.down_select_passes(x_pass[i, :], problem.O_matrix)

    # Transmit powers
    x_Ptx_dBm = res.X[:, problem.x_indices['power']].astype('float64')
    x_Ptx_dBm[~x_pass] = np.NaN

    # Antenna gain
    x_Gtx_dBi = np.squeeze(res.X[:, problem.x_indices['antenna']].astype('float64'))

    # Bandwidth
    x_B_Hz = np.squeeze(res.X[:, problem.x_indices['bandwidth']].astype('int64'))
    x_B_Hz = problem.sys_param.B_Hz_array[x_B_Hz]

    return x_pass, x_Ptx_dBm, x_Gtx_dBi, x_B_Hz


def recompute_obj(problem, res, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):
    # Recompute all four objectives
    f_pointing = np.empty_like(res.F[:, 0])
    f_energy = np.empty_like(res.F[:, 0])
    f_latency = np.empty_like(res.F[:, 0])
    f_throughput = np.empty_like(res.F[:, 0])

    for i in range(len(f_throughput)):
        ff, gg = problem.evaluate_unmasked(res.X[i, :])
        ff = ff * scale_factors
        f_pointing[i] = ff[3]  # s
        f_energy[i] = ff[2]  # Kilo Joule
        f_latency[i] = ff[1]  # s
        f_throughput[i] = ff[0]  # Gigabit

    return f_throughput, f_latency, f_energy, f_pointing


def recompute_all(problem, res, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):
    # Recompute all four objectives
    f_pointing = np.empty_like(res.F[:, 0])
    f_energy = np.empty_like(res.F[:, 0])
    f_latency = np.empty_like(res.F[:, 0])
    f_throughput = np.empty_like(res.F[:, 0])

    vcm_array_list = [None] * len(f_throughput)

    for i in range(len(f_throughput)):
        T, L, E, P, \
        b_s_array, e_s_array, linktime_s_array, vcm_array = problem.evaluate_unmasked_raw(res.X[i, :])

        ff = np.array([-1 * T, L, E, P]) * scale_factors
        f_pointing[i] = ff[3]  # s
        f_energy[i] = ff[2]  # Kilo Joule
        f_latency[i] = ff[1]  # s
        f_throughput[i] = ff[0]  # Gigabit

        vcm_array_list[i] = vcm_array

    return f_throughput, f_latency, f_energy, f_pointing, vcm_array_list


## Plotting
def plot_performance(axs, problem, setting, res, case, target, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):
    f_throughput, f_latency, f_energy, f_pointing = recompute_obj(problem, res, scale_factors)

    # Plotting
    # axs = axs.flatten()

    ax = axs[0]
    ax.grid(True)
    ax.scatter(f_energy, f_throughput, marker='.', s=1)
    ax.set_xlabel("Energy used [kJ / orbit]")
    ax.set_ylabel("Throughput [GB / orbit]")
    ax.set_axisbelow(True)

    ax = axs[1]
    ax.grid(True)
    ax.scatter(f_pointing, f_throughput, marker='.', s=1)
    ax.set_xlabel("Pointing duty cycle [%]")
    ax.set_ylabel("Throughput [GB / orbit]")
    ax.set_axisbelow(True)

    ax = axs[2]
    ax.grid(True)
    ax.scatter(f_energy, f_pointing, marker='.', s=1)
    ax.set_xlabel("Energy used [kJ / orbit]")
    ax.set_ylabel("Pointing duty cycle [%]")
    ax.set_axisbelow(True)

    # ax = axs[3]
    # ax.grid(True)
    # ax.scatter(f_energy, f_latency, marker='.', s=1)
    # ax.set_xlabel("Energy used [kJ / orbit]")
    # ax.set_ylabel("Avg. Latency [s]")
    # ax.set_axisbelow(True)
    #
    # ax = axs[4]
    # ax.grid(True)
    # ax.scatter(f_pointing, f_latency, marker='.', s=1)
    # ax.set_xlabel("Pointing duty cycle [%]")
    # ax.set_ylabel("Avg. Latency [s]")
    # ax.set_axisbelow(True)
    #
    # ax = axs[5]
    # ax.grid(True)
    # ax.scatter(f_latency, f_throughput, marker='.', s=1)
    # ax.set_xlabel("Avg. Latency [s]")
    # ax.set_ylabel("Throughput [GB / orbit]")
    # ax.set_axisbelow(True)
    #
    # axs[6].set_axis_off()
    # axs[7].set_axis_off()

    axs[3].set_axis_off()
    axs[4].set_axis_off()
    ax = axs[5]
    ax.legend(fontsize=8)
    ax.set_axis_off()

    plt.tight_layout()

def plot_performance_eo(axs, problem, setting, res, case, target, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):
    f_throughput, f_latency, f_energy, f_pointing = recompute_obj(problem, res, scale_factors)

    # Plotting
    # axs = axs.flatten()

    ax = axs[0]
    ax.grid(True)
    ax.scatter(f_energy, f_throughput, marker='.', s=1)
    ax.set_xlabel("Energy used [kJ / orbit]")
    ax.set_ylabel("Throughput [GB / orbit]")
    ax.set_axisbelow(True)

    ax = axs[1]
    ax.grid(True)
    ax.scatter(f_pointing, f_throughput, marker='.', s=1)
    ax.set_xlabel("Pointing duty cycle [%]")
    ax.set_ylabel("Throughput [GB / orbit]")
    ax.set_xlim([0, 100])
    ax.set_axisbelow(True)

    ax = axs[2]
    ax.grid(True)
    ax.scatter(f_energy, f_pointing, marker='.', s=1)
    ax.set_xlabel("Energy used [kJ / orbit]")
    ax.set_ylabel("Pointing duty cycle [%]")
    ax.set_ylim([0, 100])
    ax.set_axisbelow(True)

    axs[3].set_axis_off()
    axs[4].set_axis_off()
    ax = axs[5]
    ax.legend(fontsize=8)
    ax.set_axis_off()

    plt.tight_layout()

def plot_performance_iot(axs, problem, setting, res, case, target, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):
    f_throughput, f_latency, f_energy, f_pointing = recompute_obj(problem, res, scale_factors)

    ax = axs[0]
    ax.grid(True)
    ax.scatter(f_energy, f_latency, marker='.', s=1)
    ax.set_xlabel("Energy used [kJ / orbit]")
    ax.set_ylabel("Max Latency [s]")
    ax.set_axisbelow(True)

    ax = axs[1]
    ax.grid(True)
    ax.scatter(f_pointing, f_latency, marker='.', s=1)
    ax.set_xlabel("Pointing duty cycle [%]")
    ax.set_ylabel("Max Latency [s]")
    ax.set_xlim([0, 100])
    ax.set_axisbelow(True)

    ax = axs[2]
    ax.grid(True)
    ax.scatter(f_energy, f_pointing, marker='.', s=1)
    ax.set_xlabel("Energy used [kJ / orbit]")
    ax.set_ylabel("Pointing duty cycle [%]")
    ax.set_axisbelow(True)
    ax.set_ylim([0, 100])

    ax = axs[3] #axs[3].set_axis_off()
    ax.grid(True)
    ax.scatter(f_energy, f_throughput, marker='.', s=1)
    ax.set_xlabel("Energy used [kJ / orbit]")
    ax.set_ylabel("Throughput [GB / orbit]")
    ax.set_axisbelow(True)

    ax = axs[4] #axs[4].set_axis_off()
    ax.grid(True)
    ax.scatter(f_latency, f_throughput, marker='.', s=1)
    ax.set_xlabel("Max latency [s]")
    ax.set_ylabel("Throughput [GB / orbit]")
    ax.set_axisbelow(True)

    ax = axs[5]
    ax.legend(fontsize=8)
    ax.set_axis_off()

    plt.tight_layout()

def plot_points_iot(axs, points, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):

    for point in points:
        p = point['p'] * scale_factors
        axs[0].scatter(p[2], p[1], *point['args'], **point['kwargs'])
        axs[1].scatter(p[3], p[1], *point['args'], **point['kwargs'])
        axs[2].scatter(p[2], p[3], *point['args'], **point['kwargs'])
        axs[5].scatter(np.NaN, np.NaN, *point['args'], **point['kwargs'])

    axs[5].legend(fontsize=8)

def plot_points_eo(axs, points, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):

    for point in points:
        p = point['p'] * scale_factors
        axs[0].scatter(p[2], p[0], *point['args'], **point['kwargs'])
        axs[1].scatter(p[3], p[0], *point['args'], **point['kwargs'])
        axs[2].scatter(p[2], p[3], *point['args'], **point['kwargs'])
        axs[5].scatter(np.NaN, np.NaN, *point['args'], **point['kwargs'])

    axs[5].legend(fontsize=8)

def plot_settings(axs, problem, setting, res, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):

    x_pass, x_Ptx_dBm, x_Gtx_dBi, x_B_Hz = get_selection(problem, res)
    f_throughput, f_latency, f_energy, f_pointing, vcm_array_list = recompute_all(problem, res, scale_factors)

    # Passes uses
    ax = axs[0]
    ax.scatter(np.sum(x_pass, axis=1), f_throughput, marker='.', s=1)
    ax.set_xlabel('Passes used')
    ax.set_ylabel('Throughput [GB / orbit]')

    # Maximum Tx power
    ax = axs[1]
    ax.scatter(np.nanmax(x_Ptx_dBm, axis=1), f_throughput, marker='.', s=1)
    ax.set_xlabel('$P_{tx, max}$ [dBm]')
    ax.set_ylabel('Throughput [GB / orbit]')

    # Antenna gain
    ax = axs[2]
    ax.scatter(x_Gtx_dBi, f_throughput, marker='.', s=1)
    ax.set_xlabel('$G_{tx}$ [dBi]')
    ax.set_ylabel('Throughput [GB / orbit]')

    # Bandwidth
    ax = axs[3]
    ax.scatter(x_B_Hz / 1e6, f_throughput, marker='.', s=1)
    ax.set_xlabel('B [MHz]')
    ax.set_ylabel('Throughput [GB / orbit]')
    ax.set_xlim(0, 310)

    # VCM
    ax = axs[4]
    for i in range(len(f_throughput)):
        ax.scatter(vcm_array_list[i], f_throughput[i].repeat(len(vcm_array_list[i])), color='tab:blue', marker='.', s=1)
    ax.set_xlabel('MODCODs used [.]')
    ax.set_ylabel('Throughput [GB / orbit]')

    # Maximum power dissipated
    ax = axs[5]
    for i in range(len(f_throughput)):
        B_Hz = x_B_Hz[i]
        alpha = problem.sys_param.alpha_array[0]

        carriers = multi_carrier.get_sub_carriers(B_Hz)
        eta_bitsym_array = np.squeeze(problem.sys_param.eta_bitsym_array[:, carriers - 1])
        eta_maee_array = np.squeeze(problem.sys_param.eta_maee_array[:, carriers - 1])

        Ppa = energy.pa_power(x_Ptx_dBm[i, x_pass[i, :]], eta_maee_array[vcm_array_list[i]])

        Rs_syms = x_B_Hz[i] / (1 + alpha)  # Symbol rate in symbols per second
        Rb_bits_array = Rs_syms * eta_bitsym_array[vcm_array_list[i]]  # Data rate in bits per second

        Pmod = energy.modulator_power(Rb_bits_array)
        Pdiss = Pmod + Ppa

        ax.scatter(Pdiss, f_throughput[i].repeat(len(vcm_array_list[i])), color='tab:blue', marker='.', s=1)

    ax.set_xlabel('Power disspated [W]')
    ax.set_ylabel('Throughput [GB / orbit]')

    plt.tight_layout()


def plot_used_passes(case, instances_df, x_pass):

    pass_ind = np.nonzero(x_pass)[0]

    T_orbit = case['T_orbit_s']
    T_sim = case['T_sim_s']

    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    # Plot all passes
    for i, pass_df in instances_df.groupby(level=0):
        p = pass_df.index[0][0] - 1 # Pass index
        tof = pass_df.tof.values
        d = pass_df.d.values/1000
        line, = ax.plot(tof, d, linewidth=0.1, color='tab:grey')

    # Plot used passes
    for i, pass_df in instances_df.groupby(level=0):
        p = pass_df.index[0][0] - 1  # Pass index
        tof = pass_df.tof
        d = pass_df.d / 1000
        if p in pass_ind:
            line, = ax.plot(tof, d, linewidth=0.5, color='tab:red')

    ax.set_xlabel('Time of flight [s]')
    ax.set_ylabel('Range [km]')
    ax.set_xlim((0, T_sim))

    plt.grid()
    PlotUtil.apply_report_formatting()
    fig.set_size_inches(3.2 * 2, 2.4, forward=True)

    plt.tight_layout()