import numpy as np
import matplotlib.pyplot as plt
from pymoo.model.result import Result

import contact
import energy
import multi_carrier
import visibility
from pyreport import PlotUtil


def print_targets(case):
    print("Targets per orbit")
    print("T_target: %.2f Gb (min: %.2f Gb, max: %.2f Gb)" %
          (case['T_bitorbit_target'] / 1e9, case['T_bitorbit_min'] / 1e9, case['T_bitorbit_max'] / 1e9))
    print("L_target: %.2f s" % (case['L_sorbit_target']))
    print("E_target: %.2f kJ (max: %.2f kJ)" % (case['E_Jorbit_target'] / 1e3, case['E_Jorbit_max'] / 1e3))
    print("P_target: %.2f s" % (case['P_sorbit_target']))

def get_selection(problem, X):
    if isinstance(X, Result): # Catch for older functions todo deprecate
        import warnings
        warnings.warn("Dict functionality will be deprecated")
        X = X.X

    if X.ndim == 1:
        X = X[np.newaxis, :]

    x_pass = X[:, problem.x_indices['pass']].astype('bool')
    for i in range(x_pass.shape[0]):
        x_pass[i, :] = contact.down_select_passes(x_pass[i, :], problem.O_matrix)

    # Transmit powers
    x_Ptx_dBm = X[:, problem.x_indices['power']].astype('float64')
    x_Ptx_dBm[~x_pass] = np.NaN

    # Antenna gain
    x_Gtx_dBi = np.squeeze(X[:, problem.x_indices['antenna']].astype('float64'))

    # Bandwidth
    x_B_Hz = np.squeeze(X[:, problem.x_indices['bandwidth']].astype('int64'))
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


def recompute_all(problem, X, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):
    if isinstance(X, Result): # Catch for older functions todo deprecate
        import warnings
        warnings.warn("Dict functionality will be deprecated")
        X = X.X

    if X.ndim == 1:
        X = X[np.newaxis, :]

    # Recompute all four objectives
    f_pointing = np.empty(len(X))
    f_energy = np.empty(len(X))
    f_latency = np.empty(len(X))
    f_throughput = np.empty(len(X))

    vcm_array_list = [None] * len(X)

    for i in range(len(X)):
        T, L, E, P, \
        b_s_array, e_s_array, linktime_s_array, vcm_array = problem.evaluate_unmasked_raw(X[i, :])

        ff = np.array([-1 * T, L, E, P]) * scale_factors
        f_pointing[i] = ff[3]  # s
        f_energy[i] = ff[2]  # Kilo Joule
        f_latency[i] = ff[1]  # s
        f_throughput[i] = ff[0]  # Gigabit

        vcm_array_list[i] = vcm_array

    return f_throughput, f_latency, f_energy, f_pointing, vcm_array_list

## Plotting
def plot_performance(axs, problem, setting, res, case, target, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):
    plot_performance_eo(axs, problem, setting, res, case, target, scale_factors)

def plot_performance_eo(axs, problem, setting, res, case, target, scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1]), plot_i=False):
    f_throughput, f_latency, f_energy, f_pointing = recompute_obj(problem, res, scale_factors)

    # Plotting
    # axs = axs.flatten()

    ax = axs[0]
    ax.grid(True)
    ax.scatter(f_energy, f_throughput, marker='.', s=1)
    if plot_i:
        for i in range(len(f_energy)):
            ax.text(f_energy[i], f_throughput[i], f'{i}', fontsize=6)
    ax.set_xlabel("Energy used [kJ / orbit]")
    ax.set_ylabel("Throughput [GB / orbit]")
    ax.set_axisbelow(True)

    ax = axs[1]
    ax.grid(True)
    ax.scatter(f_pointing, f_throughput, marker='.', s=1)
    if plot_i:
        for i in range(len(f_energy)):
            ax.text(f_pointing[i], f_throughput[i], f'{i}', fontsize=6)
    ax.set_xlabel("Pointing duty cycle [%]")
    ax.set_ylabel("Throughput [GB / orbit]")
    ax.set_xlim([0, 100])
    ax.set_axisbelow(True)

    ax = axs[2]
    ax.grid(True)
    ax.scatter(f_energy, f_pointing, marker='.', s=1)
    if plot_i:
        for i in range(len(f_energy)):
            ax.text(f_energy[i], f_pointing[i], f'{i}', fontsize=6)
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

def plot_performance_iot(axs, problem, setting, res, case, target,
                         scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1]), throughput=True):
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

    if throughput:
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
    else:
        axs[3].set_axis_off()
        axs[4].set_axis_off()

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

def plot_settings(axs, problem, setting, res, points = [], scale_factors=np.array([1 / -1e9, 1, 1 / 1e3, 1])):

    x_pass, x_Ptx_dBm, x_Gtx_dBi, x_B_Hz = get_selection(problem, res.X)
    f_throughput, f_latency, f_energy, f_pointing, vcm_array_list = recompute_all(problem, res.X, scale_factors)

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
    Pdiss = [None] * len(f_throughput)

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
        Pdiss[i] = Pmod + Ppa

        ax.scatter(Pdiss[i], f_throughput[i].repeat(len(Pdiss[i])), color='tab:blue', marker='.', s=1)

    ax.set_xlabel('Power disspated [W]')
    ax.set_ylabel('Throughput [GB / orbit]')

    plt.tight_layout()

    for point in points:
        if point['ind'] >= 0:
            ind = point['ind']
            axs[0].scatter(np.sum(x_pass[ind, :]), f_throughput[ind], marker='.', s=1, color=point['kwargs']['color'])
            axs[1].scatter(np.nanmax(x_Ptx_dBm[ind, :]), f_throughput[ind], marker='.', s=1, color=point['kwargs']['color'])
            axs[2].scatter(x_Gtx_dBi[ind], f_throughput[ind], marker='.', s=1, color=point['kwargs']['color'])
            axs[3].scatter(x_B_Hz[ind] / 1e6, f_throughput[ind], marker='.', s=1, color=point['kwargs']['color'])
            axs[4].scatter(vcm_array_list[ind], f_throughput[ind].repeat(len(vcm_array_list[ind])), color=point['kwargs']['color'], marker='.', s=1)
            axs[5].scatter(Pdiss[ind], f_throughput[ind].repeat(len(Pdiss[ind])), color=point['kwargs']['color'], marker='.', s=1)

def plot_used_passes(case, instances_df, problem, x, usefull_only = True):

    x_pass, x_Ptx_dBm, x_Gtx_dBi, x_B_Hz = get_selection(problem, x)

    pass_inds = np.nonzero(x_pass[0])[0]

    if usefull_only:
        _, _, _, _, _, _, linktime_s_array, _ = problem.evaluate_unmasked_raw(x)
        # Filter out passes that don't contribute (i.e. zero link time) todo fix this in the optimization
        pass_inds = pass_inds[np.nonzero(linktime_s_array)[0]]

    T_orbit = case['T_orbit_s']
    T_sim = case['T_sim_s']

    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    # Plot all passes
    for i, pass_df in instances_df.groupby(level=0):
        p = pass_df.index[0][0] - 1 # Pass index
        tof = pass_df.tof.values
        d = pass_df.d.values/1000
        line, = ax.plot(tof, d, linewidth=0.1, color='tab:grey', alpha=0.5)

    # Plot used passes
    for i, pass_df in instances_df.groupby(level=0):
        p = pass_df.index[0][0] - 1  # Pass index
        tof = pass_df.tof
        d = pass_df.d / 1000
        if p in pass_inds:
            line, = ax.plot(tof, d, linewidth=0.5, color='tab:red')

    ax.set_xlabel('Time of flight [s]')
    ax.set_ylabel('Range [km]')
    ax.set_xlim((0, T_sim))

    plt.grid()
    PlotUtil.apply_report_formatting()
    fig.set_size_inches(3.2 * 2, 2.4, forward=True)

    plt.tight_layout()

def plot_power_energy(case, instances_df, problem, x, dt=1.0):

    T_orbit = case['T_orbit_s']
    T_sim = case['T_sim_s']

    all_tofs_s = np.arange(0, T_sim, dt)

    x_pass, x_Ptx_dBm, x_Gtx_dBi, x_B_Hz = get_selection(problem, x)
    T, L, E, P, b_s_array, e_s_array, linktime_s_array, vcm_array = problem.evaluate_unmasked_raw(x)

    # Compute power disspations
    B_Hz = x_B_Hz
    alpha = problem.sys_param.alpha_array[0]

    carriers = multi_carrier.get_sub_carriers(B_Hz)
    eta_bitsym_array = np.squeeze(problem.sys_param.eta_bitsym_array[:, carriers - 1])
    eta_maee_array = np.squeeze(problem.sys_param.eta_maee_array[:, carriers - 1])

    Ppa = energy.pa_power(x_Ptx_dBm[x_pass], eta_maee_array[vcm_array])

    Rs_syms = x_B_Hz / (1 + alpha)  # Symbol rate in symbols per second
    Rb_bits_array = Rs_syms * eta_bitsym_array[vcm_array]  # Data rate in bits per second

    Pmod = energy.modulator_power(Rb_bits_array)
    Pdiss = Pmod + Ppa

    print("Max power: %0.2f W" % np.max(Pdiss))
    print("Bandwidth: %.2f MHz" % (x_B_Hz / 1e6))
    print("Max modcod: %d " % np.max(vcm_array))

    # Reconstruct time of flight vectors
    tofs_s = [None] * len(b_s_array)

    for i in range(len(tofs_s)):
        tofs_s[i] = np.arange(b_s_array[i], e_s_array[i], dt)

    # Plotting
    fig, ax = plt.subplots(figsize=(3.2*2, 2.4))

    # Plot dissipation over time
    Pdiss_vs_tof = np.zeros_like(all_tofs_s)
    for i in range(len(tofs_s)):
        Pdiss_vs_tof[tofs_s[i].astype('int')] = Pdiss[i]
    ax.step(all_tofs_s, Pdiss_vs_tof, linewidth=0.75, color='coral')

    PlotUtil.apply_report_formatting()

    # Plot energy over time
    energyJ = np.cumsum(Pdiss_vs_tof) * dt
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.step(all_tofs_s, energyJ / 1e3, linewidth=1, color='tab:blue')

    ax.grid()
    ax.set_axisbelow(True)
    ax2.set_axisbelow(True)

    ax.set_xlabel('Time of flight [s]')
    ax.set_ylabel('Power [W]')
    ax2.set_ylabel('Energy [kJ]')

    ax.set_xlim((0, T_sim))

    PlotUtil.apply_report_formatting()

    fig.set_size_inches(3.2 * 2, 2.4, forward=True)

    plt.tight_layout()

def plot_pointing(case, problem, x):

    x_pass, x_Ptx_dBm, x_Gtx_dBi, x_B_Hz = get_selection(problem, x)
    _, _, _, _, _, _, linktime_s_array, _ = problem.evaluate_unmasked_raw(x)

    pass_inds = np.nonzero(x_pass[0])[0]
    # Filter out passes that don't contribute (i.e. zero link time) todo fix this in the optimization
    pass_inds = pass_inds[np.nonzero(linktime_s_array)[0]]

    hpbw_rad = visibility.compute_hpbw(x_Gtx_dBi)

    T_orbit = case['T_orbit_s']
    T_sim = case['T_sim_s']

    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    # Plot pointing angles
    for pass_ind in pass_inds:
        #ax.plot(problem.tof_s_list[pass_ind], np.rad2deg(problem.theta_rad_list[pass_ind]), linewidth=0.1, color='tab:grey')
        pointing = np.maximum(np.rad2deg(problem.theta_rad_list[pass_ind] - 0.5 * hpbw_rad), 0.0)
        ax.plot(problem.tof_s_list[pass_ind], pointing, linewidth=0.75, color='tab:blue')

    ax.text(0.9, 0.9, '$G_{tx,0}$: %0.2f dBi, $\\theta_{hpbw}$: %0.2f°' % (x_Gtx_dBi, np.rad2deg(hpbw_rad)), transform=ax.transAxes, fontsize=7,
            horizontalalignment='right',
            verticalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black'))

    ax.set_xlabel('Time of flight [s]')
    ax.set_ylabel('Pointing from zenith [deg]')
    ax.set_xlim((0, T_sim))
    ax.set_ylim((0, 90))

    plt.grid()
    PlotUtil.apply_report_formatting()
    fig.set_size_inches(3.2 * 2, 2.4, forward=True)

    plt.tight_layout()
