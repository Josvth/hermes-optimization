import numpy as np
import matplotlib.pyplot as plt

def power_dissipation_profile(VkVs, N = 1000):

    gamma = 0.5*(VkVs + 1)

    AmaxVs = 1 - gamma

    a = np.linspace(-AmaxVs, AmaxVs, N)
    Pdiss = gamma*(1- gamma) - (a**2)/2

    # a = np.concatenate((-1*np.flip(a[1:]), a[1:]))
    # Pdiss = np.concatenate((np.flip(Pdiss[1:]), Pdiss[1:]))

    return a, Pdiss, gamma

def plot_dissipation_profile(a, Pdiss, gamma):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(a, Pdiss)
    ax.set_xlabel('a [V]')
    ax.set_ylabel('$P_D(a, \gamma)$')
    ax.grid(True)

    def to_v(a):
        return (a + gamma) * Vs

    def to_a(v):
        return v/Vs - gamma

    secax_x2 = ax.secondary_xaxis(-0.25, functions=(to_v, to_a))
    secax_x2.set_xlabel('Drain/Collector voltage [V]')
    secax_x2.set_xlim((0, Vs))


def maee(symbols, a, disspation_profile):

    mag = np.concatenate((np.abs(symbols), -1.0*np.abs(symbols)))

    pdf_maee, edge_maee = np.histogram(mag, bins=len(disspation_profile), density=1)
    x_maee = edge_maee[:-1]
    pdf_maee = pdf_maee*np.diff(edge_maee)

    EPD = np.sum(pdf_maee * disspation_profile)
    Ptx = (np.max(a)**2 / 2)

    eta_maee = Ptx / (Ptx + EPD)

    return eta_maee

def papr(symbols):
    return np.max(np.abs(symbols))**2 / np.mean(np.abs(symbols))**2

