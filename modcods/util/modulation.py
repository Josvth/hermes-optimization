import numpy as np

def generate_apsk_symbols(modcod_df):

    m1, m2, m3 = modcod_df.m1, modcod_df.m2, modcod_df.m3
    total_symbols = m1 + m2 + m3

    symbols = np.zeros(total_symbols, dtype=np.complex_)

    s = 0   # symbol counter

    g1 = modcod_df.gamma1
    g2 = modcod_df.gamma2
    g23 = 1.0 if g2 == 1.0 else g2 / g1

    r = (1/g1) * (1/g23) * 1
    phi = np.deg2rad(modcod_df.phi1)
    m = m1
    for j in range(m1):
        symbols[s] = r * np.exp(1j*(phi + j*(2*np.pi)/m))
        s = s + 1

    r = (1/g23) * 1
    phi = np.deg2rad(modcod_df.phi2)
    m = m2
    for j in range(m2):
        symbols[s] = r * np.exp(1j*(phi + j*(2*np.pi)/m))
        s = s + 1

    r = 1
    phi = np.deg2rad(modcod_df.phi3)
    m = m3
    for j in range(m3):
        symbols[s] = r * np.exp(1j*(phi + j*(2*np.pi)/m))
        s = s + 1

    grey_mapping = np.zeros(total_symbols)
    for s in range(total_symbols):
        grey_mapping[s] = modcod_df['g%d' % s]

    return symbols, grey_mapping

def grey_code(n):
    return n ^ (n >> 1)

def raised_root_cosine(upsample, num_positive_lobes, alpha):
    """
    Root raised cosine (RRC) filter (FIR) impulse response.

    upsample: number of samples per symbol

    num_positive_lobes: number of positive overlaping symbols
    length of filter is 2 * num_positive_lobes + 1 samples

    alpha: roll-off factor
    """

    N = upsample * (num_positive_lobes * 2 + 1)
    t = (np.arange(N) - N / 2) / upsample

    # result vector
    h_rrc = np.zeros(t.size, dtype=np.float)

    # index for special cases
    sample_i = np.zeros(t.size, dtype=np.bool)

    # deal with special cases
    subi = t == 0
    sample_i = np.bitwise_or(sample_i, subi)
    h_rrc[subi] = 1.0 - alpha + (4 * alpha / np.pi)

    subi = np.abs(t) == 1 / (4 * alpha)
    sample_i = np.bitwise_or(sample_i, subi)
    h_rrc[subi] = (alpha / np.sqrt(2)) \
                * (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))))
                + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))

    # base case
    sample_i = np.bitwise_not(sample_i)
    ti = t[sample_i]
    h_rrc[sample_i] = np.sin(np.pi * ti * (1 - alpha)) \
                    + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
    h_rrc[sample_i] /= (np.pi * ti * (1 - (4 * alpha * ti) ** 2))

    return t, h_rrc

def modulate(symbols, grey_mapping, bit_sequence):

    mod_order = int(np.log2(len(symbols)))

    N_sym = int(len(bit_sequence) / mod_order)

    sym_sequence = np.zeros(N_sym, dtype=np.complex_)

    for i in range(N_sym):
        bits = bit_sequence[i * mod_order:(i + 1) * mod_order]
        dec_value = 0
        for j in range(len(bits)):
            dec_value = dec_value | (bits[j] << (mod_order - 1 - j))
        grey_value = grey_code(dec_value)
        sym_sequence[i] = symbols[np.where(grey_mapping == grey_value)]

    return sym_sequence