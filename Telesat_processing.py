
import pandas as pd

from pass_util import *

df = pd.read_csv('bulk_data/Telesat_0_7day.csv')

# Convert into passes
#passes_df = instances_to_passes(df)
telesat_passes_df = instances_to_passes_fast(df)

def transform_sff(passes_df):
    import quaternion as qn

    qq = [None] * len(passes_df)
    sff = [None] * len(passes_df)

    # Todo make this properly vectorized
    for i, pass_df in passes_df.iterrows():
        rr_a = pass_df.rr_a
        rr_ab = pass_df.rr_b - pass_df.rr_a

        # Start by finding the rotation of the sc in the ECIF
        z_ecif = np.array([[0, 0, 1]])

        def rotate_vectors(qq, vv):
            vvp = np.zeros(vv.shape)
            for j, q in enumerate(qq):
                q = q.normalized()
                vvp[j, :] = (q * np.quaternion(0, vv[j, 0], vv[j, 1], vv[j, 2]) * q.inverse()).vec

            return vvp

        ## New method
        vv1 = rr_a
        vv2 = z_ecif

        def find_quat(v1, v2):
            # v1 - vector to rotate to
            # v2 - vector to rotate from
            xyz = np.cross(v1, v2)
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            w = np.sqrt(np.linalg.norm(v1, axis=1) ** 2 * np.linalg.norm(v2, axis=1) ** 2) + np.sum(v1 * v2, axis=1)
            n = np.sqrt(x ** 2 + y ** 2 + z ** 2 + w ** 2)
            return qn.as_quat_array(np.array([w, x, y, z]).T) / n

        qq[i] = find_quat(vv1, vv2)
        sff[i] = rotate_vectors(qq[i], rr_ab)

    passes_df['qq'] = qq
    passes_df['sff'] = sff
    return passes_df


telesat_passes_df = transform_sff(telesat_passes_df)

telesat_passes_df.to_pickle('Telesat_0_7day.pkl')
telesat_passes_df = pd.read_pickle('Telesat_0_7day.pkl')

import matplotlib
import matplotlib.pyplot as plt

N_passes = 10

fig, ax = plt.subplots()

for n in range(N_passes):
    pass_df = telesat_passes_df.iloc[n]
    range = np.linalg.norm(pass_df.rr_a - pass_df.rr_b, axis=1)
    ax.plot(pass_df['ttof'], range)

fig.show()
fig.savefig('test.svg')
fig.close()

pass