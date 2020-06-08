
import pandas as pd

from pass_util import *

df = pd.read_csv('bulk_data/Telesat_0_7day.csv')

# Convert into passes
#passes_df = instances_to_passes(df)
passes_df = instances_to_passes_fast(df)

passes_df.to_pickle('Telesat_0_7day.pkl')
passes_df = pd.read_pickle('Telesat_0_7day.pkl')

import matplotlib
import matplotlib.pyplot as plt

N_passes = 10

fig, ax = plt.subplots()

for n in range(N_passes):
    pass_df = passes_df.iloc[n]
    range = np.linalg.norm(pass_df.rr_a - pass_df.rr_b, axis=1)
    ax.plot(pass_df['ttof'], range)
    fig.show()

fig.show()
fig.savefig('test.svg')

pass