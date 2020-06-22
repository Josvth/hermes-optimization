import pandas as pd
import numpy as np
from tqdm import tqdm

def instances_to_passes_fast(instances_df, step_duration = 1.0):

    # Sort in chronological order (if not already)
    #instances_df = instances_df.sort_values(by= 'tof')

    # Combine xyz positions
    r_a = np.vstack((instances_df.xyz_a_x.to_numpy().T, instances_df.xyz_a_y.to_numpy().T, instances_df.xyz_a_z.to_numpy().T)).T
    r_b = np.vstack((instances_df.xyz_b_x.to_numpy().T, instances_df.xyz_b_y.to_numpy().T, instances_df.xyz_b_z.to_numpy().T)).T

    instances_df = instances_df.assign(r_a=pd.Series(np.vsplit(r_a, len(r_a))))
    instances_df = instances_df.assign(r_b=pd.Series(np.vsplit(r_b, len(r_b))))
    instances_df = instances_df.drop(columns=['xyz_a_x', 'xyz_a_y', 'xyz_a_z', 'xyz_b_x', 'xyz_b_y', 'xyz_b_z'])

    # Group by strand
    grouped_df = instances_df.groupby('StrandName')

    pp = []

    pass_id = 0

    with tqdm(total=len(grouped_df), desc='Group') as group_bar:
        for name, group in grouped_df:

            # Calculate delta time between instances
            group = group.assign(delta_tof = group['tof'] - np.concatenate(([group['tof'].iloc[0]],group['tof'].iloc[0:-1].values)))

            i_l = group.index[0]
            for i in group.index[group['delta_tof'] > 2*step_duration]:

                # Select instances between indexes
                pass_instances = group.loc[i_l:i - 1]

                if pass_instances.tof.values[1] == pass_instances.tof.values[0]:
                    pass

                # Create pass df
                p = {"StartTof": pass_instances.iloc[0].tof,
                     "StopTof": pass_instances.iloc[-1].tof,
                     "ttof": [pass_instances.tof.values],
                     "rr_a": [np.concatenate(pass_instances.r_a.values)],
                     "rr_b": [np.concatenate(pass_instances.r_b.values)],
                     "rrange": [np.linalg.norm(np.concatenate(pass_instances.r_b.values) - np.concatenate(pass_instances.r_a.values), axis=1)],
                     'StrandName': pass_instances.iloc[0].StrandName}

                # Create pass df and add to list
                pass_df = pd.DataFrame.from_dict(p)
                pp.append(pass_df)

                # # plot pass
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots()
                # ax.plot(pass_df.ttof[0], pass_df.rrange[0])
                # fig.show()

                i_l = i

            group_bar.update(1)

        passes_df = pd.concat(pp, ignore_index=True)
        passes_df['Duration'] = passes_df['StopTof'] - passes_df['StartTof']
        #return passes_df
        return passes_df.sort_values(by='StartTof').reset_index(drop=True)

def instances_to_passes(instances_df, step_duration = 1.0):

    # Sort in chronological order (if not already)
    instances_df = instances_df.sort_values(by= 'tof')

    # Combine xyz positions
    r_a = np.vstack((instances_df.xyz_a_x.to_numpy().T, instances_df.xyz_a_y.to_numpy().T, instances_df.xyz_a_z.to_numpy().T)).T
    r_b = np.vstack((instances_df.xyz_b_x.to_numpy().T, instances_df.xyz_b_y.to_numpy().T, instances_df.xyz_b_z.to_numpy().T)).T

    instances_df = instances_df.assign(r_a=pd.Series(np.vsplit(r_a, len(r_a))))
    instances_df = instances_df.assign(r_b=pd.Series(np.vsplit(r_b, len(r_b))))
    instances_df = instances_df.drop(columns=['xyz_a_x', 'xyz_a_y', 'xyz_a_z', 'xyz_b_x', 'xyz_b_y', 'xyz_b_z'])

    # Group by strand
    grouped_df = instances_df.groupby('StrandName')

    pp = []
    passes_df = pd.DataFrame()

    pass_id = 0

    with tqdm(total=len(grouped_df), desc='Group') as group_bar:
        for name, group in grouped_df:

            p = {"ID": pass_id,
                 "StartTof": group.iloc[0].tof,
                 "StopTof": group.iloc[0].tof,
                 "ttof": [group.iloc[0].tof],
                 "rr_a": [group.iloc[0].r_a],
                 "rr_b": [group.iloc[0].r_b],
                 'StrandName': group.iloc[0].StrandName}

            for index, instance in group.iterrows():

                last_tof = p["StopTof"] # last found tof in this pass
                instance_tof = instance.tof                   # tof of this new instance

                if instance_tof - last_tof <= 1.5 * step_duration:   # check if this step is part of the previous period
                    # if so the stop time is extended
                    p["StopTof"] = instance_tof

                    # and the tof and xyz coordinates are added to the lists
                    p["ttof"].append(instance_tof)
                    p["rr_a"].append(instance.r_a)
                    p["rr_b"].append(instance.r_b)

                else:
                    # if not then a pass was finished, add it to the passes_df and continue
                    p["ttof"] = [np.hstack(p["ttof"])]
                    p["rr_a"] = [np.vstack(p["rr_a"])]
                    p["rr_b"] = [np.vstack(p["rr_b"])]

                    pass_df = pd.DataFrame.from_dict(p)
                    pass_df.set_index("ID")
                    pp.append(pass_df)

                    pass_id = pass_id + 1   # increment pass id

                    # initialize the next pass df
                    p = {"ID": pass_id,
                         "StartTof": instance.tof,
                         "StopTof": instance.tof,
                         "ttof": [instance.tof],
                         "rr_a": [instance.r_a],
                         "rr_b": [instance.r_b],
                         'StrandName': instance.StrandName}

            group_bar.update(1)

    # Finalize the last row
    pass_df = pd.DataFrame.from_dict(p)
    pass_df.set_index("ID")
    pp.append(pass_df)

    # Create dataframe of passes
    passes_df = pd.concat(pp)
    passes_df.set_index("ID")
    # Calculate total duration of passes
    passes_df['Duration'] = passes_df['StopTof'] - passes_df['StartTof']


    return passes_df


