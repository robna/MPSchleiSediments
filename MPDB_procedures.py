import pandas as pd

def blank_procedure(samples_MP, IPF_blank_MP):
    """
    Identifies particles that need to be eliminated from the data set,
    because they match with particles found in corresponding IPF lab blanks
    """
    
    samples_MP_copy = samples_MP.copy()  # take copy to retain an unaltered version of samples_MP

    IPF_elimination_list = pd.DataFrame(columns=[  # create empty dataframe to collect particles-to-be-flagged in a loop
        'ID_blank_particle',
        'ID_sample_particle',
        'Sample',
        'polymer_type',
        'Colour',
        'Shape'])

    for label, row_content in IPF_blank_MP.iterrows():  # loop through the list of IPFblanks, particle by particle
        # label contains the particle ID of the current blank particle, row_content contains the data of that particle

        current_blank_particle = samples_MP_copy.reset_index().merge(row_content.to_frame().T,
                                                                     on=['Sample', 'polymer_type', 'Colour', 'Shape'],
                                                                     how='inner').set_index('IDParticles')
        # current_blank_particle is basically extract of samples_MP_copy, where only particles which match the current
        # blank particle in all fields after the "on =" are listed where a match is found, all fields of both lines are
        # written as on long line. column names that exist in both of the merged dataframes get an appendix x or y we
        # only need the entry of blank_size_geom_mean to be written as a new column in all lines (i.e. particles) that
        # have the same phenotype as the current blank particle with this we can calculate the difference between the
        # size_geom_mean of the particle and the blank_size_geom_mean

        if len(current_blank_particle) > 0:  # there might be the case where no particles were found to match a specific
            # blank particle, so we check for this with this if clause

            current_blank_particle['size_diff'] = abs((current_blank_particle['size_geom_mean'] - current_blank_particle[
                'blank_size_geom_mean']))  # here we take the size difference as described above

            eliminee = pd.to_numeric(current_blank_particle['size_diff']).idxmin()  # the particle that has the smallest
            # difference in size_geom_mean to that of the current blank particle is our candidate for elimination,
            # and we save its ID as 'eliminee'

            IPF_elimination_list = pd.concat([IPF_elimination_list, pd.DataFrame({
                # now we keep a record entry of all details of the particle that gets eliminated and append it to the
                # prepared data frame
                'ID_blank_particle': label,
                'ID_sample_particle': eliminee,
                'Sample': current_blank_particle.Sample.iloc[0],
                'polymer_type': current_blank_particle.polymer_type.iloc[0],
                'Colour': current_blank_particle.Colour.iloc[0],
                'Shape': current_blank_particle.Shape.iloc[0]
            }, index=[0])], ignore_index=True)

            samples_MP_copy.drop(eliminee, inplace=True)  # finally we drop the line of the eliminated particle from
            # our particles dataframe, so we can't match it to another blank particle in the next round

            print('For blank particle #', label, ': ', 'Env. particle #', eliminee, 'was eliminated.')

        else:

            print('For blank particle #', label, ': ', 'Nothing to clean up.')
    print('Total number of particles eliminated due to IPF blank particle matches: ', len(IPF_elimination_list))
    return samples_MP_copy, IPF_elimination_list


def blind_procedure(env_MP, syn_blind):
    """
    Identifies particles that need to be eliminated from the data set,
    because they match with particles found in corresponding IOW blinds
    """
    
    env_MP_copy = env_MP.copy()
    IOW_elimination_list = pd.DataFrame(
        columns=['ID_blind_particle', 'Blind_sample', 'ID_sample_particle', 'Sample', 'polymer_type', 'Colour', 'Shape'])
    old_length = 0  # initiate length variable for elimination list for logging purposes

    for sample_name, sample_group in env_MP.groupby('Sample'):
        for label, row_content in syn_blind.iterrows():
            current_blind_particle = sample_group.reset_index().merge(row_content.to_frame().T,
                                                                      on=['polymer_type', 'Colour', 'Shape'],
                                                                      how='inner').set_index('IDParticles')

            if len(current_blind_particle) > 0:
                current_blind_particle['size_diff'] = abs(
                    (current_blind_particle['size_geom_mean'] - current_blind_particle['blind_size_geom_mean']))
                eliminee = pd.to_numeric(current_blind_particle['size_diff']).idxmin()
                sample_group.drop([eliminee], inplace=True)

                IOW_elimination_list = pd.concat([IOW_elimination_list, pd.DataFrame({
                    'ID_blind_particle': label,
                    'Blind_sample': current_blind_particle.Sample_y.iloc[0],
                    'ID_sample_particle': eliminee,
                    'Sample': env_MP_copy.loc[eliminee, 'Sample'],
                    'polymer_type': current_blind_particle.polymer_type.iloc[0],
                    'Colour': current_blind_particle.Colour.iloc[0],
                    'Shape': current_blind_particle.Shape.iloc[0]
                }, index=[0])], ignore_index=True)

                env_MP_copy.drop(eliminee, inplace=True)

        print(len(IOW_elimination_list) - old_length, ' particles elimnated in:  ', sample_name)
        old_length = len(IOW_elimination_list)
    print('Total number of particles eliminated due to IOW blind particle matches: ', len(IOW_elimination_list))
    
    return env_MP_copy, IOW_elimination_list


def update_env_and_blind(samples_MP_copy, IOW_blind_MP):
    """
    Porting the particle eliminations that happened in the joint `samples_MP_copy` (env. + blind MP)
    onto the individual dataframes `env_MP` and `IOW_blind_MP`.
    """
    
    IOW_blind_MP = samples_MP_copy[samples_MP_copy.index.isin(IOW_blind_MP.index)].copy()
    # For differentiation to env_MP their `size_geom_mean` is renamed to `blind_size_geom_mean`.
    IOW_blind_MP.rename(columns={'size_geom_mean': 'blind_size_geom_mean'}, inplace=True)

    env_MP = samples_MP_copy[~samples_MP_copy.index.isin(IOW_blind_MP.index)].copy()
    
    return IOW_blind_MP, env_MP



def make_syn_blind(IOW_blind_MP):
    """
    Create a combined dataframe from all blind samples,
    but only including the correct amount of particles
    E.g. in the case of 4 blind samples: only every 4th
    particle would be included (sorted by size, to not
    distort the size spectrum of particles).
    """
    
    blind_PhTs = IOW_blind_MP.groupby(['polymer_type', 'Colour', 'Shape'])  # collection of phenotypes
    blinds = pd.unique(IOW_blind_MP.Sample).size  # number of blind samples that were included
    
    syn_blind = IOW_blind_MP[0:0]
    for group_name, group_content in blind_PhTs:
        current_group = group_content.sort_values(by=['blind_size_geom_mean'], ascending=False)
        syn_blind = pd.concat([syn_blind, current_group[0::blinds]])
    
    return syn_blind