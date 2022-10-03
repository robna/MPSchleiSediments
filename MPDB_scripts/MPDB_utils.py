import pandas as pd
import numpy as np
from MPDB_settings import polyDropList, blindList, Config


def poly_exclude(MP):
    MP = MP.query('polymer_type not in @polyDropList and library_entry not in @polyDropList')
    # equal to: MP[~MP['polymer_type'].isin(polyDropList) & ~MP['library_entry'].isin(polyDropList)]
    return MP


def particle_amplification(MP):
    # Replace NaN with 1 in Fraction_analysed, assuming whole sample has been analysed when no value was provided
    MP.Fraction_analysed.fillna(1, inplace=True)

    # Take 1 divided by Fraction_analysed (rounded to next integer) to get the factor each particle needs to be repeated
    # by to extrapolate to whole sample.
    repeater = 1 / MP.Fraction_analysed
    repeater.loc[MP.Particle_name.str.contains('IOW') == True] = 1  # particles with 'IOW' in their name have been picked before splitting, thus should not be repeated
    repeater.loc[MP.Sample == "Schlei_S1_15cm"] = 100 / MP.loc[  # S1_15_cm had varying individual filter splits, so particles ned to get their repeater value from the Comment field in the analysis table
        MP.Sample == 'Schlei_S1_15cm', 'Comment'
        ].str.split('(').str[1].str.extract('(\d+)').astype(float)[0]
    MP = MP.loc[np.repeat(MP.index.values, round(repeater))]  # Need to use "round" here because we cannot create "3.5" particles out of one. This distorts the results slightly in cases were the fraction analysed is not a whole number fraction like 1/2 or 1/3.
    MP.IDParticles = MP.IDParticles.astype(str) + '_' + MP.groupby('IDParticles').cumcount().astype(str)  # add a number to the ID to make it unique
    return MP


def geom_mean(MP):
    MP['size_geom_mean'] = np.sqrt(MP['Size_1_[µm]'] * MP['Size_2_[µm]'])
    return MP


def size_filter(MP):
    MP = MP[MP[Config.size_filter_dimension] >= Config.size_filter_highpass]
    MP = MP[MP[Config.size_filter_dimension] < Config.size_filter_lowpass]
    return MP


def shape_colour(MP):
    MP['Colour'].replace(['transparent', 'undetermined', 'white', 'non-determinable', 'grey', 'brown', 'black'],
                         'unspecific', inplace=True)
    MP['Colour'].replace(['violet'], 'blue', inplace=True)
    MP['Shape'].replace(['spherule', 'irregular', 'flake', 'foam', 'granule', 'undetermined'], 'irregular',
                        inplace=True)  # combine all non-fibrous particle shapes
    return MP


def set_id_to_index(MP):
    MP.set_index('IDParticles', inplace=True)
    return MP


def separate_MPs(MP):
    # take environmental MP from dataset (without any blinds or blanks):
    env_MP = MP.loc[(MP.Compartment == 'sediment') & (MP.Site_name == 'Schlei') & (MP.Contributor == 27)].copy()

    #take IOW blinds from dataset:
    IOW_blind_MP = MP.loc[(MP.Site_name == "lab") & (MP.Project == "BONUS MICROPOLL") & (MP.Contributor == 27)]
    IOW_blind_MP = IOW_blind_MP.loc[IOW_blind_MP.Sample.isin(blindList)]

    # make combined env. MP and blind MP dataframe:
    samples_MP = pd.concat([env_MP, IOW_blind_MP], axis=0)

    #take IPF blanks from dataset:
    IPF_blank_MP = MP.loc[(MP.sample_ID.isin(samples_MP.lab_blank_ID))].copy()
    # For differentiation to env_MP their `size_geom_mean` is renamed to `blank_size_geom_mean`.
    IPF_blank_MP.rename(columns={'size_geom_mean': 'blank_size_geom_mean'}, inplace=True)
    IPF_blank_MP['Sample'] = IPF_blank_MP['Sample'].str.replace('Blank_', '', 1)
    # the last option (called count, here "1") was added here because some of the IOW blinds
    # have the sample name "Blank_xxxxx" and their corresponding IPF blanks
    # have the sample name "Blank_Blank_xxxxx". So with count option set to "1",
    # only the first occurence of "Blank_" is replaced by "".
    return env_MP, IOW_blind_MP, samples_MP, IPF_blank_MP
