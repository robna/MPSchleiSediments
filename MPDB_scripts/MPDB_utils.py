import mysql.connector
import pandas as pd
import numpy as np
import getpass
from MPDB_settings import polyDropList, blankList


def poly_exclude(MP):
    MP = MP[~MP[['polymer_type', 'library_entry']].isin(polyDropList)]
    return MP


def particle_amplification(MP):
    # Replace NaN with 1 in Fraction_analysed, assuming whole sample has been analysed when no value was provided
    MP.Fraction_analysed.fillna(1, inplace=True)

    # Take 1 divided by Fraction_analysed (rounded to next integer) to get the factor each particle needs to be repeated
    # by to extrapolate to whole sample. Then do the repetition using np. repeat and write it back into "MP"
    MP = MP.loc[np.repeat(MP.index.values, round(1 / MP.Fraction_analysed))]

    MP.IDParticles = MP.IDParticles.astype(str) + '_' + MP.groupby('IDParticles').cumcount().astype(str)  # .replace('0','')
    # --> replace 0 if not wanted
    return MP


def geom_mean(MP):
    MP['size_geom_mean'] = np.sqrt(MP['Size_1_[µm]'] * MP['Size_2_[µm]'])
    MP.set_index('IDParticles', inplace=True)
    MP = MP[MP['Size_1_[µm]'] >= 50]
    return MP


def shape_colour(MP):
    MP['Colour'].replace(['transparent', 'undetermined', 'white', 'non-determinable', 'grey', 'brown', 'black'],
                         'unspecific', inplace=True)
    MP['Colour'].replace(['violet'], 'blue', inplace=True)
    MP['Shape'].replace(['spherule', 'irregular', 'flake', 'foam', 'granule', 'undetermined'], 'irregular',
                        inplace=True)  # combine all non-fibrous particle shapes
    return MP


def separate_MPs(MP):
    # take environmental MP from dataset (without any blinds or blanks):
    env_MP = MP.loc[(MP.Compartment == 'sediment') & (MP.Site_name == 'Schlei') & (MP.Contributor == 27)].copy()

    #take IOW blinds from dataset:
    IOW_blind_MP = MP.loc[(MP.Site_name == "lab") & (MP.Project == "BONUS MICROPOLL") & (MP.Contributor == 27)]
    IOW_blind_MP = IOW_blind_MP.loc[IOW_blind_MP.Sample.isin(blankList)]

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
