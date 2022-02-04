import numpy as np
import pandas as pd

from settings import Config


def low_freq_out(df):
    """Identify samples with less then a threshold frequency of particles and remove them from the data set."""
    
    groupy = df.groupby('Sample')
    
    genuine_particle_numbers = groupy.size() * groupy.Fraction_analysed.first()  # number of original particles in each sample (not reproduced by splitting factor)

    # remove samples with less than a threshold number of particles
    df = df[df.Sample.isin(genuine_particle_numbers[genuine_particle_numbers >= Config.min_part_count].index)]

    print(f'Removed {len(genuine_particle_numbers) - len(df.Sample.unique())} samples with less than {Config.min_part_count} particles.')
    print(genuine_particle_numbers[genuine_particle_numbers < Config.min_part_count])

    return df
