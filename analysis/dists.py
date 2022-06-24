import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

from settings import dist_params
from plots import dist_hist


def get_distributed_data(particle_type, prop, dist_name='triang', low=0, peak=0.5, high=1, n=1000, plot_bins=0):
    """
    Use scipy.stats to calculate n random data points following a triangular distribution.

    :param particle_type: str
    :param prop: str
    :param low: lower bound of the distribution
    :param peak: peak of the distribution
    :param high: upper bound of the distribution
    :param n: number of data points to generate
    :param plot_bins: number of bins to use for the plotting a histogram, default: 0 -> no plotting
    :return: numpy array of random data points
    """

    dist = eval(f'stats.{dist_name}')

    if dist_name == 'triang':
            
        loc = low
        scale = high - low
        c = (peak - low) / scale

        r = dist.rvs(loc=loc, scale=scale, c=c, size=n)

        if plot_bins:
            x = np.linspace(dist.ppf(0.01, c, loc, scale), dist.ppf(0.99, c, loc, scale), 100)
            pdf = dist.pdf(x, c, loc, scale)
            dist_hist(prop, particle_type, dist_name, x, pdf, r, plot_bins)

        return r

    if dist_name == 'norm':
            
        loc = peak
        scale = (high - low) / 2
        
        r = dist.rvs(loc=loc, scale=scale, size=n)

        if plot_bins:
            x = np.linspace(dist.ppf(0.01, loc, scale), dist.ppf(0.99, loc, scale), 100)
            pdf = dist.pdf(x, loc, scale)
            dist_hist(prop, particle_type, dist_name, x, pdf, r, plot_bins)
            
        return r


def compose_dists_df(n, plot_bins=0):

    df = pd.DataFrame(dist_params)
    of = pd.DataFrame(columns=['particle_type'])

    for _, row in df.iterrows():
        if row['particle_type'] not in of['particle_type'].to_list():
            of = pd.concat([of, pd.DataFrame({
                'particle_type': row['particle_type'],
                row['prop']: get_distributed_data(
                    particle_type = row['particle_type'],
                    prop = row['prop'],
                    dist_name = row['dist_name'],
                    low=row['low'],
                    peak=row['peak'],
                    high=row['high'],
                    n=n,
                    plot_bins=plot_bins)
                    })],axis=0)
        else:
            of.loc[of['particle_type']==row['particle_type'], row['prop']] = get_distributed_data(
                particle_type = row['particle_type'],
                prop = row['prop'],dist_name = row['dist_name'],
                low=row['low'],
                peak=row['peak'],
                high=row['high'],
                n=n,
                plot_bins=plot_bins)
    return of.reset_index(drop=True)