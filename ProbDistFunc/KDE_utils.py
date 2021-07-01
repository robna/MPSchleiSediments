import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from KDE_settings import Config


def optimise_bandwidth(data):
    """"Finds an "optimised" kernel bandwidth for the data
    using sklearn GridSearchCrossValidation and LeaveOneOut algorithms"""

    grid = GridSearchCV(KernelDensity(kernel=Config.kernel),
                        {'bandwidth': Config.bandwidths},
                        cv=LeaveOneOut())
    grid.fit(data[:, None])
    bw = grid.best_params_['bandwidth']
    return bw


def calculate_kde(data):
    """Makes a kernel density estimation using parameters from the Config class in KDE_settings.py.
    Data should be a 1D np-array, x_d is the discrete values where the probability density is evaluated,
    bw is the bandwidth to be used for the kernels"""

    bw = optimise_bandwidth(data) if Config.optimise_bw else Config.fixed_bw

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bw, kernel=Config.kernel)
    kde.fit(data[:, None])
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(Config.x_d[:, None])
    kde_result = np.exp(logprob)

    return kde_result, bw


def per_sample_kde(pdd_sdd_MP):
    """Loop through the MP df (grouped by samples) and calculate one kde per sample.
    Returns a df with the x-values in the first column followed by computed densities for each x of each sample."""
    kde_results = pd.DataFrame({'x_d': Config.x_d})  # initialise results df to be filled in loop

    for SampleName, SampleGroup in pdd_sdd_MP.groupby(['Sample']):
        x = SampleGroup.size_geom_mean.values
        kde_result, bw = calculate_kde(x)

        kde_results[SampleName] = kde_result

        print(f'{SampleName}:    bandwidth is {round(bw, 2)}')
    return kde_results




