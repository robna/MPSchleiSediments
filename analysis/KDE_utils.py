import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from settings import Config
import prepare_data


def optimise_bandwidth(data):
    """
    Finds an "optimised" kernel bandwidth for the data
    using sklearn GridSearchCrossValidation and LeaveOneOut algorithms
    """

    grid = GridSearchCV(KernelDensity(kernel=Config.kernel),
                        {'bandwidth': Config.bandwidths},
                        cv=LeaveOneOut())
    grid.fit(data[:, None])
    bw = grid.best_params_['bandwidth']
    return bw


def calculate_kde(data, x_d):
    """
    Makes a kernel density estimation using parameters from the Config class in settings.py.
    Data should be a 1D np-array, x_d is the discrete values where the probability density is evaluated,
    bw is the bandwidth to be used for the kernels
    """

    bw = optimise_bandwidth(data) if Config.optimise_bw else Config.fixed_bw

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bw, kernel=Config.kernel)
    kde.fit(data[:, None])
    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    kde_result = np.exp(logprob)

    return kde_result, bw


def per_sample_kde(pdd_MP, x_d):
    """
    Loop through the MP df (grouped by samples) and calculate one kde per sample.
    Returns a df with the x-values in the first column
    followed by computed densities for each x of each sample.
    """

    kde_results = pd.DataFrame({'x_d': x_d}).T  # initialise results df to be filled in loop

    for SampleName, SampleGroup in pdd_MP.groupby(['Sample']):
        x = SampleGroup[Config.size_dim].values
        kde_result, bw = calculate_kde(x, x_d)

        kde_results.loc[SampleName] = kde_result

        print(f'{SampleName}:    bandwidth is {round(bw, 2)}                  ', end='\r')
        # time.sleep(0.05)

    kde_results = kde_results.T.set_index('x_d').T  # workaround as pandas has no df.set_columns() function
    # kde_results.columns = kde_results.columns.astype(int)  # ...and turn column headers back to integers
    kde_results = kde_results.loc[:, Config.lower_size_limit:Config.upper_size_limit]  # truncate to relevant range

    kde_results.index.name = 'Sample'
    return kde_results


def probDens2conc(size_pdfs, sdd_MP):
    """
    Converts df of probability densities into df of concentrations per size bins.
    Size bins are represented by their lower boundary as the df's index (inclusive)
    and reach up to the next index (exclusive). Both boundaries are named in the final index labels.
    """

    steps = size_pdfs.T.reset_index().x_d.shift(-1) - size_pdfs.T.reset_index().x_d

    size_prob = size_pdfs.mul(list(steps), axis=1)
    size_conc = size_prob.mul(sdd_MP.set_index('Sample').Concentration, axis=0)

    size_conc = prepare_data.complete_index_labels(size_conc)

    return size_conc
