import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from settings import Config
import prepare_data
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()

def r_setup(packnames):
    # import R's "utils" package
    utils = rpackages.importr('utils')
    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    # Selectively install what needs to be install.
    # We are fancy, just because we can.
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))


def bound_kde(n, x, low, high, bw=Config.fixed_bw, method='adjustedKDE'):
    r_setup(['scdensity'])
    sckde = rpackages.importr('scdensity')
    kde = sckde.scdensity(x, bw=bw,
                          constraint=ro.StrVector(["boundedLeft", "boundedRight", "monotoneRightTail"]),
                          opts=ro.ListVector({"lowerBound": low, "upperBound": high}),
                          method=method)
    # sample required amount of values from kde, adapted from https://alfurka.github.io/2020-12-16-sampling-with-density-function/
    cdf = r.cumsum(kde.rx2('y')) / r.cumsum(kde.rx2('y'))[r.length(kde.rx2('y'))-1]
    set_seed = r('set.seed')
    sampled_values = []
    for i in range(n):
        set_seed(i)
        sampled_values.append(kde.rx2('x')[r.findInterval(r.runif(1), cdf)+1-1][0])
    return sampled_values, kde, cdf


def optimise_bandwidth(data, weights):
    """
    Finds an "optimised" kernel bandwidth for the data
    using sklearn GridSearchCrossValidation and LeaveOneOut algorithms
    """

    grid = GridSearchCV(KernelDensity(kernel=Config.kernel),
                        {'bandwidth': Config.bandwidths},
                        cv=LeaveOneOut())
    grid.fit(data[:, None], sample_weight=weights)
    bw = grid.best_params_['bandwidth']
    return bw


def calculate_kde(data, x_d, weights):
    """
    Makes a kernel density estimation using parameters from the Config class in settings.py.
    Data should be a 1D np-array, x_d is the discrete values where the probability density is evaluated,
    bw is the bandwidth to be used for the kernels
    """

    bw = optimise_bandwidth(data, weights) if Config.optimise_bw else Config.fixed_bw

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bw, kernel=Config.kernel)
    kde.fit(data[:, None], sample_weight=weights)

    params = kde.get_params(deep=True)

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(x_d[:, None])
    kde_result = np.exp(logprob)

    return kde_result, params


def per_sample_kde(pdd_MP, x_d, weight_col=None, size_dim=Config.size_dim):
    """
    Loop through the MP df (grouped by samples) and calculate one kde per sample.
    Returns a df with the x-values in the first column
    followed by computed densities for each x of each sample.
    """

    kde_results = pd.DataFrame({'x_d': x_d}).T  # initialise results df to be filled in loop

    for SampleName, SampleGroup in pdd_MP.groupby(['Sample']):
        if weight_col is not None:
            weights = SampleGroup[weight_col].values
        else:
            weights = None

        x = SampleGroup[size_dim].values
        kde_result, params = calculate_kde(x, x_d, weights)

        kde_results.loc[SampleName] = kde_result

        print(f'{SampleName}:    bandwidth is {round(params["bandwidth"], 2)}                  ')#, end='\r')  # use end='\r' to overwrite previous line
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
