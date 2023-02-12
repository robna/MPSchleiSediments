import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from settings import Config
import prepare_data
import rpy2.robjects as ro  # Obs: rpy2 needs to be installed in v.3.5.1. most recent version (3.5.4) did only work in the jupyter notebooks, but not in streamlit!!
import rpy2.robjects.packages as rpackages
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()


def optimise_bandwidth(data, weights=None):
    """
    Finds an "optimised" kernel bandwidth for the data
    using sklearn GridSearchCrossValidation and LeaveOneOut algorithms
    """

    grid = GridSearchCV(KernelDensity(kernel=Config.kernel),
                        {'bandwidth': Config.bandwidths},
                        cv=5,
                        n_jobs=-1)
    grid.fit(np.array(data)[:, np.newaxis], sample_weight=weights)
    bw = grid.best_params_['bandwidth']
    return bw


def calculate_kde(data, x_d, weights=None, bw=Config.fixed_bw, optimise=Config.optimise_bw):
    """
    Makes a kernel density estimation using parameters from the Config class in settings.py.
    Data should be a 1D np-array, x_d is the discrete values where the probability density is evaluated,
    bw is the bandwidth to be used for the kernels
    """

    bw = optimise_bandwidth(data, weights) if optimise else bw

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=bw, kernel=Config.kernel)
    kde.fit(np.array(data)[:, np.newaxis], sample_weight=weights)

    params = kde.get_params(deep=True)

    # score_samples returns the log of the probability density
    logprob = kde.score_samples(np.array(x_d)[:, np.newaxis])
    pdf = np.exp(logprob)

    return pdf, params, kde


def per_sample_kde(pdd_MP, x_d, weight_col=None, size_dim=Config.size_dim, bw=Config.fixed_bw, optimise=Config.optimise_bw):
    """
    Loop through the MP df (grouped by samples) and calculate one kde per sample.
    Returns a df with the x-values in the first column
    followed by computed densities for each x of each sample.
    """

    pdfs = pd.DataFrame({'x_d': x_d}).T  # initialise results df to be filled in loop

    for SampleName, SampleGroup in pdd_MP.groupby('Sample'):
        if weight_col is not None:
            weights = SampleGroup[weight_col].values
        else:
            weights = None

        x = SampleGroup[size_dim].values
        pdf, params, _ = calculate_kde(x, x_d, weights, bw, optimise)

        pdfs.loc[SampleName] = pdf

        if optimise:
            print(f'{SampleName}:    bandwidth is {round(params["bandwidth"], 2)}                  ')#, end='\r')  # use end='\r' to overwrite previous line
        # time.sleep(0.05)

    pdfs = pdfs.T.set_index('x_d').T  # workaround as pandas has no df.set_columns() function
    # pdfs.columns = pdfs.columns.astype(int)  # ...and turn column headers back to integers
    # pdfs = pdfs.loc[:, Config.lower_size_limit:Config.upper_size_limit]  # may be activated truncate to relevant range, if the supplied x_d extends beyond the relevant range

    pdfs.index.name = 'Sample'
    return pdfs
    

def probDens2prob(size_pdfs, sdd_MP=None):
    """
    Converts df of probability densities into df of concentrations per size bins.
    Size bins are represented by their lower boundary as the df's index (inclusive)
    and reach up to the next index (exclusive). Both boundaries are named in the final index labels.
    """

    #steps = size_pdfs.T.reset_index().x_d.shift(-1) - size_pdfs.T.reset_index().x_d
    lower_bounds = size_pdfs.columns.to_numpy()
    bin_widths = lower_bounds[1:] - lower_bounds[:-1]
    bin_widths = np.append(bin_widths, np.nan)
    
    size_prob = size_pdfs.mul(bin_widths, axis=1)
    
    # TODO: if it is needed to get conc. per size bin instead of probs, these 2 lines should be moved somewhere later in the pipline. Here they are of no good use and cause problems with the subsequent calculation of the distributions medians.
    # if Config.bin_conc:
    #    size_prob = size_prob.mul(sdd_MP.set_index('Sample').Concentration, axis=0)

    size_prob = prepare_data.complete_index_labels(size_prob)
    size_prob = prepare_data.close_compositional_data(size_prob)

    return size_prob


def load_manual_height_measurments(low, high):
        df = pd.read_csv('../data/ManualHeights_Schlei_S8_v2.csv')
        x = df.loc[(df.manual_Size_3_um >= low) & (df.manual_Size_3_um <= high), 'manual_Size_3_um']
        return x


def r_setup(packnames):
    """
    Function to install R packages. Taken from https://rpy2.github.io/doc/latest/html/introduction.html#installing-packages
    :param packnames: list of R packages to install
    """
    # Selectively install what needs to be install.
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        # import R's "utils" package
        utils = rpackages.importr('utils')
        # select a mirror for R packages
        utils.chooseCRANmirror(ind=1) # select the first mirror in the list
        utils.install_packages(ro.StrVector(names_to_install))


def bound_kde(n, low, high, x=None, bw=Config.fixed_bw, method='adjustedKDE'):
    """
    Calculates a shape-restricted KDE in R.
    :param n: number of values to sample from the calculated distribution
    :param low: lower boundary of the shape restriction (i.e. force probability density of 0 below this value)
    :param high: upper boundary of the shape restriction (i.e. force probability density of 0 above this value)  -- Obs: only works correctly for 'greedy' method
    :param x: array / series of values to evaluate the probability density at
    :param bw: bandwidth to use for the KDE
    :param method: method to use for the shape restriction. Options are 'adjusted' (default), 'weighted' and 'greedy'
    :return sampled_values: array of sampled values from the calculated distribution
    :return kde: the R kde object
    :return cdf: the corresponding cumulative distribution function
    """
    if x is None:
        x = load_manual_height_measurments(low, high)
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
        set_seed(i+42)  # set seed for reproducibility, added 42, as using only i sampled one negative value in our case
        sampled_values.append(kde.rx2('x')[r.findInterval(r.runif(1), cdf)+1-1][0])
    return sampled_values, kde, cdf


def unbound_kde(n, low, high, x=None, exceed_high_by=Config.exceed_high_by):
    """
    Calculates a shape-unrestricted KDE. The sorted samples from the KDE may be used to replace the original values in an ordered manner (i.e. assign the smallest sampled value to the smallest original value, etc.)
    :param n: number of values to sample from the calculated distribution
    :param low: lowest particle height that should be replaced by the calculated distribution
    :param high: largest particle height that should be replaced by the calculated distribution
    :param x: array / series of values to evaluate the probability density at
    :param exceed_high_by: how much (in %) may the largest possibly sampled values exceed the largest original particle height (if set to 0, heights close to the upper boundary will consequently be reduced)
    :return sampled_values: array of sampled values from the calculated distribution, sorted from low to high
    :return kde: the kde object
    :return cdf: the corresponding cumulative distribution function
    """
    if x is None:
        x = load_manual_height_measurments(low, high)
    _, params, kde = calculate_kde(x, x, optimise=True)
    print(f"KDE bandwidth for particle height estimation from manually measured heights: {params['bandwidth']}")
    sampled_values = np.zeros(n)
    random_state_tester = 0
    mask = [True]
    while any(mask):
        random_state_tester += 1
        mask = (sampled_values <= 0) | (sampled_values >= Config.height_high * (1+exceed_high_by/10))
        sampled_values[mask] = kde.sample(mask.sum(), random_state=random_state_tester)[:,0]
    sampled_values = np.sort(sampled_values)
    return sampled_values, kde
