import numpy as np
import pandas as pd
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


def calculate_kde(data, x_d=Config.x_d[:, None]):
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


def per_sample_kde(pdd_sdd_MP, x_d=Config.x_d):
    """
    Loop through the MP df (grouped by samples) and calculate one kde per sample.
    Returns a df with the x-values in the first column
    followed by computed densities for each x of each sample.
    """

    kde_results = pd.DataFrame({'x_d': x_d})  # initialise results df to be filled in loop

    for SampleName, SampleGroup in pdd_sdd_MP.groupby(['Sample']):
        x = SampleGroup.size_geom_mean.values
        kde_result, bw = calculate_kde(x, x_d)

        kde_results[SampleName] = kde_result

        # print(f'{SampleName}:    bandwidth is {round(bw, 2)}')

    kde_results.set_index('x_d', inplace=True)
    kde_results = kde_results.loc[Config.lower_size_limit:Config.upper_size_limit]  # truncate to relevant range

    kde_results.columns.name = 'sample'
    return kde_results


def probDens2conc(size_pdfs, sdd_MP_sed):
    """
    Converts df of probability densities into df of concentrations per size bins.
    Size bins are represented by their lower boundary as the df's index (inclusive)
    and reach upt to the next index (exclusive). Both boundary are named in the final index labels.
    """

    steps = size_pdfs.reset_index().x_d.shift(-1) - size_pdfs.reset_index().x_d

    size_prob = size_pdfs.mul(list(steps), axis=0)
    size_conc = size_prob * sdd_MP_sed.set_index('Sample').Concentration

    size_conc = prepare_data.complete_index_labels(size_conc)

    return size_conc


def range_aggregator(df_in):
    """
    Calculates an extended DF showing not only concentrations / freqs in single size bins,
    but additionally also all possible consecutive aggregated (summed) bin combination.
    This function can be used for MP and sediment DFs alike if the input DFs are in the right shape:
    Columns: samples
    Rows: single size bin with lower boundary in µm as index labels
    
    TODO: maybe it is possible to avoid nested loop by summing up shifting DFs?
    """

    df_in.index.name = 'a'
    df_in.reset_index(inplace=True)  # temporary fix because x_d has turn from column to index
    df_out = pd.DataFrame()

    for i in df_in.a:
        for j in df_in.a[df_in.a > i]:
            step = j - i

            relevant_sizes = df_in.loc[(df_in.a >= i) & (df_in.a < j)]
            size_sum = relevant_sizes.sum()
            size_sum.drop('a', inplace=True)

            size_sum.rename(f'{i}_{j}', inplace=True)

            df_out = df_out.append(size_sum)

    df_out.rename_axis(columns='sample', inplace=True)

    return df_out
