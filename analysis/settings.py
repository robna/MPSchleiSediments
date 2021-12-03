import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

predictors = ['Dist_WWTP', 'TOC', 'D50', 'PC1', 'PC2']  # columns to be used as predictors


class Config:
    rebinning: bool = False  # whether or not to aggregate sizes to coarser bins
    rebin_by_n: int = 5  # make sediment size bins coarser: sum up every n bins into one new

    lower_size_limit: int = 0  # the smallest particle size in µm included in the kde computation
    upper_size_limit: int = 5000  # the largest particle size in µm included in the kde computation

    allowed_zeros: float = 1  # the fraction of allowed zeros in a sediment size bin

    kde_steps: int = 496  # number points on the size axis where the kde is defined (like number of bins in histogram)
    optimise_bw: bool = False  # if True: compute an individual bandwidth for each sample before computing the KDE
    bws_to_test: int = 100  # if optimise_bw = True: how many bandwidth values should be tried out?
    fixed_bw: int = 50  # if optimise_bw = False: fixed bandwidth value to use for all kde's
    kernel: str = 'gaussian'  # type of kernel to be used

    bin_conc: bool = False  # True: calculate MP conc. (#/kg) for individual size bins; False: calculate percentages

    MPlow: int = 50
    MPup: int = 250
    SEDlow: int = 50
    SEDup: int = 250

    ILR_transform: bool = True  # if True: use the ILR transform for sediment size data before dimensionality reduction

    glm_family: str = 'Gamma'  # type of family to be used for the GLM
    glm_link: str = None  # Link function for GLM; use None for default link of chosen family
    glm_formula: str = f'Concentration ~ {predictors[0]} +' \
                       f'{predictors[1]}'


# creates the x-axis data for the prob. dist. func.
Config.x_d: np.array = np.linspace(Config.lower_size_limit, Config.upper_size_limit, Config.kde_steps)

Config.bandwidths: np.array = 10 ** np.linspace(0, 2,
                                                Config.bws_to_test)  # creates the range of bandwidths to be tested

regio_sep = {
    'Schlei_S1': 'inner',
    'Schlei_S1_15cm': 'inner',
    'Schlei_S2': 'inner',
    'Schlei_S2_15cm': 'inner',
    'Schlei_S3': 'inner',
    'Schlei_S4': 'inner',
    'Schlei_S4_15cm': 'inner',
    'Schlei_S5': 'river',
    'Schlei_S6': 'inner',
    'Schlei_S7': 'inner',
    'Schlei_S8': 'inner',
    'Schlei_S10': 'inner',
    'Schlei_S10_15cm': 'inner',
    'Schlei_S11': 'inner',
    'Schlei_S12': 'inner',
    'Schlei_S13': 'inner',
    'Schlei_S14': 'outlier',
    'Schlei_S15': 'inner',
    'Schlei_S16': 'inner',
    'Schlei_S17': 'inner',
    'Schlei_S18': 'inner',
    'Schlei_S18_15cm': 'inner',
    'Schlei_S19': 'outlier',
    'Schlei_S20': 'outer',
    'Schlei_S21': 'outer',
    'Schlei_S22': 'outer',
    'Schlei_S23': 'outer',
    'Schlei_S24': 'outer',
    'Schlei_S25': 'outer',
    'Schlei_S26': 'outer',
    'Schlei_S27': 'outer',
    'Schlei_S28': 'outer',
    'Schlei_S29': 'outer',
    'Schlei_S30': 'outer',
    'Schlei_S31': 'outer',
    'Schlei_S32': 'outer'
}
