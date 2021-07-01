import numpy as np

class Config:
    lower_size_limit: int = 1  # the smallest particle size in µm included in the kde computation
    upper_size_limit: int = 1000  # the largest particle size in µm included in the kde computation
    kde_steps: int = 1000  # number points on the size axis where the kde is defined (like number of bins in histogram)
    optimise_bw: bool = False  # if True: compute an individual bandwidth for each sample before computing the KDE
    bws_to_test: int = 100  # if optimise_bw = True: how many bandwidth values should be tried out?
    fixed_bw: int = 50  # if optimise_bw = False: fixed bandwidth value to use for all kde's
    kernel: str = 'gaussian'  # type of kernel to be used


# creates the x-axis data for the prob. dist. func.
Config.x_d: np.array = np.linspace(Config.lower_size_limit, Config.upper_size_limit, Config.kde_steps)

Config.bandwidths: np.array = 10 ** np.linspace(0, 2, Config.bws_to_test)  # creates the range of bandwidths to be tested

Regio_Sep = {
    'Schlei_S1_15cm': 'inner',
    'Schlei_S2': 'inner',
    'Schlei_S3': 'inner',
    'Schlei_S5': 'river',
    'Schlei_S8': 'inner',
    'Schlei_S10': 'inner',
    'Schlei_S10_15cm': 'inner',
    'Schlei_S11': 'inner',
    'Schlei_S13': 'inner',
    'Schlei_S14': 'outlier',
    'Schlei_S15': 'inner',
    'Schlei_S17': 'inner',
    'Schlei_S19': 'outlier',
    'Schlei_S22': 'outer',
    'Schlei_S23': 'outer',
    'Schlei_S24': 'outer',
    'Schlei_S25': 'outer',
    'Schlei_S26': 'outer',
    'Schlei_S27': 'outer',
    'Schlei_S30': 'outer',
    'Schlei_S31': 'outer'
}
