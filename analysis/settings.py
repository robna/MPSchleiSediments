import logging
from pathlib import Path
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, make_scorer
from cv_helpers import median_absolute_percentage_error

target = 'Concentration'
featurelist = [
    'Depth',
    'Dist_Land',
    'WWTP_influence_as_mean_time_travelled__sed_18µm_allseasons_444',#pre-final GLM
    'WWTP_influence_as_mean_time_travelled__sed_18µm_autumn_222',#pre-final GLM
    'WWTP_influence_as_cumulated_residence__nosed_18µm_autumn_222', #pre-final RF
    'WWTP_influence_as_mean_time_travelled__nosed_18µm_allseasons_222',#pre-final GLM
    'WWTP_influence_as_cumulated_residence__nosed_0µm_allseasons_222_',#pre-final RF*
    'WWTP_influence_as_mean_time_travelled__nosed_0µm_allseasons_222_',#pre-final GLM*
    # 'SED_D50', #pre-final RF
    # 'perc_MUD', # pre-final GLM
    # 'PC1', # pre-final GLM + RF*
    # 'PC2'
]

default_predictors = ['Dist_WWTP', 'TOC', 'Q("SED_D50")"', 'PC1', 'PC2']  # columns to be used as predictors

sediment_data_filepaths = {
    'IOW_Volume_logscale': 'data/sediment_grainsize_IOW_vol_log-cau_not-closed.csv',
    'IOW_Volume_linscale': 'data/sediment_grainsize_IOW_vol_linear_not-closed.csv',
    'IOW_Counts_logscale': 'data/sediment_grainsize_IOW_count_log-cau_not-closed.csv',
    'CAU_Volume_logscale': 'data/sediment_grainsize_CAU_vol_log-cau_closed.csv',
    'IOW_GRADISTAT_Volume_logscale': 'data/GRADISTAT_IOW_vol_log-cau_not-closed.csv',
    'IOW_GRADISTAT_Counts_logscale': 'data/GRADISTAT_IOW_count_log-cau_not-closed.csv',
    'CAU_GRADISTAT_Volume_logscale': 'data/GRADISTAT_CAU_vol_log-cau_closed.csv',
}

class Config:
    # General data preparation settings
    sediment_layer_depth: float = 0.05  # thickness of the sampled sediment layer in m
    sediment_grainsize_basis: str = 'Volume_logscale'  # 'Volume' for volume based grainsize distribution, 'Count' for count based
    min_part_count: int = 0  # how many MP particles are required to be considered as a valid sample 
    rebinning: bool = False  # whether or not to aggregate sizes to coarser bins
    closing: int = 1  # make comp data closed to int value: 0 for no closure, 1 for fraction, 100 for percentages
    rebin_by_n: int = 5  # make sediment size bins coarser: sum up every n bins into one new
    vertical_merge: bool = True  # whether or not to merge the vertical dimension of the data (i.e. combine data at stations where surface and core data exists)
    warnow: bool = False  # if True: load available Warnow data and concatenate with SDD df for comparisons Warnow / Schlei
    cau_droplist: list = [  # CAU samples containing no data
        '20170425_G47',
        '20170426_G68',
        '20170426_G75',
        '20170426_G77',
        '20170426_G78',
        '20170427_G85',
    ]
    cau_impute: dict = {  # regression formulas to estimate missing sediment grain size data points from a regression of available values
        'TOC': '0.477*np.exp(0.039*perc_MUD)',  # R2=0.77
        'SED_D50': '-99.342*np.log(TOC)+235.451',  # R2=0.75
        'perc_MUD': '27.147*np.log(TOC)+22.62',  # R2=0.83
        'PC1': '-0.274*np.log(TOC)+0.337',  # R2= 0.82
        'PC2': '-0.01*TOC+0.051',  # R2=0.05
    }
    massConc_from_numConc: dict = {  # Correlation of "MassConcentration" against "Concentration" of IOW samples without outliers: S05, S32 (like in model) 
        'MassConcentration_predicted': '0.1452*Concentration_predicted**1.1898',  # R²=0.9102, based on power regression with MassConcentration as µg kg⁻¹, OBS: there is theoretical justifiaction to use power reg here: samples with high number conc are more likely to include very rare very large MP (which disproportionally increase MassConc)
        # 'MassConcentration_predicted': '0.9640*Concentration_predicted-425.9779',  # R²=0.9135, based on linear regression with MassConcentration as µg kg⁻¹, OBS: can yield negative values!
    }
    
    # Geospacial settings
    baw_epsg: int = 25832  # epsg code of the baw data
    restrict_tracers_to_depth: float = 30.0  # for BAW tracer particles depth values larger than this, will be replaced by an interpolation from their neighbours, set to 0 for no depth correction
    station_buffers: float = 444.0  # buffer radius in meters around each sample station , in which tracer ocurrences from the BAW simulation are counted
    dem_path: str = '../data/.DGM_Schlei_1982_bis_2002_UTM32.zip'  # path to the DEM of water depths
    dem_resolution: float = 5.0  # resolution of the digital elevation model in meters
    interpolation_resolution: float = 5.0  # spatial resolution for geospatial interpolation in metres
    interpolation_method: str = 'linear'  # {‘linear’, ‘nearest’, ‘cubic’} method for scipy.interpolate.griddata
    use_seasons: list = ['spring']  # which seasons to use for the tracer-based WWTP influence estimation: must be a list of one or more of 'spring', 'summer','autumn'
    sed_contact_dist: float = 0.01  # distance in meters to the sediment contact, below which a tracer is considered to have sedimented
    sed_contact_dur: int = 2  # number of timesteps a tracer has to be closer to the sediment than sed_contact_dist to be considered as sedimented
    arrest_on_nth_sedimentation: int = 0  # set to 0 for no arrest, otherwise any positive integer will truncate at the respective sedimentation event: e.g. set to 1 to only include traces before first sediment contact
    tracer_mean_time_fillna: int = 489  # fill value for NaNs in WWTP influence parameter calculated as mean time travelled until first entrance to a samples buffer zone: will be used for samples where no trace made it into their buffer zone (489 is one more than the last time step)

    # Settings for streamlit app filters (the actual values of these are controlled by the app filters)
    size_dim: str = 'vESD'  # which size dimension to use for the analysis
    lower_size_limit: float = 0.0  # the smallest particle size in µm included in the kde computation
    upper_size_limit: float = 5000.0  # the largest particle size in µm included in the kde computation
    lower_density_limit: float = 900.0  # the smallest particle density in kg/m3 included in the analysis
    upper_density_limit: float = 2000.0  # the largest particle density in kg/m3 included in the analysis
    size_filter_on_sed_grainsizes: bool = False  # whether to also filter grainsize df using the lower and upper_size_limit

    # KDE settings
    optimise_bw: bool = False  # if True: compute an individual bandwidth for each sample before computing the KDE
    bws_to_test: int = 100  # if optimise_bw = True: how many bandwidth values should be tried out?
    fixed_bw: float = 20.0  # if optimise_bw = False: fixed bandwidth value to use for all kde's
    kernel: str = 'gaussian'  # type of kernel to be used
    kde_weights: str = 'particle_volume_share'  # None for count-based distributions, 'particle_volume_share' for volume-based distributions or 'particle_mass_share' for mass-based distributions
    bin_conc: bool = False  # True: calculate MP conc. (#/kg) for individual size bins; False: calculate percentages

    # Settings for shape-constrained rpy2 KDE for particle heights
    height_low: float = 0.0  # lowest height value which should be selected for height update from KDE
    height_high: float = 20.0  # highest height value which should be selected for height update from KDE
    exceed_high_by: float = 50.0  # sampled heights from KDE may be up to this much bigger (in %) than height_high

    # Settings for the statsmodels GLM
    glm_family: str = 'Poisson'  # type of family to be used for the GLM
    glm_tweedie_power: float = 2.0  # Tweedie power for GLM; only used if family is Tweedie
    glm_link: str = None  # Link function for GLM; use None for default link of chosen family
    glm_power_power: float = 2.0  # Power exponent for the power link function; only used if link is Power
    glm_formula: str = f'{target} ~ {default_predictors[0]} + ' \
                       f'{default_predictors[1]}'

    # Settings for the CV notebook
    mutual_exclusive: list = [  # Mutual exclusive list (list of lists detailing predictors that are not allowed together in one model candidate)
        ['SED_MODE1', 'SED_D50'],
        ['SED_MODE1', 'PC1'],
        ['SED_MODE1', 'perc_MUD'],
        ['SED_MODE1', 'TOC'],
        ['SED_D50', 'PC1'],
        ['SED_D50', 'perc_MUD'],
        ['SED_D50', 'TOC'],
        ['perc_MUD', 'PC1'],
        ['perc_MUD', 'TOC'],
        ['PC1', 'TOC'],
        # ['Dist_Land', 'Depth']
        ]
    exclusive_keywords: list = ['WWTP']  # only feature_candidates sets with max 1 feature containing each keyword will be considered
    scorers: dict = {  # dictionary of scores to be used by gridsearch: values are lists of corresponding [scorer, gridsearch refit scorer string or callable]
                        'R2': [r2_score, 'r2'],
                        'MAE': [mean_absolute_error, 'neg_mean_absolute_error'],
                        'MAPE': [mean_absolute_percentage_error, 'neg_mean_absolute_percentage_error'],
                        'MedAE': [median_absolute_error, 'neg_median_absolute_error'],
                        'MedAPE': [median_absolute_percentage_error, make_scorer(median_absolute_percentage_error, greater_is_better=False)],
                        # 'MSE': [mean_squared_error, 'neg_mean_squared_error'],
                        # 'MSLE': [mean_squared_log_error, 'neg_mean_squared_log_error'],
                    }
    refit_scorer: str = 'R2'  # one of the keys in scoring dict above: will be used to refit at set best estimator of the gridsearch object
    select_best: str = 'median'  # type of average to be used to identify the best model of a gridsearch: can be 'median', 'mean' or 'iqm'
    ncv_mode: str = 'competitive'  # 'competitive' for running all activated model param sets against each other, 'comparative' for running separate repNCVs for each model param set
    log_path: str = '../data/exports/models/logs'  # default path to logfile
    log_file: str = 'model.log'  # default name for log file

Config.bandwidths: np.array = 10 ** np.linspace(0, 3,
                                                Config.bws_to_test)  # creates the range of bandwidths to be tested

regio_sep = {
    'Schlei_S1': 'inner',
    'Schlei_S1_15cm': 'inner',
    'Schlei_S2': 'inner',
    'Schlei_S2_15cm': 'inner',
    'Schlei_S3': 'inner',
    'Schlei_S4': 'inner',
    'Schlei_S4_15cm': 'inner',
    'Schlei_S5': 'outlier',
    'Schlei_S6': 'inner',
    'Schlei_S7': 'inner',
    'Schlei_S8': 'WWTP',
    'Schlei_S9': 'WWTP',
    'Schlei_S10': 'WWTP',
    'Schlei_S10_15cm': 'WWTP',
    'Schlei_S11': 'inner',
    'Schlei_S12': 'inner',
    'Schlei_S13': 'inner',
    'Schlei_S14': 'inner',
    'Schlei_S15': 'inner',
    'Schlei_S16': 'inner',
    'Schlei_S17': 'inner',
    'Schlei_S18': 'inner',
    'Schlei_S18_15cm': 'inner',
    'Schlei_S19': 'inner',
    'Schlei_S20': 'inner',
    'Schlei_S21': 'outer', 
    'Schlei_S22': 'outer',
    'Schlei_S23': 'outer',
    'Schlei_S24': 'outer',
    'Schlei_S25': 'outer',
    'Schlei_S26': 'outer',
    'Schlei_S27': 'outer',
    'Schlei_S29': 'outer',
    'Schlei_S30': 'outer',
    'Schlei_S31': 'outer',
    'Schlei_S32': 'outlier',
}

shortnames = {
    'Acrylic resin': 'AcrR',
    'Acrylnitril-Butadien-Styrol-Copolymer': 'ABS',
    'Alkyd resin': 'AlkR',
    'Epoxy resin': 'EPX',
    'Ethylen-Vinylacetat-Copolymer': 'EVA',
    'Nitrile rubber': 'NBR',
    'Poly (ethylene terephthalate)': 'PET',
    'Poly (methyl methacrylate)': 'PMMA',
    'Poly (vinyl chloride)': 'PVC',
    'Polyamide ': 'PA',
    'Polycaprolacton': 'PCL',
    'Polycarbonate': 'PC',
    'Polyethylene': 'PE',
    'Polyhydroxybutyrat': 'PHB',
    'Polylactide': 'PLA',
    'Polyoxymethylene': 'POM',
    'Polypropylene': 'PP',
    'Polysiloxane': 'SIL',
    'Polystyrene': 'PS',
    'Polyurethane ': 'PU',
    'Silicone-rubber': 'SR',
    'Styrene-butadiene-styrene block copolymer': 'SBS',
    'Schlei_S1': 'S01',
    'Schlei_S1_15cm': 'S01d',
    'Schlei_S2': 'S02',
    'Schlei_S2_15cm': 'S02d',
    'Schlei_S3': 'S03',
    'Schlei_S4': 'S04',
    'Schlei_S4_15cm': 'S04d',
    'Schlei_S5': 'S05',
    'Schlei_S6': 'S06',
    'Schlei_S7': 'S07',
    'Schlei_S8': 'S08',
    'Schlei_S9': 'S09',
    'Schlei_S10': 'S10',
    'Schlei_S10_15cm': 'S10d',
    'Schlei_S11': 'S11',
    'Schlei_S12': 'S12',
    'Schlei_S13': 'S13',
    'Schlei_S14': 'S14',
    'Schlei_S15': 'S15',
    'Schlei_S16': 'S16',
    'Schlei_S17': 'S17',
    'Schlei_S18': 'S18',
    'Schlei_S18_15cm': 'S18d',
    'Schlei_S19': 'S19',
    'Schlei_S20': 'S20',
    'Schlei_S21': 'S21',
    'Schlei_S22': 'S22',
    'Schlei_S23': 'S23',
    'Schlei_S24': 'S24',
    'Schlei_S25': 'S25',
    'Schlei_S26': 'S26',
    'Schlei_S27': 'S27',
    'Schlei_S28': 'S28',
    'Schlei_S29': 'S29',
    'Schlei_S30': 'S30',
    'Schlei_S31': 'S31',
    'Schlei_S32': 'S32',
    'D10 (µm)': 'SED_D10',
    'D50 (µm)': 'SED_D50',
    'D90 (µm)': 'SED_D90',
    'perc CLAY': 'perc_CLAY',
    'perc MUD': 'perc_MUD',
    'perc SAND': 'perc_SAND',
    'perc GRAVEL': 'perc_GRAVEL',
    'MODE 1 (µm)': 'SED_MODE1',
    'MODE 2 (µm)': 'SED_MODE2',
    'MODE 3 (µm)': 'SED_MODE3',
}

densities = {  # Polymer densities in kg m⁻³
    'generic': 1141,  # assume a general average density where exact density is not available, ref: https://doi.org/10.1021/acs.est.0c02982
    'Acrylic resin': 1600, #when dried resins are denser (averagely 1600 kg m⁻³ than original liquid polymer densities, see calculation in Enders et al. 2019
    'Acrylnitril-Butadien-Styrol-Copolymer': 1060,  #Enders2019, Stuart2002
    'Alkyd resin': 1600,
    'Epoxy resin': 1600,
    'Ethylen-Vinylacetat-Copolymer': 935,  #Enders2019, Stuart2002
    'Nitrile rubber': 1000, #1250 # MatWeb
    'Poly (ethylene terephthalate)': 1395,  #Enders2019, Stuart2002
    'Poly (methyl methacrylate)': 1180,  #Enders2019, Stuart2002
    'Poly (vinyl chloride)': 1395,  #Enders2019, Stuart2002
    'Polyamide ': 1135,  #Enders2019, Stuart2002
    'Polycaprolacton': 1150, # MatWeb #polymerdatabase ?
    'Polycarbonate': 1210,  #Enders2019, Stuart2002
    'Polyethylene': 940,  #Enders2019, Stuart2002
    'Polyhydroxybutyrat': 1250, # MatWeb
    'Polylactide': 1250, # MatWeb
    'Polyoxymethylene': 1415, # MatWeb
    'Polypropylene': 910, #Enders2019, Stuart2002 850 #MatWeb 910
    'Polysiloxane': 1250, # MatWeb
    'Polystyrene': 1050,  #Enders2019, Stuart2002
    'Polyurethane ': 1600, #1230,  #Enders2019, Stuart2002 or resin?
    'Silicone-rubber': 1200, # MatWeb
    'Styrene-butadiene-styrene block copolymer': 940, # MatWeb
    'Unknown': None
}

dist_params = {  # paramters of particle properties for generating distributions, reference: https://doi.org/10.1038/s41578-021-00411-y 
    'particle_type': ['SED',     'SED',    'SED',       'POM',     'POM',    'POM',       'MP',                  'MP',      'MP'       ],
    'prop':          ['density', 'size',   'longevity', 'density', 'size',   'longevity', 'density',             'size',    'longevity'],
    'dist_name':     ['triang',  'triang', 'triang',    'triang',  'triang', 'triang',    'triang',              'triang',  'triang'   ],
    'low':           [ 1100,      0.06,     10**4.3,     800,       0.2,      10**1.2,     800,                   1,         10**1.8   ],
    'peak':          [ 2650,      30,       10**6.3,     1000,      5,        10**3.1,     densities['generic'],  20,        10**5     ],
    'high':          [ 2800,      2000,     10**9,       1600,      2500,     10**4.5,     2000,                  1000,      10**7.2   ]
}

baw_tracer_reduction_factors = {  # Amounts of BAW tracers of each simulated size class (ESD) need to be adjusted to represent the actual relative frequency of particles of the respective size class. Fractions here originate from the KDE of MP volume-based ESD distribution.
    0:0,
    18: 1,#0.19,
    50: 0,#0.48,
    100:0,#1,
    300:0#0.115
}


def getLogger(log_path=Config.log_path, log_file=Config.log_file):
    # make path a Path object
    log_path = Path(log_path)
    # ensure existence of log dir
    log_path.mkdir(parents=True, exist_ok=True)
    # create logger
    logger = logging.getLogger()
    # set level
    logger.setLevel(logging.INFO)
    # set logging format
    logFormatter = logging.Formatter('%(message)s')  # ('%(asctime)s - %(levelname)s - %(message)s')
    # add file handler
    fileHandler = logging.FileHandler(str(log_path) + '/' + log_file, encoding='utf-8')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    # # add console handler
    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setLevel(logging.INFO)
    # consoleHandler.setFormatter(logFormatter)
    # logger.addHandler(consoleHandler)
    logger.handlers[0].setFormatter(logFormatter)  # also format the root handler
    return logger
