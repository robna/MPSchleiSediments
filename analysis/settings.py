import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

target = 'Concentration'

predictors = ['Dist_WWTP', 'TOC', 'Q("D50 (µm)")"', 'PC1', 'PC2']  # columns to be used as predictors


class Config:
    min_part_count: float = 0  # how many MP particles are required to be considered as a valid sample 

    rebinning: bool = False  # whether or not to aggregate sizes to coarser bins
    closing: int = 100 # make comp data closed to int value: 0 for no closure, 1 for fraction, 100 for percentages
    rebin_by_n: int = 5  # make sediment size bins coarser: sum up every n bins into one new

    size_dim: str = 'vESD'  # which size dimension to use for the analysis
    lower_size_limit: float = 0  # the smallest particle size in µm included in the kde computation
    upper_size_limit: float = 5000  # the largest particle size in µm included in the kde computation
    lower_density_limit: float = 900  # the smallest particle density in kg/m3 included in the analysis
    upper_density_limit: float = 2000  # the largest particle density in kg/m3 included in the analysis

    # allowed_zeros: float = 1  # the fraction of allowed zeros in a sediment size bin  # TODO: not used anymore

    # kde_steps: int = 496  # number points on the size axis where the kde is defined (like number of bins in histogram)  # TODO: can this be deleted?
    optimise_bw: bool = False  # if True: compute an individual bandwidth for each sample before computing the KDE
    bws_to_test: int = 20  # if optimise_bw = True: how many bandwidth values should be tried out?
    fixed_bw: int = 75  # if optimise_bw = False: fixed bandwidth value to use for all kde's
    kernel: str = 'gaussian'  # type of kernel to be used

    # shape-constrained KDE for particle heights
    height_low: float = 0  # lowest height value which should be selected for height update from KDE
    height_high: float = 30  # highest height value which should be selected for height update from KDE

    bin_conc: bool = False  # True: calculate MP conc. (#/kg) for individual size bins; False: calculate percentages

    MPlow: int = 50  # TODO: are these still needed?
    MPup: int = 250
    SEDlow: int = 50
    SEDup: int = 250

    ILR_transform: bool = True  # if True: use the ILR transform for sediment size data before dimensionality reduction

    glm_family: str = 'Poisson'  # type of family to be used for the GLM
    glm_link: str = None  # Link function for GLM; use None for default link of chosen family
    glm_formula: str = f'{target} ~ {predictors[0]} + ' \
                       f'{predictors[1]}'


# creates the x-axis data for the prob. dist. func.
# Config.x_d: np.array = np.linspace(Config.lower_size_limit, Config.upper_size_limit, Config.kde_steps)  # TODO: can this be deleted?

Config.bandwidths: np.array = 10 ** np.linspace(2, 3,
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
    'warnow': 'warnow'
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
    'Schlei_S32': 'S32'
}

densities = {  # Polymer densities in kg m⁻³
    'generic': 1141,  # assume a general average density where exact density is not available, ref: https://doi.org/10.1021/acs.est.0c02982
    'Acrylic resin': None,
    'Acrylnitril-Butadien-Styrol-Copolymer': 1040,
    'Alkyd resin': None,
    'Epoxy resin': None,
    'Ethylen-Vinylacetat-Copolymer': 951,
    'Nitrile rubber': None,
    'Poly (ethylene terephthalate)': 1350,
    'Poly (methyl methacrylate)': 1190,
    'Poly (vinyl chloride)': 1400,
    'Polyamide ': 1130,
    'Polycaprolacton': 1145,
    'Polycarbonate': 1200,
    'Polyethylene': 940,
    'Polyhydroxybutyrat': 1225,
    'Polylactide': 1320,
    'Polyoxymethylene': 1415,
    'Polypropylene': 900,
    'Polysiloxane': None,
    'Polystyrene': 1043,
    'Polyurethane ': 1190,
    'Silicone-rubber': None,
    'Styrene-butadiene-styrene block copolymer': None,
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
