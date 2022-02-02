import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

predictors = ['Dist_WWTP', 'TOC', 'D50 (µm)', 'PC1', 'PC2']  # columns to be used as predictors


class Config:
    min_part_count: float = 20  # how many MP particles are required to be considered as a valid sample 

    rebinning: bool = False  # whether or not to aggregate sizes to coarser bins
    closing: int = 100 # make comp data closed to int value: 0 for no closure, 1 for fraction, 100 for percentages
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

shortnames = {
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
    'Polycaprolacton': None,
    'Polycarbonate': 1200,
    'Polyethylene': 940,
    'Polyhydroxybutyrat': None,
    'Polylactide': None,
    'Polyoxymethylene': None,
    'Polypropylene': 900,
    'Polysiloxane': None,
    'Polystyrene': 1043,
    'Polyurethane ': 1190,
    'Silicone-rubber': None,
    'Styrene-butadiene-styrene block copolymer': None,
    'Unknown': None
}
