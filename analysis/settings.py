import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

target = 'Concentration'

predictors = ['Dist_WWTP', 'TOC', 'Q("D50 (µm)")"', 'PC1', 'PC2']  # columns to be used as predictors


class Config:
    # General data preparation settings
    min_part_count: float = 0  # how many MP particles are required to be considered as a valid sample 
    rebinning: bool = False  # whether or not to aggregate sizes to coarser bins
    closing: int = 1  # make comp data closed to int value: 0 for no closure, 1 for fraction, 100 for percentages
    rebin_by_n: int = 5  # make sediment size bins coarser: sum up every n bins into one new
    
    # Geospacial settings
    baw_epsg: int = 25832  # epsg code of the baw data
    restrict_tracers_to_depth: float = 30  # for BAW tracer particles depth values larger than this, will be replaced by an interpolation from their neighbours, set to 0 for no depth correction
    station_buffers: int = 222  # buffer radius in meters around each sample station , in which tracer ocurrences from the BAW simulation are counted
    dem_resolution: float = 5  # resolution of the digital elevation model in meters
    sed_contact_dist: float = 0.01  # distance in meters to the sediment contact, below which a tracer is considered to have sedimented
    sed_contact_dur: int = 2  # number of timesteps a tracer has to be closer to the sediment than sed_contact_dist to be considered as sedimented
    arrest_on_nth_sedimentation: int = 3  # set to 0 for no arrest, otherwise any positive integer will truncate at the respective sedimentation event: e.g. set to 1 to only include traces before first sediment contact
    tracer_mean_time_fillna: int = 489  # fill value for NaNs in WWTP influence parameter calculated as mean time travelled until first entrance to a samples buffer zone: will be used for samples where no trace made it into their buffer zone (489 is one more than the last time step)

    # Settings for streamlit app filters (the actual values of these are controlled by the app filters)
    size_dim: str = 'vESD'  # which size dimension to use for the analysis
    lower_size_limit: float = 0  # the smallest particle size in µm included in the kde computation
    upper_size_limit: float = 5000  # the largest particle size in µm included in the kde computation
    lower_density_limit: float = 900  # the smallest particle density in kg/m3 included in the analysis
    upper_density_limit: float = 2000  # the largest particle density in kg/m3 included in the analysis

    # KDE settings
    optimise_bw: bool = False  # if True: compute an individual bandwidth for each sample before computing the KDE
    bws_to_test: int = 200  # if optimise_bw = True: how many bandwidth values should be tried out?
    fixed_bw: int = 75  # if optimise_bw = False: fixed bandwidth value to use for all kde's
    kernel: str = 'gaussian'  # type of kernel to be used
    kde_weights: str = 'particle_volume_share'  # None for count-based distributions, 'particle_volume_share' for volume-based distributions or 'particle_mass_share' for mass-based distributions
    bin_conc: bool = False  # True: calculate MP conc. (#/kg) for individual size bins; False: calculate percentages

    # Settings for shape-constrained rpy2 KDE for particle heights
    height_low: float = 0  # lowest height value which should be selected for height update from KDE
    height_high: float = 20  # highest height value which should be selected for height update from KDE
    exceed_high_by: float = 50  # sampled heights from KDE may be up to this much bigger (in %) than height_high

    # Settings for the statsmodels GLM
    glm_family: str = 'Poisson'  # type of family to be used for the GLM
    glm_link: str = None  # Link function for GLM; use None for default link of chosen family
    glm_formula: str = f'{target} ~ {predictors[0]} + ' \
                       f'{predictors[1]}'


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
    18: 0.19,
    50: 0.48,
    100: 1,
    300: 0.115
}