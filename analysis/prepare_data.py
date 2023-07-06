import numpy as np
import pandas as pd
from patsy import dmatrix
from skbio.stats.composition import closure
from settings import densities, regio_sep, shortnames, sediment_data_filepaths, Config
from KDE_utils import unbound_kde
import outliers
import geo


def fix_gradistat_names(df):
    df = df.rename(columns=shortnames)
    return df.loc[:, [col for col in df.columns if col in shortnames.values()]]


def merge_vertical(df, avg=False):
    """
    Merges the vertical samples of the same site into one sample
    """

    if df.index.name == 'Sample':
        df = df.reset_index()
    df.Sample = df.Sample.str.replace('_15cm', '')
    if avg:
        df = df.groupby('Sample').mean(numeric_only=True)
    return df


def get_pdd():
    """
    Reads the MP particle domain data (PDD) from the CSV file and does some preprocessing
    """
    
    mp_pdd = pd.read_csv('../data/mp_pdd.csv', index_col=0)
    mp_pdd.drop(columns=['Site_name', 'Compartment', 'Contributor', 'Project', 'Particle_name', 'lab_blank_ID', 'sample_ID'], inplace=True)  # columns not needed for analysis
    mp_pdd = height_vol_dens_mass(mp_pdd)  # calculate particle weights
    mp_pdd = geom_mean_3d(mp_pdd)  # calculate the geometric mean from 3 size dimensions
    mp_pdd, gpn = outliers.low_freq_out(mp_pdd)  # remove low frequency outliers
    # mp_pdd = mp_pdd.loc[mp_pdd.Shape=='irregular']  # filter to only use fibres or irregulars
    mp_pdd['polymer_type'] = mp_pdd['polymer_type'].map(shortnames).fillna(mp_pdd['polymer_type'])  # use abbreviations for polymer names but retain original names for polymers not present in shortnames
    mp_pdd.columns = mp_pdd.columns.str.replace("[\[( )\]]", "", regex=True)  # remove brackets from column names

    # mp_pdd = mp_pdd.loc[~mp_pdd.polymer_type.isin(['AcrR', 'AlkR', 'EPX'])]  # exclude paint flakes

    if Config.vertical_merge:
        rng = np.random.RandomState(42)
        for n, g in mp_pdd.groupby('Sample'):
            srn = float(rng.random(1)) / 1_000_000  # generate a small random number
            mp_pdd.loc[mp_pdd.Sample == n, 'Sampling_weight_kg'] = g['Sampling_weight_kg'] + srn  # add a small random number to every original sample's mass in order to make it unique. After vertical merge, these small additions will be removed by rounding
        mp_pdd = merge_vertical(mp_pdd)

    return mp_pdd


def get_grainsizes(
        IOW_sed_gs_path=sediment_data_filepaths[f'IOW_{Config.sediment_grainsize_basis}'],
        CAU_sed_gs_path=sediment_data_filepaths['CAU_Volume_logscale']):
    """
    Reads the sediment grainsize data from the CSV file and does some preprocessing
    """

    grainsize_iow = pd.read_csv('../' + IOW_sed_gs_path)

    if Config.vertical_merge:
        grainsize_iow = merge_vertical(grainsize_iow)

    grainsize_cau = pd.read_csv('../' + CAU_sed_gs_path)
    grainsize_cau.dropna(subset=grainsize_cau.iloc[:,1:].columns, how='all', inplace=True)  # CAU sediment data contains empty sammples which are dropped here

    # Get the binning structure of the imported sediment data and optionally rebin it (make binning coarser) for faster computation
    grainsize_iow, boundaries = sediment_preps(grainsize_iow)
    grainsize_cau, _ = sediment_preps(grainsize_cau)

    return grainsize_iow, grainsize_cau, boundaries


def sed_bulk_density(smpls):
    smpls['SedDryBulkDensity_JiaEtAl'] = -8.86 * smpls.perc_MUD + 1518.91  # DOI: 10.1016/S0272-7714(02)00406-7
    # smpls['SedDryBulkDensity_BirdEtAl'] = np.exp(0.31686604 + (-0.0032) * smpls.perc_MUD * np.log(np.maximum(smpls.perc_MUD, 0.0000001)) + (-0.0665) * smpls.TOC ** 0.5 * np.log(np.maximum(smpls.TOC, 0.0000001))) * 1000  # DOI:10.1017/S0033822200039734
    smpls['SedDryBulkDensity_vanRijnEtAl'] = (1 - smpls.TOC*2.22/ 100) * (400 * (smpls.perc_CLAY / 100) + 800 * ((smpls.perc_MUD - smpls.perc_CLAY) / 100) + 1600 * (smpls.perc_SAND / 100))  # DOI: 10.1061/(ASCE)WW.1943-5460.0000483
    ## Some stations have NaN in perc_Clay and perc_SAND (but not in perc_MUD) and therefore get NaN in SedDryBulkDensity_vanRijnEtAl

    smpls['SedDryBulkDensity'] = smpls[['SedDryBulkDensity_JiaEtAl', 'SedDryBulkDensity_vanRijnEtAl']].apply(lambda x: x.mean(), axis=1)  # using average of van Rjin and Jia only, because Bird is from Mangroves

    return smpls


def patsy_transform(formula, df):
    '''
    Calculate new columns for a df using patsy formulas.
    dmatrix with "-1" in formula returns results as df without intercept column.
    Using squeeze, turns it into a series.
    More info: https://learn-scikit.oneoffcoder.com/patsy.html
    '''
    return dmatrix(f'I({formula}) - 1', df, return_type='dataframe').squeeze() 


def impute_cau(sdd_cau):
    '''
    Calculates missing values in sediment grainsize data of CAU stations. 
    In settings > Config.cau_impute there is a dict with regression equations
    for each sediment grainsize parameter where it is necessary to impute missing
    values.
    '''

    sdd_cau = sdd_cau[~sdd_cau.Sample.isin(Config.cau_droplist)]  # remove samples lacking data

    for k, v in Config.cau_impute.items():
        missing = sdd_cau[k].isna()
        imputes = patsy_transform(v, sdd_cau.loc[missing])
        sdd_cau.loc[missing, k] = imputes
        # print(f'imputed {k}\n', sdd_cau.loc[missing, k], f'\nLength: {len(sdd_cau.loc[missing, k])}\n\n\n')

    sdd_cau = sed_bulk_density(sdd_cau)
    return sdd_cau


def get_medians_from_size_dist(size_pmf, boundaries_dict, medians_name):
    size_cpmf = size_pmf.cumsum(axis=1) # cumulative sum of the probability mass functions
    size_bin_number_containing_median = (size_cpmf.T.reset_index(drop=True).T - 0.5).abs().idxmin(axis=1)  # Find the size bins which enclose the 50% of the probability mass. OBS: if choosing a different colsing value than 1, the subtractionnn of 0.5 needs to be adjusted!!
    medians = pd.DataFrame(boundaries_dict).center[size_bin_number_containing_median]
    medians.name = medians_name
    medians.index = size_bin_number_containing_median.index
    return medians


def get_mode1s_from_size_dist(size_pmf, boundaries_dict, mode1s_name):
    size_bin_number_containing_mode1 = size_pmf.T.reset_index(drop=True).T.idxmax(axis=1)  # Find the size bins with the maximum probability (i.e. the mode of the distribution)
    mode1s = pd.DataFrame(boundaries_dict).center[size_bin_number_containing_mode1]
    mode1s.name = mode1s_name
    mode1s.index = size_bin_number_containing_mode1.index
    return mode1s

def height_vol_dens_mass(df):
    """Adds height, volume, density and mass to each MP particle"""

    # Adjust heights (a.k.a. Size_3): not all Gepard heights can be trusted, see inline comments below
    # ------------------------------------------------------------------------------------------------
    df['orig_gepard_height'] = df['Size_3_[µm]']  # save original height for later use

    # There should be no more particles without height. Hence, the operation below is commented out and only here for documentation.
    #df.loc[(df['Shape'] == 'irregular') & (df['Size_3_[µm]'].isna())  # where no Gepard height exists, use regressed height
         #|(df['Size_3_[µm]'] == 0)  # due to insufficient z-resolution, small particles tend to cluster at 0 µm height
         #|(df['Size_3_[µm]'] == 20)
         #,'Size_3_[µm]'] = (1.919764*np.log(df['Size_2_[µm]'])**0.660945)  # regression from particle with existing Gepard height data
    
    df.loc[(df['Shape'] == 'irregular') & (df['Size_3_[µm]'] > df['Size_1_[µm]']), 'Size_3_[µm]'] = df['Size_2_[µm]']  # when irreg higher than long, use Size_1 as height
   
    df.loc[(df['Shape'] == 'fibre') & (df['Size_3_[µm]'] > df['Size_2_[µm]']), 'Size_3_[µm]'] = df['Size_2_[µm]'] 
    df.loc[(df['Shape'] == 'fibre') & (df['Size_3_[µm]'] < 5), 'Size_3_[µm]'] = df['Size_2_[µm]'] 
   
    #df.loc[df['Shape'] == 'fibre', 'Size_3_[µm]'] = df['Size_2_[µm]']  # Heights for fibres (alternative 1): use Size_2 as height for all fibres
  
    # df.loc[(df['Shape'] == 'fibre') & (  # Heights for fibres (alternative 2): when fibre without height OR higher than wide OR < 5: use Size_2 as height
    #         (ds (df['Size_3_[µm]'].isna())
    #         ), 'Size_3_[µm]']= df['Size_2_[µm]']

    # where particle Gepard height is low, get height from shape-constrained height distribution of manually measured particles
    # this is to mitigate the effect of the low z-resolution of the Gepard, which produces wrong and clustered height data for small particles
    hw = df.loc[(df['Shape'] == 'irregular') &  # generate weights for the scKDE-sampled height values
                (df['Size_3_[µm]'] >= Config.height_low) &
                (df['Size_3_[µm]'] <= Config.height_high)
                , 'Size_3_[µm]'
               ]#.apply(lambda x: 0.8 *  # weight for scKDE-sampled height values at the lower end of the height range, default 80%
                    # max(0,  # don't allow negative weights
                    #     min(1,(  # don't allow weights > 1
                    #         (Config.height_high - x) / (Config.height_high - Config.height_low)  # calculate the individual weights
                    #           )
                    #        )
                    #    )
                    #   )
    # sampled_values, kde, cdf = bound_kde(hw.shape[0], Config.height_low, Config.height_high, bw=3, method="weight")
    # # blend the scKDE-sampled height values with the Gepard height values applying a linear weight decrease with heights increasing within the [low, high] range
    # df.loc[hw.index, 'Size_3_[µm]'] = hw * sampled_values + (1 - hw) * df.loc[hw.index, 'Size_3_[µm]']

    sampled_values, kde = unbound_kde(hw.shape[0], Config.height_low, Config.height_high)
    hw = hw.sample(frac=1, random_state=21)
    hw.sort_values(inplace=True)
    hw[:] = sampled_values
    df.loc[hw.index, 'Size_3_[µm]'] = hw
    

    # Calculate volumes
    # -----------------
    # Estimate volumes, ref: # from https://doi.org/10.1016/j.watres.2018.05.019  -> not used anymore, as Kristina showed it to be inaccurate
    # df['size_dimension_decrease_factor'] = df.loc[df.Shape == 'irregular', 'Size_2_[µm]'] / df.loc[df.Shape == 'irregular', 'Size_1_[µm]']  # calculates the factor, by which size dimishes from 1st to 2nd dimension
    # df.loc[df['Shape'] == 'irregular', 'particle_volume_[µm3]'] = 4/3 * np.pi * (df['Size_1_[µm]']/2) * (df['Size_2_[µm]']/2)**2 * df['size_dimension_decrease_factor']  # ellipsoid volume with 3rd dim = 2nd dim * (2nd dim / 1st dim)
    df.loc[df['Shape'] == 'irregular', 'particle_volume_[µm3]'] = (  # ellipsoid volume: 4/3 * π * (a/2) * (b/2) * (c/2)
        4 / 3 * np.pi *
        (df['Size_1_[µm]'] / 2) *
        (df['Size_2_[µm]'] / 2) *
        (df['Size_3_[µm]'] / 2)
     )
    df.loc[df['Shape'] == 'fibre', 'particle_volume_[µm3]'] = (  # elliptical cylinder volume: π * (a/2) * (b/2) * (length)
        np.pi *
        (df['Size_2_[µm]'] / 2) *
        (df['Size_3_[µm]'] / 2) *
        df['Size_1_[µm]']
     )
    df['vESD'] = ((6 * df['particle_volume_[µm3]'])/np.pi)**(1/3)  # volume equivalent spherical diameter

    # Get densities
    # -------------
    df['density'] = df['polymer_type'].map(densities)
    df['density'].fillna(
        densities['generic'],
        inplace=True)  # assume a general average density where exact density is not available, ref: https://doi.org/10.1021/acs.est.0c02982
    
    # Get masses
    # ----------
    df['particle_mass_[µg]'] = df['particle_volume_[µm3]'] * df['density'] * 1e-9
    # calculate volume and mass share of each particle grouped by Sample
    for sample_name, Sample in df.groupby('Sample', sort=False):
        Sample['particle_volume_share'] = Sample['particle_volume_[µm3]'] / Sample['particle_volume_[µm3]'].sum()
        df.loc[Sample.index, 'particle_volume_share'] = Sample['particle_volume_share']
        Sample['particle_mass_share'] = Sample['particle_mass_[µg]'] / Sample['particle_mass_[µg]'].sum()
        df.loc[Sample.index, 'particle_mass_share'] = Sample['particle_mass_share']

    return df

def geom_mean_3d(df):
    df['size_geom_mean_3d'] = (df['Size_1_[µm]'] * df['Size_2_[µm]'] * df['Size_3_[µm]']) ** (1/3)
    return df


def aggregate_SDD(mp_pdd):
    """Calculates certain Sample domain data (SDD) aggregation from the particle domain data (PDD)"""

    if isinstance(mp_pdd, pd.DataFrame):  # if raw DF is submitted instead of groupby object, then group first
        mp_pdd = mp_pdd.groupby('Sample')

    mp_sdd = mp_pdd.agg(
        Frequency=('Sample', 'count'),
        FrequencyA500=('Size_1_µm', lambda x: (x >= 500).sum()),
        FrequencyB500=('Size_1_µm', lambda x: (x < 500).sum()),
        MPvolume=('particle_volume_µm3', 'sum'),
        MPmass=('particle_mass_µg', 'sum'),
        Mass=('Sampling_weight_kg', lambda x: round(x.unique().sum(), 3)),
        Split=('Fraction_analysed', lambda x: x.unique().mean()),
        MP_D50=(Config.size_dim, np.median)
        # MP_D50_A500 = ('size_geom_mean' >= 500.median()),
        # MP_D50_B500 = ('size_geom_mean', lambda x: (x<500).median())
    ).reset_index()

    mp_sdd['Concentration'] = round(mp_sdd['Frequency'] / mp_sdd['Mass'])
    mp_sdd['ConcentrationA500'] = round(mp_sdd['FrequencyA500'] / mp_sdd['Mass'])
    mp_sdd['ConcentrationB500'] = round(mp_sdd['FrequencyB500'] / mp_sdd['Mass'])
    mp_sdd['ConcentrationA500_div_B500'] = mp_sdd['ConcentrationA500'] / mp_sdd['ConcentrationB500']
    mp_sdd['VolumeConcentration'] = round(mp_sdd['MPvolume'] / mp_sdd['Mass'])
    mp_sdd['MassConcentration'] = round(mp_sdd['MPmass'] / mp_sdd['Mass'])
    return mp_sdd


def additional_sdd_merging(mp_sdd, how='left'):
    """Takes SDD and amends it with corresponding sediment data"""

    # import d50 values  # TODO: commented out, as we use D50 and <63 from Gradistat; may be deleted (also in merge section)
    # sed_d50 = pd.read_csv('../csv/Schlei_Sed_D50_<63.csv', index_col=0)

    # import gradistat results
    IOW_sed_gradistat_path=sediment_data_filepaths[f'IOW_GRADISTAT_{Config.sediment_grainsize_basis}']
    sed_gradistat = pd.read_csv(f'../{IOW_sed_gradistat_path}', index_col=0)
    if Config.vertical_merge:
        sed_gradistat = merge_vertical(sed_gradistat, avg=True)
    sed_gradistat = fix_gradistat_names(sed_gradistat)

    # import organic matter size, TOC, Hg data
    sed_om = pd.read_csv('../data/Schlei_OM.csv', index_col=0)
    if Config.vertical_merge:
        sed_om = merge_vertical(sed_om, avg=True)

    # import sampling log data
    slogs = pd.read_csv('../data/Metadata_IOW_sampling_log.csv', index_col=0)
    if Config.vertical_merge:
        slogs = merge_vertical(slogs, avg=True)

    # # import distance to waste water treatment plant  # TODO: can be removed (also in merge section): Dist_WWTP now included in slogs
    # dist_wwtp = pd.read_csv('../data/Schlei_Sed_Dist_WWTP.csv', index_col=0)

    # merge with mp per station
    mp_sdd_amended = pd.merge(mp_sdd, slogs.reset_index()[
        ['Sample', 'Depth', 'LON', 'LAT', 'Dist_Marina', 'Dist_WWTP', 'Dist_WWTP2']], on=['Sample'], how=how).merge(  # add metadata
        # sed_d50.reset_index(), on=['Sample'], how=how).merge(  # add sediment D50
        sed_gradistat.reset_index(), on=['Sample'], how=how).merge(  # add sediment gradistat
        sed_om.reset_index()[['Sample', 'OM_D50', 'TOC', 'Hg', 'TIC']], on=['Sample'], how=how).merge(  # add OM data
        # dist_wwtp.reset_index(), on=['Sample'], how=how).merge(  # add distance to WWTP
        pd.DataFrame.from_dict(regio_sep, orient='index', columns=['regio_sep']), left_on='Sample', right_index=True, how='left')  # add flags for regions

    # calculate distance from shore
    mp_sdd_amended['Dist_Land'] = geo.get_distance_to_shore(mp_sdd_amended['LON'], mp_sdd_amended['LAT'])
    # calculate WWTP_influence from BAW tracer model data
    mp_sdd_amended = geo.get_wwtp_influence(mp_sdd_amended, tracks_file='../data/BAW_tracer_simulations.zip', file_postfix='_IOW')
    
    # concatenate with Warnow data: only activate if you want to use Warnow data for comparison
    if Config.warnow:
        warnow = pd.read_csv('../data/Warnow_sdd.csv', index_col=0)
        warnow["ConcentrationA500"] = warnow["Concentration"]
        mp_sdd_amended = pd.concat([mp_sdd_amended, warnow], sort=False)
    
    # optionally: uncomment to export the final data
    # sdd_iow.to_csv('../csv/MP_Stats_SchleiSediments.csv')

    mp_sdd_amended = sed_bulk_density(mp_sdd_amended)

    return mp_sdd_amended


def melt_size_ranges(df, value_name):
    """
    Converts df with size bins or ranges in rows and samples in columns into a long-format df with 
    """

    melted_df = df.reset_index().melt(id_vars='Sample', var_name='ranges', value_name=value_name)
    melted_df[['lower', 'upper']] = melted_df.ranges.str.split('_', expand=True).astype(float)
    melted_df.drop(columns='ranges', inplace=True)

    return melted_df


def merge_size_ranges(df1, value_name1, df2, value_name2, cart_prod=False):
    """
    Merges (after sending through melt) MP and sediment DFs.
    Expects Dataframe 1 followed by a string argument
    for the name of the value column in the merged DF.
    Then the same for Dataframe 2.
    
    Arg 'cart_prod = False' (default):
    It returns the merged-melted df with rows for each
    stations MP and sed concentration per single size range.
    
    Arg 'cart_prod = True':
    It returns the merged-melted df with rows for each
    stations MP concentration per single size range repeated
    for each possible combination with a single size range
    frequency of sediments. I.e. a cartesian product of the
    two dataframes (without cross-listing samples)
    """

    melted1 = melt_size_ranges(df1, value_name1)
    melted2 = melt_size_ranges(df2, value_name2)
    if cart_prod:
        df = melted1.merge(melted2, how='cross')  # make cartesian product df, samples are crossed too!
        df = df.loc[df.Sample_x == df.Sample_y].drop(['Sample_y'],  # only keep entries where samples match
                                                     axis=1).rename(columns=
                                                                    {'Sample_x': 'Sample',
                                                                     'lower_x': 'lower_MP',
                                                                     'upper_x': 'upper_MP',
                                                                     'lower_y': 'lower_SED',
                                                                     'upper_y': 'upper_SED'})
    else:
        df = melted1.merge(melted2, on=['Sample', 'lower', 'upper'])

    return df


def equalise_mp_and_sed(mp, sed):
    """
    Various data harmonisation steps on DFs containing per size bin data of MP concentrations (mp)
    and sediment frequencies (sed). See inline comments for separate steps.
    """
    
    mp = mp.copy()
    sed= sed.copy()

    # make mp and sed contain only columns (size ranges) and rows (samples), which they have in common
    mp.drop([col for col in mp.columns if col not in sed.columns], axis=1, inplace=True)
    sed.drop([col for col in sed.columns if col not in mp.columns], axis=1, inplace=True)
    mp.drop([row for row in mp.index if row not in sed.index], axis=0, inplace=True)
    sed.drop([row for row in sed.index if row not in mp.index], axis=0, inplace=True)

    # Normalise to summed conc of all size bins,
    # i.e. turn from concentrations to percentage abundances.
    # Can be omitted by making bin_conc = True in settings Config.
    # In this case values will continue to represent concentrations.
    # if not Config.bin_conc:
    #     mp = mp.apply(lambda x: x / x.sum() * 100, axis=1)  # deactiviated as Config.bin_conc is now handled in KDE_utils.probDens2conc. If no problems occur, this whole part can be removed.

    mp_sed_melt = merge_size_ranges(mp, 'MP', sed, 'SED')

    return mp, sed, mp_sed_melt


def rebin(df):
    """
    Sum columns into larger bins
    Returns a df with fewer columns, i.e. coarser bins
    """

    df2 = df.groupby([[i // Config.rebin_by_n for i in range(0, len(df.T))]], axis=1).sum()
    df2.columns = df.columns[::Config.rebin_by_n]

    return df2


def complete_index_labels(df):
    """
    Turns index labels from single number (lower boundary of size bin) to range ('lower'_'upper').
    Drops the last row (which has no upper).
    """

    df.columns = df.iloc[0, :].add_suffix('_').index.values + np.append(df.columns[1:].values, 'x')
    df = df.iloc[:, 0:-1]

    return df


def sediment_preps(sed_df):
    """
    Takes the imported sediment grain size data from Master Sizer csv export and prepares a
    dataframe suitable for the following data analyses
    """

    sed_df = sed_df.groupby('Sample').mean(numeric_only=True)  # average all repeated Master Sizer measurements on individual samples
    # sed_df.rename(columns={'0.01': '0'}, inplace=True)  # renaming lowest size bin to 0
    size_cols = pd.to_numeric(sed_df.columns, errors='coerce')  # only columns that hold size bin data
    if Config.size_filter_on_sed_grainsizes:
        sed_df = sed_df.loc[:,  # keep only size bin columns which are between set size limits
            (size_cols > Config.lower_size_limit) &
            (size_cols <= Config.upper_size_limit) 
        ]
    else:
        sed_df = sed_df.loc[:, size_cols > 0]  # keep only size bin columns
    sed_df.columns = sed_df.columns.astype(float)

    if Config.rebinning:
        sed_df = rebin(sed_df)

    sed_df = complete_index_labels(sed_df)

    lowers = sed_df.columns.str.split('_').str[0].astype(float).values  # extract size bin lower boundaries from column names
    uppers = sed_df.columns.str.split('_').str[1].astype(float).values  # extract size bin upper boundaries from column names
    centers = (lowers + uppers) / 2  # calculate size bin centers from lower and upper boundaries

    if Config.closing:
        sed_df = close_compositional_data(sed_df)  # close compositional data

    return sed_df, {'lower': lowers, 'center': centers, 'upper': uppers}


def close_compositional_data(df, full=Config.closing):
    """
    Closes compositional data according to `full` value. In the closed df the sum of all rows will equal to `full`.
    """
    df[:] = closure(df.to_numpy()) * full
    return df


def combination_sums(df):  # TODO: convert to samples-in-rows-format
    """
    Append new rows to a df, where each new row is a column-wise sum of an original row
    and any possible combination of consecutively following rows. The input df must have
    an index according to the scheme below.
    
    
    Example:
    
                INPUT  DF                             OUTPUT DF
                
             A        B        C                        A        B        C
             
      0_10   a        b        c               0_10     a        b        c   
     10_20   d        e        f     -->      10_20     d        e        f
     20_30   g        h        i              20_30     g        h        i
     30_40   j        k        l              30_40     j        k        l 
                                               0_20    a+d      b+e      c+f
                                               0_30   a+d+g    b+e+h    c+f+i
                                               0_40  a+d+g+j  b+e+h+k  c+f+i+l
                                              10_30    d+g      e+h      f+i
                                              10_40   d+g+j    e+h+k    f+i+l
                                              20_40    g+j      h+k      i+l
    """

    ol = len(df)  # original length

    for i in range(ol):
        for j in range(i + 1, ol):
            new_row_name = df.index[i].split('_')[0] + '_' + df.index[j].split('_')[
                1]  # creates a string for the row index from the first and the last rows in the sum
            df.loc[new_row_name] = df.iloc[i:j].sum()

    return df
