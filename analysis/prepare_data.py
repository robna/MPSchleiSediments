import numpy as np
import pandas as pd
import geopandas as gpd
from skbio.stats.composition import closure
from settings import densities, regio_sep, shortnames, Config
import outliers
import geo


def get_pdd():
    """
    Reads the MP particle domain data (PDD) from the CSV file and does some preprocessing
    """
    
    mp_pdd = pd.read_csv('../data/env_MP_clean_list_SchleiSediments.csv', index_col=0)
    mp_pdd = mass_conversion(mp_pdd)  # calculate particle weights
    mp_pdd, gpn = outliers.low_freq_out(mp_pdd)  # remove low frequency outliers
    # mp_pdd = mp_pdd.loc[mp_pdd.Shape=='irregular']  # filter to only use fibres or irregulars
    mp_pdd['polymer_type'] = mp_pdd['polymer_type'].map(shortnames).fillna(mp_pdd['polymer_type'])  # use abbreviations for polymer names but retain original names for polymers not present in shortnames
    mp_pdd.columns = mp_pdd.columns.str.replace("[\[( )\]]", "")  # remove brackets from column names

    # mp_pdd = mp_pdd.loc[~mp_pdd.polymer_type.isin(['AcrR', 'AlkR', 'EPX'])]  # exclude paint flakes

    return mp_pdd


def get_grainsizes():
    """
    Reads the sediment grainsize data from the CSV file and does some preprocessing
    """

    grainsize_iow = pd.read_csv('../data/sediment_grainsize_IOW_vol_log-cau_not-closed.csv')
    grainsize_cau = pd.read_csv('../data/sediment_grainsize_CAU_vol_log-cau_closed.csv')
    grainsize_cau.dropna(subset=grainsize_cau.iloc[:,1:].columns, how='all', inplace=True)  # CAU sediment data contains empty sammples which are dropped here

    # Get the binning structure of the imported sediment data and optionally rebin it (make binning coarser) for faster computation
    grainsize_iow, boundaries = sediment_preps(grainsize_iow)
    grainsize_cau, _ = sediment_preps(grainsize_cau)

    return grainsize_iow, grainsize_cau, boundaries


def mass_conversion(df):
    """Adds MP density, volume and mass to each particle"""

    df['density'] = df['polymer_type'].map(densities)
    df['density'].fillna(
        densities['generic'],
        inplace=True)  # assume a general average density where exact density is not available, ref: https://doi.org/10.1021/acs.est.0c02982

    df.loc[df['Shape'] == 'irregular', 'Size_3_[µm]'] = (
        0.312 *
        df['size_geom_mean'] +
        3.706
      )  # calculates the 3rd dimension of non-fibres, according to Kristinas correlation between size_geom_mean and manually measured height (n=116, R²=0.49)
    
    df.loc[df['Shape'] == 'fibre', 'Size_3_[µm]'] = df['Size_2_[µm]'] # fibre height is just the same as fibre width

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
     )  # because of they way how Gepard detects MP we do not assume a fibre void fraction here

    df['particle_mass_[µg]'] = df['particle_volume_[µm3]'] * df['density'] * 1e-9

    # calculate volume and mass share of each particle grouped by Sample
    for sample_name, Sample in df.groupby(['Sample'], sort=False):
        Sample['particle_volume_share'] = Sample['particle_volume_[µm3]'] / Sample['particle_volume_[µm3]'].sum()
        df.loc[Sample.index, 'particle_volume_share'] = Sample['particle_volume_share']
        Sample['particle_mass_share'] = Sample['particle_mass_[µg]'] / Sample['particle_mass_[µg]'].sum()
        df.loc[Sample.index, 'particle_mass_share'] = Sample['particle_mass_share']

    return df


def aggregate_SDD(mp_pdd):
    """Calculates certain Sample domain data (SDD) aggregation from the particle domain data (PDD)"""

    if isinstance(mp_pdd, pd.DataFrame):  # if raw DF is submitted instead of groupby object, then group first
        mp_pdd = mp_pdd.groupby(['Sample'])

    mp_sdd = mp_pdd.agg(
        Frequency=('Site_name', 'count'),
        FrequencyA500=('Size_1_µm', lambda x: (x >= 500).sum()),
        FrequencyB500=('Size_1_µm', lambda x: (x < 500).sum()),
        MPmass=('particle_mass_µg', 'sum'),
        Mass=('Sampling_weight_kg', np.mean),
        # using "mean" here is actually weird as all entries are the same. Is there something like "first"?
        # LON=('GPS_LON', np.mean),  # TODO: switched to geodata from sampling log
        # LAT=('GPS_LAT', np.mean),  # TODO: switched to geodata from sampling log
        Split=('Fraction_analysed', np.mean),
        MP_D50=('size_geom_mean', np.median)
        ##MP_D50_A500 = ('size_geom_mean' >= 500.median()),
        # MP_D50_B500 = ('size_geom_mean', lambda x: (x<500).median())
    ).reset_index()

    mp_sdd['Concentration'] = round(mp_sdd['Frequency'] / mp_sdd['Mass'])
    mp_sdd['ConcentrationA500'] = round(mp_sdd['FrequencyA500'] / mp_sdd['Mass'])
    mp_sdd['ConcentrationB500'] = round(mp_sdd['FrequencyB500'] / mp_sdd['Mass'])
    mp_sdd['ConcentrationA500_div_B500'] = mp_sdd['ConcentrationA500'] / mp_sdd['ConcentrationB500']
    mp_sdd['MassConcentration'] = round(mp_sdd['MPmass'] / mp_sdd['Mass'])
    return mp_sdd


def additional_sdd_merging(mp_sdd, how='left'):
    """Takes SDD and amends it with corresponding sediment data"""

    # import d50 values  # TODO: commented out, as we use D50 and <63 from Gradistat; may be deleted (also in merge section)
    # sed_d50 = pd.read_csv('../csv/Schlei_Sed_D50_<63.csv', index_col=0)

    # import gradistat results
    sed_gradistat = pd.read_csv('../data/GRADISTAT_IOW_vol_log-cau_not-closed.csv', index_col=0)

    # import organic matter size, TOC, Hg data
    sed_om = pd.read_csv('../data/Schlei_OM.csv', index_col=0)

    # import sampling log data
    slogs = pd.read_csv('../data/Metadata_IOW_sampling_log.csv', index_col=0)

    # # import distance to waste water treatment plant  # TODO: can be removed (also in merge section): Dist_WWTP now included in slogs
    # dist_wwtp = pd.read_csv('../data/Schlei_Sed_Dist_WWTP.csv', index_col=0)

    # merge with mp per station
    mp_sdd_amended = pd.merge(mp_sdd, slogs.reset_index()[
        ['Sample', 'Depth', 'LON', 'LAT', 'Dist_Marina', 'Dist_WWTP', 'Dist_WWTP2']], on=['Sample'], how=how).merge(  # add metadata
        # sed_d50.reset_index(), on=['Sample'], how=how).merge(  # add sediment D50
        sed_gradistat.reset_index(), on=['Sample'], how=how).merge(  # add sediment gradistat
        sed_om.reset_index()[['Sample', 'OM_D50', 'TOC', 'Hg', 'TIC']], on=['Sample'], how=how).merge(  # add OM data
        # dist_wwtp.reset_index(), on=['Sample'], how=how).merge(  # add distance to WWTP
        pd.DataFrame.from_dict(regio_sep, orient='index', columns=['regio_sep']), left_on='Sample', right_index=True, how=how)  # add flags for regions

    # calculate distance from shore
    mp_sdd_amended['Dist_Land'] = geo.get_distance_to_shore(mp_sdd_amended['LON'], mp_sdd_amended['LAT'])
    
    # concatenate with Warnow data: only activate if you want to use Warnow data for comparison
    warnow = pd.read_csv('../data/Warnow_sdd.csv', index_col=0)
    warnow["ConcentrationA500"] = warnow["Concentration"]
    mp_sdd_amended = pd.concat([mp_sdd_amended, warnow], sort=False)
    
    # optionally: uncomment to export the final data
    # sdd_iow.to_csv('../csv/MP_Stats_SchleiSediments.csv')
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

    # make mp and sed contain only columns (size ranges) and rows (samples), which they have in common
    mp.drop([col for col in mp.columns if col not in sed.columns], axis=1, inplace=True)
    sed.drop([col for col in sed.columns if col not in mp.columns], axis=1, inplace=True)
    mp.drop([row for row in mp.index if row not in sed.index], axis=0, inplace=True)
    sed.drop([row for row in sed.index if row not in mp.index], axis=0, inplace=True)

    # Normalise to summed conc of all size bins,
    # i.e. turn from concentrations to percentage abundances.
    # Can be omitted by making bin_conc = True in settings Config.
    # In this case values will continue to represent concentrations.
    if not Config.bin_conc:
        mp = mp.apply(lambda x: x / x.sum() * 100, axis=1)

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

    sed_df = sed_df.groupby('Sample').mean()  # average all repeated Master Sizer measurements on individual samples
    # sed_df.rename(columns={'0.01': '0'}, inplace=True)  # renaming lowest size bin to 0
    sed_df = sed_df.loc[:,
             pd.to_numeric(sed_df.columns, errors='coerce') > 0]  # only keep columns that hold size bin data
    sed_df.columns = sed_df.columns.astype(float)

    # TODO: check if this is necessary...
    # sed_df = sed_df.loc[:, (sed_df.columns.astype('float') >= Config.lower_size_limit) &
    #                     (sed_df.columns.astype('float') <= Config.upper_size_limit)]  # truncate to relevant size range

    if Config.rebinning:
        sed_df = rebin(sed_df)

    # sed_lower_boundaries = sed_df.columns.values  # write the size bins lower boundaries in an array

    sed_df = complete_index_labels(sed_df)

    lowers = sed_df.columns.str.split('_').str[0].astype(float).values  # extract size bin lower boundaries from column names
    uppers = sed_df.columns.str.split('_').str[1].astype(float).values  # extract size bin upper boundaries from column names
    centers = (lowers + uppers) / 2  # calculate size bin centers from lower and upper boundaries

    if Config.closing:
        sed_df[:] = closure(sed_df.to_numpy()) * Config.closing  # close compositional data

    # TODO: check if this is still necessary...
    # non_zero_counts = sed_df.fillna(1).astype(bool).sum(axis=0)  # count number of non-zero-values in each column
    # sed_df = sed_df.loc[:, non_zero_counts[non_zero_counts >= int(
    #     (1 - Config.allowed_zeros) * sed_df.shape[0])].index.values]  # drop columns with too many zeros

    return sed_df, {'lower': lowers, 'center': centers, 'upper': uppers}


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
