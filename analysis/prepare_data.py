import numpy as np
import pandas as pd
from settings import Regio_Sep, Config


def aggregate_SDD(pdd_MP):
    """Calculates certain Sample domain data (SDD) aggregation from the particle domain data (PDD)"""

    sdd_MP = pdd_MP.groupby(['Sample']).agg(
        Frequency=('Site_name', 'count'),
        FrequencyA500=('size_geom_mean', lambda x: (x >= 500).sum()),
        FrequencyB500=('size_geom_mean', lambda x: (x < 500).sum()),
        Mass=('Sampling_weight_[kg]', np.mean),
        # using "mean" here is actually weird as all entries are the same. Is there something like "first"?
        GPS_LONs=('GPS_LON', np.mean),
        GPS_LATs=('GPS_LAT', np.mean),
        Split=('Fraction_analysed', np.mean),
        MP_D50=('size_geom_mean', np.median)
        ##MP_D50_A500 = ('size_geom_mean' >= 500.median()),
        # MP_D50_B500 = ('size_geom_mean', lambda x: (x<500).median())
    ).reset_index()

    sdd_MP['Concentration'] = round(sdd_MP['Frequency'] / (sdd_MP['Mass'] * sdd_MP['Split']))
    sdd_MP['ConcentrationA500'] = round(sdd_MP['FrequencyA500'] / (sdd_MP['Mass'] * sdd_MP['Split']))
    sdd_MP['ConcentrationB500'] = round(sdd_MP['FrequencyB500'] / (sdd_MP['Mass'] * sdd_MP['Split']))
    return sdd_MP


def add_sediment(sdd_MP):
    """Takes SDD and amends it with corresponding sediment data"""
    # import d50 values
    sed_d50 = pd.read_csv('../csv/Schlei_Sed_D50_new.csv', index_col=0)

    # import ogranic matter size, TOC, Hg data
    sed_OM = pd.read_csv('../csv/Schlei_OM.csv', index_col=0)

    # import sampling log data
    slogs = pd.read_csv('../csv/Schlei_sed_sampling_log.csv', index_col=0)

    # import distance to waste water treatment plant
    Dist_WWTP = pd.read_csv('../csv/Schlei_Sed_Dist_WWTP.csv', index_col=0)

    # merge with mp per station
    sdd_MP_sed = pd.merge(sdd_MP, slogs.reset_index(), on=['Sample'], how='left')
    sdd_MP_sed = pd.merge(sdd_MP_sed, sed_d50.reset_index(), on=['Sample'], how='left')
    sdd_MP_sed = pd.merge(sdd_MP_sed, sed_OM.reset_index(), on=['Sample'], how='left')
    sdd_MP_sed = pd.merge(sdd_MP_sed, Dist_WWTP.reset_index(), on=['Sample'], how='left')

    # flag entries as belonging to a certain region of the Schlei fjord
    sdd_MP_sed = sdd_MP_sed.merge(pd.DataFrame.from_dict(Regio_Sep, orient='index',
                                                         columns=['Regio_Sep']),
                                  left_on='Sample', right_index=True)

    # export the final data
    # sdd_MP_sed.to_csv('../csv/MP_Stats_SchleiSediments.csv')
    return sdd_MP_sed


def sdd2pdd(sdd_MP, pdd_MP):
    """
    Some of the SDD data are merged onto the PDD df,
    meaning their values get repeated for each particle of a sample
    """

    pdd_sdd_MP = pdd_MP.merge(sdd_MP[['Sample', 'TOC', 'Regio_Sep']], on='Sample')
    pdd_sdd_MP.rename(columns={'TOC': 'TOCs', 'Sampling_weight_[kg]': 'Sampling_weight'}, inplace=True)

    # the PDD df get very large, so we drop certain data columns that are not needed for the plots
    pdd_sdd_MP.drop(['Site_name', 'GPS_LON', 'GPS_LAT', 'Compartment',
                     'Contributor', 'Project', 'Size_1_[µm]', 'Size_2_[µm]', 'Shape', 'Colour',
                     'polymer_type', 'library_entry', 'lab_blank_ID', 'sample_ID'], axis=1, inplace=True)
    return pdd_sdd_MP


def melt_size_ranges(df, value_name):
    """
    Converts df with size bins or ranges in rows and samples in columns into a long-format df with 
    """

    melted_df = df.reset_index().melt(id_vars='sample', var_name='ranges', value_name=value_name)
    melted_df[['lower', 'upper']] = melted_df.ranges.str.split('_', expand=True).astype(int)
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
        df = df.loc[df.sample_x == df.sample_y].drop(['sample_y'],  # only keep entries where samples match
                                                     axis=1).rename(columns=
                                                                    {'sample_x': 'sample',
                                                                     'lower_x': 'lower_MP',
                                                                     'upper_x': 'upper_MP',
                                                                     'lower_y': 'lower_SED',
                                                                     'upper_y': 'upper_SED'})
    else:
        df = melted1.merge(melted2, on=['sample', 'lower', 'upper'])

    return df


def equalise_MP_and_Sed(mp, sed):
    """
    Various data harmonisation steps on DFs containing per size bin data of MP concentrations (mp)
    and sediment frequencies (sed). See inline comments for separate steps.
    """

    # make mp and sed contain only rows (size ranges) and columns (samples), which they have in common
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

    MPsedMelt = merge_size_ranges(mp, 'MP', sed, 'SED')

    return mp, sed, MPsedMelt


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
    df = df.iloc[:,0:-1]

    return df


def sediment_preps(sed_df):
    """
    Takes the imported sediment grain size data from Master Sizer csv export and prepares a
    dataframe suitable for the following data analyses
    """

    sed_df = sed_df.groupby('sample').mean()  # average all repeated Master Sizer measurements on individual samples
    sed_df.rename(columns={'0.01': '0'}, inplace=True)
    sed_df.columns = sed_df.columns.astype(int)

    sed_df = sed_df.loc[:, Config.lower_size_limit:Config.upper_size_limit]  # truncate to relevant size range

    if Config.rebinning:
        sed_df = rebin(sed_df)

    sed_lower_boundaries = sed_df.columns.values.astype(int)  # write the size bins lower boundaries in an array

    sed_df = complete_index_labels(sed_df)

    non_zero_counts = sed_df.fillna(1).astype(bool).sum(axis=0)  # count number of non-zero-values in each column
    sed_df = sed_df.loc[:, non_zero_counts[non_zero_counts >= int(
        (1 - Config.allowed_zeros) * sed_df.shape[0])].index.values]  # drop columns with too many zeros

    return sed_df, sed_lower_boundaries


def combination_sums(df):  # TODO: not tested yet
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
            step = j-i

            relevant_sizes = df_in.loc[(df_in.a >= i) & (df_in.a < j)]
            size_sum = relevant_sizes.sum()
            size_sum.drop('a', inplace=True)

            size_sum.rename(f'{i}_{j}', inplace=True)

            df_out = df_out.append(size_sum)

    df_out.rename_axis(columns='sample', inplace=True)

    return df_out

