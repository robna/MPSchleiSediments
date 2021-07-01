import numpy as np
import pandas as pd
from KDE_settings import Regio_Sep


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
    sdd_MP_sed = sdd_MP_sed.merge(pd.DataFrame.from_dict(Regio_Sep, orient='index', columns=['Regio_Sep']),
                                  left_on='Sample', right_index=True)

    # export the final data
    sdd_MP_sed.to_csv('../csv/MP_Stats_SchleiSediments.csv')
    return sdd_MP_sed


def sdd2pdd(sdd_MP, pdd_MP):
    """Some of the SDD data are merged onto the PDD df,
    meaning their values get repeated for each particle of a sample"""
    #sdd_MP = pd.read_csv('../csv/MP_Stats_SchleiSediments.csv', index_col=0)

    pdd_sdd_MP = pdd_MP.merge(sdd_MP[['Sample', 'TOC', 'Regio_Sep']], on='Sample')
    pdd_sdd_MP.rename(columns={'TOC': 'TOCs', 'Sampling_weight_[kg]': 'Sampling_weight'}, inplace=True)

    # the PDD df get very large, so we drop certain data columns that are not needed for the plots
    pdd_sdd_MP.drop(['Site_name', 'GPS_LON', 'GPS_LAT', 'Compartment',
                 'Contributor', 'Project', 'Size_1_[µm]', 'Size_2_[µm]', 'Shape', 'Colour',
                 'polymer_type', 'library_entry', 'lab_blank_ID', 'sample_ID'], axis=1, inplace=True)
    return pdd_sdd_MP