import pandas as pd

import prepare_data
import KDE_utils
from components import PCOA
from settings import Config, sediment_data_filepaths

import streamlit as st


@st.cache()
def data_load_and_prep():
    mp_pdd = prepare_data.get_pdd()
    #mp_pdd = use_shortnames(mp_pdd)
    return mp_pdd


@st.cache()
def load_grainsize_data():
    # Import sediment data (sediment frequencies per size bin from master sizer export)
    grainsize_iow, grainsize_cau, boundaries_dict = prepare_data.get_grainsizes(sediment_data_filepaths[f'IOW_{Config.sediment_grainsize_basis}'])
    #grainsize_iow = use_shortnames(grainsize_iow.reset_index().rename(columns={'index': 'Sample'})).set_index('Sample')
    sed_scor, sed_load, sed_expl = PCOA(grainsize_iow, 2)
    return sed_scor, grainsize_iow, boundaries_dict


@st.cache()
def pdd2sdd(mp_pdd):
    # ...some data wrangling to prepare particle domain data and sample domain data for MP and combine with certain sediment aggregates.
    mp_sdd = prepare_data.aggregate_SDD(mp_pdd)

    sdd_iow = prepare_data.additional_sdd_merging(mp_sdd)
    #sdd_iow = use_shortnames(sdd_iow)
    
#     sdd_iow['pred_Ord_Poly_ConcentrationA500'] = np.exp(
#          0.505 + 0.0452 * sdd_iow['perc_MUD'] + 0.0249 * 2.22 * sdd_iow['TOC'])  # TODO: temporarily added to compare to the prediction from Kristinas Warnow paper
#         #-0.2425 + 0.0683 * sdd_iow['perc_MUD'] - 0.0001 * sdd_iow['Dist_WWTP']+ 0.0205 * 2.22 * sdd_iow['TOC']
       
#     sdd_iow['pred_Paint_ConcentrationA500'] = np.exp(
#         2.352 + 0.032 * sdd_iow['perc_MUD'] - 0.003 * sdd_iow['Dist_Marina'])  
        
#  #TODO: temporarily added to compare to the prediction from Kristinas Warnow paper

#     sdd_iow['pred_TMP_ConcentrationA500'] = np.exp(
#         -0.4207 + 0.0826 * sdd_iow['perc_MUD'] + 0.056 * 8.33 * sdd_iow['TIC'] - 0.0002 * sdd_iow['Dist_WWTP'])  
#         #2.4491 + 0.0379 * sdd_iow['perc_MUD'])
    return sdd_iow


@st.cache()
def get_size_kde(mp_pdd, boundaries_dict, grainsize_iow):
    boundaries = boundaries_dict['center']
    mp_size_pdfs = KDE_utils.per_sample_kde(mp_pdd, boundaries, size_dim = Config.size_dim, weight_col=Config.kde_weights, bw=Config.fixed_bw, optimise=Config.optimise_bw)  # calculate mp size probability density functions
    mp_size_pmf = KDE_utils.probDens2prob(mp_size_pdfs)  # calculate mp size probability mass functions, i.e. probability of finding a MP particle in a specific size bin
    mp_size_pmf.columns = grainsize_iow.columns[:-1]  # when using "centers" of size bin boundaries for KDE, we need to adjust the column names to match with the sediment df again!
    _, _, mp_sed_melt = prepare_data.equalise_mp_and_sed(mp_size_pmf, grainsize_iow)  # if needed: returns truncated copies of mp_size_pmf and grainsize_iow as element [0] and [1].
    mp_size_cpmf = mp_size_pmf.cumsum(axis=1) # cumulative sum of the probability mass functions
    size_bin_number_containing_median = (mp_size_cpmf.T.reset_index(drop=True).T - 0.5).abs().idxmin(axis=1)  # Find the size bins which enclose the 50% of the probability mass. OBS: if choosing a different colsing value than 1, the subtractionnn of 0.5 needs to be adjusted!!
    KDE_medians = pd.DataFrame(boundaries_dict).center[size_bin_number_containing_median]
    KDE_medians.name = 'MP_size_median_from_KDE'
    KDE_medians.index = size_bin_number_containing_median.index
    return KDE_medians, mp_sed_melt

