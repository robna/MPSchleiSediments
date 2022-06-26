import numpy as np
import pandas as pd
from math import floor, ceil
import streamlit as st
from statsmodels.graphics.gofplots import qqplot

import prepare_data
import glm
import cv
from components import PCOA
from plots import scatter_chart, poly_comp_chart, histograms, station_map
from settings import Config, shortnames

st.set_page_config(layout="wide")

featurelist = ['Frequency', 'Concentration', 'MassConcentration', 'MP_D50',  # endogs
               'ConcentrationA500', 'pred_Ord_Poly_ConcentrationA500', 'pred_TMP_ConcentrationA500','pred_Paint_ConcentrationA500', 'ConcentrationB500', 'ConcentrationA500_div_B500', # endogs derivatives
               'Split', 'Mass', 'LON', 'LAT', 'Depth', 'Dist_Marina', 'Dist_WWTP', 'Dist_WWTP2', 'regio_sep',  # sampling related exogs
               'PC1', 'PC2',   # sediment size PCOA outputs
               # 'MoM_ari_MEAN', 'MoM_ari_SORTING', 'MoM_ari_SKEWNESS', 'MoM_ari_KURTOSIS',  # sediments (gradistat) exogs
               # 'MoM_geo_MEAN', 'MoM_geo_SORTING', 'MoM_geo_SKEWNESS', 'MoM_geo_KURTOSIS',
               # 'MoM_log_MEAN', 'MoM_log_SORTING', 'MoM_log_SKEWNESS', 'MoM_log_KURTOSIS',
               # 'FW_geo_MEAN', 'FW_geo_SORTING', 'FW_geo_SKEWNESS', 'FW_geo_KURTOSIS',
               # 'FW_log_MEAN', 'FW_log_SORTING', 'FW_log_SKEWNESS', 'FW_log_KURTOSIS',
               # 'MODE 1 (µm)', 'MODE 2 (µm)', 'MODE 3 (µm)',
               'D10 (µm)', 'D50 (µm)', 'D90 (µm)',
               # '(D90 div D10) (µm)', '(D90 - D10) (µm)', '(D75 div D25) (µm)', '(D75 - D25) (µm)',
               'perc GRAVEL', 'perc SAND', 'perc MUD', 'perc CLAY', 
               # 'perc V COARSE SAND', 'perc COARSE SAND', 'perc MEDIUM SAND', 'perc FINE SAND', 'perc V FINE SAND',
               # 'perc V COARSE SILT', 'perc COARSE SILT', 'perc MEDIUM SILT', 'perc FINE SILT', 'perc V FINE SILT',
               'OM_D50', 'TOC', 'Hg', 'TIC'  # other exogs
               ]


@st.cache()
def data_load_and_prep():
    mp_pdd = prepare_data.get_pdd()

    # Also import sediment data (sediment frequencies per size bin from master sizer export)
    grainsize_iow = prepare_data.get_grainsizes()[0]  # get_grainsizes returns 3 objects: iow, cau and sed_lower_boundaries

    scor, load, expl = PCOA(grainsize_iow, 2)

    return mp_pdd, scor


@st.cache()
def pdd2sdd(mp_pdd, regions):
    # ...some data wrangling to prepare particle domain data and sample domain data for MP and combine with certain sediment aggregates.
    mp_sdd = prepare_data.aggregate_SDD(mp_pdd)

    sdd_iow = prepare_data.additional_sdd_merging(mp_sdd)
    sdd_iow = sdd_iow.loc[
        sdd_iow.regio_sep.isin(regions)]  # filter based on selected regions

#     sdd_iow['pred_Ord_Poly_ConcentrationA500'] = np.exp(
#          0.505 + 0.0452 * sdd_iow['perc MUD'] + 0.0249 * 2.22 * sdd_iow['TOC'])  # TODO: temporarily added to compare to the prediction from Kristinas Warnow paper
#         #-0.2425 + 0.0683 * sdd_iow['perc MUD'] - 0.0001 * sdd_iow['Dist_WWTP']+ 0.0205 * 2.22 * sdd_iow['TOC']
       
    
#     sdd_iow['pred_Paint_ConcentrationA500'] = np.exp(
#         2.352 + 0.032 * sdd_iow['perc MUD'] - 0.003 * sdd_iow['Dist_Marina'])  
        
#  #TODO: temporarily added to compare to the prediction from Kristinas Warnow paper

#     sdd_iow['pred_TMP_ConcentrationA500'] = np.exp(
#         -0.4207 + 0.0826 * sdd_iow['perc MUD'] + 0.056 * 8.33 * sdd_iow['TIC'] - 0.0002 * sdd_iow['Dist_WWTP'])  
#         #2.4491 + 0.0379 * sdd_iow['perc MUD'])
    return sdd_iow


def main():
    mp_pdd, scor = data_load_and_prep()  # load data

    raw_data_checkbox = st.sidebar.checkbox('Show raw data')
    if raw_data_checkbox:
        with st.expander("Original particle domain data"):
            st.write(mp_pdd)
            st.write('Shape: ', mp_pdd.shape)

    sizedim = st.sidebar.radio('Select size dimension', ['size_geom_mean', 'Size_1_µm', 'Size_2_µm'])
    Config.size_dim = sizedim
    size_lims = floor(mp_pdd[Config.size_dim].min() / 10) * 10, ceil(mp_pdd[Config.size_dim].max() / 10) * 10
    Config.lower_size_limit = st.sidebar.number_input('Lower size limit',
                                                      value=size_lims[0],
                                                      min_value=size_lims[0],
                                                      max_value=size_lims[1],
                                                      step=100)
    Config.upper_size_limit = st.sidebar.number_input('Upper size limit',
                                                      value=size_lims[1],
                                                      min_value=size_lims[0],
                                                      max_value=size_lims[1],
                                                      step=100)

    density_lims = mp_pdd.density.min().astype(int).item(), mp_pdd.density.max().astype(int).item()
    Config.lower_density_limit = st.sidebar.number_input('Lower density limit',
                                                         value=density_lims[0],
                                                         min_value=density_lims[0],
                                                         max_value=density_lims[1],
                                                         step=100)
    Config.upper_density_limit = st.sidebar.number_input('Upper density limit',
                                                         value=density_lims[1],
                                                         min_value=density_lims[0],
                                                         max_value=density_lims[1],
                                                         step=100)

    samplefilter = st.sidebar.multiselect('Select samples:', mp_pdd.Sample.unique(), default=mp_pdd.Sample.unique())
    shapefilter = st.sidebar.multiselect('Select shapes:', ['irregular', 'fibre'], default=['irregular', 'fibre'])
    polymerfilter = st.sidebar.multiselect('Select polymers:', mp_pdd.polymer_type.unique(),
                                           default=mp_pdd.polymer_type.unique())
    mp_pdd = mp_pdd.loc[mp_pdd.Shape.isin(shapefilter)
                        & mp_pdd.polymer_type.isin(polymerfilter)
                        & mp_pdd.Sample.isin(samplefilter)
                        & (mp_pdd[Config.size_dim] >= Config.lower_size_limit)
                        & (mp_pdd[Config.size_dim] <= Config.upper_size_limit)
                        & (mp_pdd.density >= Config.lower_density_limit)
                        & (mp_pdd.density <= Config.upper_density_limit)
                        ]  # filter mp_pdd based on selected values

    regionfilter = st.sidebar.multiselect('Select regions:', ['WWTP', 'inner', 'middle', 'outer', 'river', 'warnow'],
                                          default=['WWTP', 'inner', 'middle', 'outer', 'river', 'warnow'])

    sdd_iow = pdd2sdd(mp_pdd, regionfilter)
    df = sdd_iow.merge(scor, right_index=True, left_on='Sample', how='left')

    if raw_data_checkbox:
        with st.expander("Filtered particle domain data"):
            st.write(mp_pdd)
            st.write('Shape: ', mp_pdd.shape)
            st.write(mp_pdd.dtypes)

        with st.expander("Sample domain data"):
            st.write(df)
            st.write('Shape: ', df.shape)
            st.write(df.dtypes)

    st.title('Microplastics and sediment analysis')
    st.markdown('___', unsafe_allow_html=True)

    # if st.checkbox('Plot map'):
    #     station_map(mp_pdd)  # plot map
    # st.markdown('___', unsafe_allow_html=True)
    # st.text("")  # empty line to make some distance

    st.subheader('MP size histograms')
    if st.checkbox('Size histogram'):
        st.write(histograms(mp_pdd))
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance

    st.subheader('Polymer composition')
    if st.checkbox('Polymer composition'):
        st.write(poly_comp_chart(mp_pdd, df))
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance


    st.subheader('GLM')
    if st.checkbox('Calculate GLM'):
        col1, col2 = st.columns(2)
        families = ['Gaussian', 'Poisson', 'Gamma', 'Tweedie', 'NegativeBinomial']
        family = col1.radio('Select distribution family:', families, index=families.index('Poisson'))
        # for neg.binom use CT-alpha-estimator from here: https://web.archive.org/web/20210303054417/https://dius.com.au/2017/08/03/using-statsmodels-glms-to-model-beverage-consumption/
        Config.glm_family = family

        links = [None, 'identity', 'Power', 'inverse_power', 'sqrt', 'log']
        link = col2.selectbox('Select link function (use None for default link of family):', links, index=0)
        Config.glm_link = link

        Config.glm_formula = st.text_input('GLM formula:', 'Concentration ~ Dist_WWTP + Q("D50 (µm)") + PC2')

        # resp = st.sidebar.selectbox('Select Response', reponselist)
        glm_res = glm.glm(df)

        col1, col2, col3 = st.columns(3)
        col1.write(glm_res.summary())

        st.text("")  # empty line to make some distance

        df['yhat'] = glm_res.mu
        df['pearson_resid'] = glm_res.resid_pearson
        col2.write(scatter_chart(df, 'yhat', Config.glm_formula.split(' ~')[0],
                                 c='regio_sep', equal_axes=True,
                                 title='GLM --- y vs. yhat'))
        col3.write(scatter_chart(df, 'yhat', 'pearson_resid',
                                 c='regio_sep', title='GLM --- Pearson residuals'))

        resid = glm_res.resid_deviance.copy()
        col3.pyplot(qqplot(resid, line='r'))

        if st.checkbox('LOOCV'):
            _, metrics = cv.loocv(df)
            st.write(metrics)

    st.text("")  # empty line to make some distance
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance
    st.subheader('Single predictor correlation and colinearity check')

    predx = st.selectbox('x-Values:', featurelist, index=featurelist.index('Depth'))
    predy = st.selectbox('y-Values:', featurelist, index=featurelist.index('Concentration'))
    c = st.selectbox('Color:', featurelist, index=featurelist.index('Dist_WWTP'))
    reg=st.radio('Regression type:', ['linear', 'log', 'exp', 'pow'], index=0)
    labels = st.checkbox('Show data labels')

    st.write(scatter_chart(df, predx, predy, c, 'Sample' if labels else None, reg, title='', width=800, height=600))

    # TODO: temporary check for r-values (remove later)
    from scipy.stats import pearsonr
    r, p = pearsonr(df[predx], df[predy])
    st.write(f'Pearson r: {r}, p: {p}')
    from sklearn.metrics import r2_score
    st.write(f'R2: {r2_score(df[predx], df[predy])}')
    
    st.write('Sum: ',df.ConcentrationA500.sum())
    st.write('Mean_A500: ',df.ConcentrationA500.mean())

    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance


if __name__ == "__main__":
    main()
