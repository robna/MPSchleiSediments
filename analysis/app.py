import pandas as pd
from math import floor, ceil
from statsmodels.graphics.gofplots import qqplot

import prepare_data
import glm
import cv
from components import pca, PCOA
from plots import scatter_chart, poly_comp_chart, histograms, biplot, station_map
from settings import Config, shortnames, regio_sep

import streamlit as st
st.set_page_config(layout="wide")

featurelist = [

    'Concentration', 'MassConcentration', 'MP_D50',  # endogs
    'ConcentrationA500', 'ConcentrationB500', 'ConcentrationA500_div_B500', # endog derivatives
    # 'pred_Ord_Poly_ConcentrationA500', 'pred_TMP_ConcentrationA500','pred_Paint_ConcentrationA500',  # more endog derivatives
    # 'Concentration_paint', 'Concentration_PS_Beads', 'Concentration_ord_poly', 'Concentration_irregular',  # even more endog derivatives

    'LON', 'LAT', 'X', 'Y', 'Depth', 'Dist_Land', 'Dist_Marina', 'Dist_WWTP', 'Dist_WWTP2', 'regio_sep',  # geo related exogs
    'WWTP_influence_as_tracer_mean_dist', 'WWTP_influence_as_cumulated_residence', 'WWTP_influence_as_mean_time_travelled',
    
    # 'Split', 'Mass', 'Frequency', 'FrequencyA500', 'FrequencyB500', 'MPmass',  # sampling related exogs
    
    'PC1', 'PC2',   # sediment size PCOA outputs
    # 'SAMPLE TYPE ', 'TEXTURAL GROUP ', 'SEDIMENT NAME ',  # sediments (gradistat) exogs
    # 'MoM_ari_MEAN', 'MoM_ari_SORTING', 'MoM_ari_SKEWNESS', 'MoM_ari_KURTOSIS',
    # 'MoM_geo_MEAN', 'MoM_geo_SORTING', 'MoM_geo_SKEWNESS', 'MoM_geo_KURTOSIS',
    # 'MoM_log_MEAN', 'MoM_log_SORTING', 'MoM_log_SKEWNESS', 'MoM_log_KURTOSIS',
    # 'FW_geo_MEAN',  'FW_geo_SORTING',  'FW_geo_SKEWNESS',  'FW_geo_KURTOSIS',
    # 'FW_log_MEAN',  'FW_log_SORTING',  'FW_log_SKEWNESS',  'FW_log_KURTOSIS',
    'MODE 1 (µm)', #'MODE 2 (µm)', 'MODE 3 (µm)',
    'D10 (µm)', 'D50 (µm)', 'D90 (µm)',
    # '(D90 div D10) (µm)', '(D90 - D10) (µm)', '(D75 div D25) (µm)', '(D75 - D25) (µm)',
    'perc GRAVEL', 'perc SAND', 'perc MUD', 'perc CLAY', 
    # 'perc V COARSE SAND', 'perc COARSE SAND', 'perc MEDIUM SAND', 'perc FINE SAND', 'perc V FINE SAND',
    # 'perc V COARSE SILT', 'perc COARSE SILT', 'perc MEDIUM SILT', 'perc FINE SILT', 'perc V FINE SILT',
    
    'OM_D50', 'TOC', 'Hg', 'TIC',  # other exogs

    'Sample'  # sample name

    ]


@st.cache()
def data_load_and_prep():
    mp_pdd = prepare_data.get_pdd()

    # Also import sediment data (sediment frequencies per size bin from master sizer export)
    grainsize_iow = prepare_data.get_grainsizes()[0]  # get_grainsizes returns 3 objects: iow, cau and centers of sediment size bins

    sed_scor, sed_load, sed_expl = PCOA(grainsize_iow, 2)

    return mp_pdd, sed_scor


@st.cache()
def pdd2sdd(mp_pdd, regions):
    # ...some data wrangling to prepare particle domain data and sample domain data for MP and combine with certain sediment aggregates.
    mp_sdd = prepare_data.aggregate_SDD(mp_pdd)

    sdd_iow = prepare_data.additional_sdd_merging(mp_sdd)
    sdd_iow = sdd_iow.replace({'Sample': shortnames}).sort_values(by='Sample')
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


def filters(mp_pdd):
    
    Config.size_dim = st.sidebar.radio('Select size dimension', ['size_geom_mean', 'Size_1_µm', 'Size_2_µm', 'Size_3_µm',
                                                                 'vESD', 'particle_volume_µm3', 'particle_mass_µg'])
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

    regionfilter = st.sidebar.multiselect('Select regions:', set(regio_sep.values()),
                                          default=set(regio_sep.values()))
    return mp_pdd, regionfilter


def df_expander(df, title):
    with st.expander(title):
        st.write(df)
        st.write('Shape: ', df.shape)
        col1, col2 = st.columns([1,3])
        col1.write(df.dtypes)
        col2.write(df.describe())


def new_chap(title = None):
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance
    if title:
        st.subheader(title)

    
def get_selections(optionlist, defaults, key=0):
    """
    Provide the necessary selection for scatter_chart from plots.py
    :param optionlist: list of options to select from
    :param defaults: tuple of column names: (x, y, color) used as their respective default selections
    :param key: a unique value for each call of this function to allow multiple instances of the selection widgets
    """
    col1, col2, col3 = st.columns(3)
    class sel_dict(object): pass  # create dummy class to get access to __dict__
    sel = sel_dict()
    sel.y = col1.selectbox('y-Values:', optionlist, index=optionlist.index(defaults[1]), key='y'+str(key))
    sel.x = col1.selectbox('x-Values:', optionlist, index=optionlist.index(defaults[0]), key='x'+str(key))
    sel.color = col1.selectbox('Color:', [None, *optionlist], index=optionlist.index(defaults[2])+1, key='color'+str(key))
    sel.xtransform = col2. checkbox('Log transform x-data', key='xtransform'+str(key))
    sel.ytransform = col2. checkbox('Log transform y-data', key='ytransform'+str(key))
    sel.reg = col2.radio('Regression type:', [None, 'linear', 'log', 'exp', 'pow'], index=0, key='reg'+str(key))
    sel.reg_groups = col2.checkbox('Calculate separate regressions by color?', key='reg_groups'+str(key))
    sel.xscale = col3.radio('X-Axis type:', ['linear', 'log', 'sqrt'], index=0, key='xscale'+str(key))
    sel.yscale = col3.radio('Y-Axis type:', ['linear', 'log', 'sqrt'], index=0, key='yscale'+str(key))
    sel.equal_axes = col3.checkbox('Equal axes?', key='equal_axes'+str(key))
    sel.identity = col3.checkbox('Show identity line (dashed)?', key='identity'+str(key))
    sel.mix_lines = col3.checkbox('Show conservative mixing lines?', key='mix_lines'+str(key))
    sel.labels = col3.selectbox('Labels:', [None, *optionlist], index=0, key='labels'+str(key))
    cols = (col1, col2, col3)
    return sel.__dict__, cols


def main():
#%%
    st.title('Microplastics and sediment analysis')
    new_chap()
    
    mp_pdd, sed_scor = data_load_and_prep()  # load data

    raw_data_checkbox = st.sidebar.checkbox('Show raw data')
    if raw_data_checkbox:
        df_expander(mp_pdd, "Original particle domain data")            

    mp_pdd, regionfilter = filters(mp_pdd)  # provide side bar menus and filter data
    sdd_iow = pdd2sdd(mp_pdd, regionfilter)
    df = sdd_iow.merge(sed_scor, right_index=True, left_on='Sample', how='left')

    if raw_data_checkbox:
        df_expander(mp_pdd, "Filtered particle domain data")
        with st.expander("Plot particle properties"):
            particle_chart_selections, cols = get_selections(mp_pdd.columns.tolist(), ('size_geom_mean', 'Size_3_µm', 'Shape'), key='_raw')
            particle_scatters, particle_reg_params = scatter_chart(
            mp_pdd, **particle_chart_selections,
            title='', width=800, height=600)
            cols[0].write(particle_scatters)
            cols[2].markdown('___', unsafe_allow_html=True)
            cols[2].text("")  # empty line to make some distance
            cols[2].write('Regression parameters:')
            cols[2].write(particle_reg_params)
        df_expander(df, "Sample domain data")


#%%
    # new_cap('Map')
    # if st.checkbox('Plot map'):
    #     station_map(mp_pdd)  # plot map
    # st.markdown('___', unsafe_allow_html=True)
    # st.text("")  # empty line to make some distance

#%%
    new_chap('MP properties')
    if st.checkbox('Size histogram'):
        st.write(histograms(mp_pdd))

    if st.checkbox('Polymer composition'):
        st.write(poly_comp_chart(mp_pdd, df))

        com = prepare_data.aggregate_SDD(
                    mp_pdd.groupby(['Sample', 'polymer_type'])
            ).merge(sdd_iow[['Sample', 'Dist_WWTP', 'regio_sep']], on='Sample'
            ).pivot(index=['Sample'], columns=['polymer_type'], values=['Concentration']
            ).droplevel(0,axis=1
            ).fillna(0)
        
        st.write(biplot(
            scor=PCOA(com)[0],
            load=PCOA(com)[1],
            expl=PCOA(com)[2],
            discr=sdd_iow,
            x='PC1',
            y='PC2',
            sc='regio_sep',
            ntf=7,
            normalise='standard'
            ))


#%%
    new_chap('Feature analysis')  # TODO: we have 2 x PC1 / PC2 (from sediment PCOA and from here), this is confusing...
    if st.checkbox('Feature analysis'):
        feats = ['Depth', 'Dist_Land', 'Dist_WWTP', 'PC1', 'PC2', 'MODE 1 (µm)', 'D50 (µm)', 'perc MUD', 'TOC']
        featfilter = st.multiselect('Select features:', featurelist, default=feats)

        st.write(df.set_index('Sample')[featfilter].dropna())

        scor, load, expl = pca(df.set_index('Sample')[featfilter].dropna())
        st.write(biplot(scor, load, expl, sdd_iow.dropna(axis=1).dropna(), 'PC1', 'PC2', sc='Concentration', ntf=7, normalise='standard'))

        df2 = pd.concat([scor, df.set_index('Sample')[['Dist_WWTP', 'regio_sep', 'LON', 'LAT']]], axis=1)
        st.write(scatter_chart(df2.reset_index(), 'Dist_WWTP', 'PC1', 'regio_sep', 'Sample', width=800, height=600)[0])

#%%
    new_chap('Single predictor correlation and colinearity check')
    sample_chart_selections, cols = get_selections(featurelist, ('perc MUD', 'Concentration', 'regio_sep'), key='_sdd')

    scatters, reg_params = scatter_chart(df, **sample_chart_selections, title='', width=800, height=600)

    cols[0].write(scatters)
    cols[2].markdown('___', unsafe_allow_html=True)
    cols[2].text("")  # empty line to make some distance
    cols[2].write('Regression parameters:')
    cols[2].write(reg_params)

    # TODO: temporary check for r-values (remove later)
    # from scipy.stats import pearsonr
    # r, p = pearsonr(df[predx], df[predy])
    # st.write(f'Pearson r: {r}, p: {p}')
    # from sklearn.metrics import r2_score
    # st.write(f'R2: {r2_score(df[predx], df[predy])}')

#%%
    new_chap('GLM')
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
        col2.write(scatter_chart(df, Config.glm_formula.split(' ~')[0], 'yhat',
                                 color='regio_sep',
                                 identity=True, equal_axes=False,
                                 width=400, height=300,
                                 title='GLM --- yhat vs. y')[0])
        col3.write(scatter_chart(df, 'yhat', 'pearson_resid',
                                 color='regio_sep', title='GLM --- Pearson residuals')[0])

        resid = glm_res.resid_deviance.copy()
        col3.pyplot(qqplot(resid, line='r'))

        if st.checkbox('LOOCV'):
            _, metrics = cv.loocv(df)
            st.write(metrics)

#%%
    new_chap()

if __name__ == "__main__":
    main()
