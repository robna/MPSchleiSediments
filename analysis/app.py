import pandas as pd
import numpy as np
from math import floor, ceil
from statsmodels.graphics.gofplots import qqplot
from sklearn.metrics import r2_score

import prepare_data
import KDE_utils
import glm
import cv
from components import pca, PCOA
from plots import scatter_chart, poly_comp_chart, poly_comp_pie, histograms, biplot, station_map, size_kde_combined_samples_dist_plot
from settings import Config, shortnames, regio_sep, featurelist, sediment_data_filepaths

import streamlit as st
st.set_page_config(layout="wide")

endogs = ['Concentration', 'MassConcentration', 'MP_D50', 'MP_size_median_from_KDE',]  # endogs
endog_derivatives = [
    'ConcentrationA500', 'ConcentrationB500', 'ConcentrationA500_div_B500',  # endog derivatives
    # 'pred_Ord_Poly_ConcentrationA500', 'pred_TMP_ConcentrationA500','pred_Paint_ConcentrationA500',  # more endog derivatives
    # 'Concentration_paint', 'Concentration_PS_Beads', 'Concentration_ord_poly', 'Concentration_irregular',  # even more endog derivatives
]
additional_exogs = [
    'LON', 'LAT', 'X', 'Y', 'Depth', 'Dist_Land', 'Dist_Marina', 'Dist_WWTP', 'Dist_WWTP2', 'regio_sep', 'OM_D50', 'Split', 'TOC', 'SED_MODE1'
]

featurelist = endogs + endog_derivatives + [f for f in additional_exogs if f not in featurelist] + featurelist + ['Sample']

def use_shortnames(df):
    return df.replace({'Sample': shortnames}).sort_values(by='Sample')


@st.cache()
def data_load_and_prep():
    mp_pdd = prepare_data.get_pdd()
    #mp_pdd = use_shortnames(mp_pdd)
    return mp_pdd


def load_grainsize_data():
    # Import sediment data (sediment frequencies per size bin from master sizer export)
    grainsize_iow, grainsize_cau, boundaries_dict = prepare_data.get_grainsizes(sediment_data_filepaths[f'IOW_{Config.sediment_grainsize_basis}'])
    #grainsize_iow = use_shortnames(grainsize_iow.reset_index().rename(columns={'index': 'Sample'})).set_index('Sample')
    sed_scor, sed_load, sed_expl = PCOA(grainsize_iow, 2)
    return sed_scor, grainsize_iow, boundaries_dict


@st.cache()
def pdd2sdd(mp_pdd, regions):
    # ...some data wrangling to prepare particle domain data and sample domain data for MP and combine with certain sediment aggregates.
    mp_sdd = prepare_data.aggregate_SDD(mp_pdd)

    sdd_iow = prepare_data.additional_sdd_merging(mp_sdd)
    #sdd_iow = use_shortnames(sdd_iow)
    sdd_iow = sdd_iow.loc[
        sdd_iow.regio_sep.isin(regions)]  # filter based on selected regions

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


#@st.cache()
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


def filters(mp_pdd):
    
    Config.sediment_grainsize_basis = st.sidebar.radio('Select basis for sediment grain size distributions', ['Volume', 'Counts'], index=0)
    Config.kde_weights = st.sidebar.radio('Select basis for MP size distributions (selecting "None" means distributions are particle count-based, "particle_volume_share" means volume distributions, "particle_mass_share" means mass distributions)', [None, 'particle_volume_share', 'particle_mass_share'])
    Config.fixed_bw = st.sidebar.number_input('Fixed bandwidth for MP size distribution KDEs (no optimisation)', value=75.0, min_value=0.0, max_value=200.0, step=10.0)
    Config.optimise_bw = st.sidebar.checkbox('Optimise KDE bandwidth for each sample? (very slow, check console output...)')
    Config.size_dim = st.sidebar.radio('Select size dimension', ['size_geom_mean', 'Size_1_µm', 'Size_2_µm', 'Size_3_µm',
                                                                 'vESD', 'particle_volume_µm3', 'particle_mass_µg'], index=4)
    size_lims = floor(mp_pdd[Config.size_dim].min() / 10) * 10, ceil(mp_pdd[Config.size_dim].max() / 10) * 10
    Config.lower_size_limit = st.sidebar.number_input('Lower size limit (with respect to selected size dimension)',
                                                      value=size_lims[0],
                                                      min_value=size_lims[0],
                                                      max_value=size_lims[1],
                                                      step=100)
    Config.upper_size_limit = st.sidebar.number_input('Upper size limit (with respect to selected size dimension)',
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


def df_expander(df, title, height=300, row_sums=False):
    with st.expander(title):
        if row_sums:
            st.write('**Sum of row is shown in front of first column**')
        st.dataframe(df.set_index(df.sum(axis=1), append=True).rename_axis([df.index.name, 'Sum of row']) if row_sums else df, height=height)
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
    col1, col2, col3 = st.columns((2,1,1))
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
    sel.linref = col3.checkbox('Show linear reference line (dotted)?', key='linref'+str(key))
    if sel.linref is True:
        sel.linref_slope = col3.number_input('Line slope:', value=1.0, min_value=0.0, max_value=2.0, step=0.1, key='linref_slope'+str(key))
        sel.linref_intercept = col3.number_input('Line offset:', value=0.0, min_value=-100.0, max_value=100.0, step=10.0, key='linref_intercept'+str(key))
    sel.mix_lines = col3.checkbox('Show conservative mixing lines?', key='mix_lines'+str(key))
    sel.labels = col3.selectbox('Labels:', [None, *optionlist], index=0, key='labels'+str(key))
    cols = (col1, col2, col3)
    return sel.__dict__, cols


def main():
#%%
    st.title('Microplastics and sediment analysis')
    new_chap()
    
    mp_pdd = data_load_and_prep()  # load data

    raw_data_checkbox = st.sidebar.checkbox('Show raw data')
    if raw_data_checkbox:
        df_expander(mp_pdd, "Original MP particle domain data")            

    mp_pdd, regionfilter = filters(mp_pdd)  # provide side bar menus and filter data
    sdd_iow = pdd2sdd(mp_pdd, regionfilter)

    sed_scor, grainsize_iow, boundaries_dict = load_grainsize_data()
    KDE_medians, mp_sed_melt = get_size_kde(mp_pdd, boundaries_dict, grainsize_iow)
    df = sdd_iow.merge(sed_scor, right_index=True, left_on='Sample', how='left').join(KDE_medians, on='Sample')
    
    if raw_data_checkbox:
        df_expander(mp_pdd, "Filtered MP particle domain data")
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
        df_expander(df, "MP Sample domain data", height=1000)

    if raw_data_checkbox:
        df_expander(grainsize_iow, f"Sediment grainsize data (IOW), loaded from: {sediment_data_filepaths[f'IOW_{Config.sediment_grainsize_basis}']}", row_sums=True)
        
#%%
    # new_cap('Map')
    # if st.checkbox('Plot map'):
    #     station_map(mp_pdd)  # plot map
    # st.markdown('___', unsafe_allow_html=True)
    # st.text("")  # empty line to make some distance

#%%
    new_chap('MP properties')
    if st.checkbox('Particle sizes'):
        st.write(histograms(mp_pdd, title=f'Distribution of measured MP particle {Config.size_dim}'))
        st.write(size_kde_combined_samples_dist_plot(mp_sed_melt, title='KDE modelled size distribution of MP and sediment as average of all samples'))

    if st.checkbox('Polymer composition'):
        composition_of_ = st.radio('Select property:', ['polymer_type', 'Shape'], index=0)
        comp0 = prepare_data.aggregate_SDD(
                    mp_pdd.groupby(['Sample', composition_of_])
            ).merge(sdd_iow[['Sample', 'Dist_WWTP', 'regio_sep']], on='Sample'
            )
        comp1 = comp0.pivot(
                index=['Sample'],
                columns=[composition_of_],
                values=['Concentration']
            ).droplevel(0,axis=1
            ).fillna(0
            )
        comp2 = comp1.sum().rename('Concentration').to_frame().reset_index()
        
        col1, col2 = st.columns([2,1])
        col1.write(poly_comp_chart(comp0, composition_of_))
        #col2.write(poly_comp_pie(comp2))
        st.write(biplot(
            scor=PCOA(comp1)[0],
            load=PCOA(comp1)[1],
            expl=PCOA(comp1)[2],
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
        feats = ['Depth', 'Dist_Land', 'Dist_WWTP', 'PC1', 'PC2', 'SED_MODE1', 'SED_D50', 'perc_MUD', 'TOC']
        featfilter = st.multiselect('Select features:', featurelist, default=feats)

        st.write(df.set_index('Sample')[featfilter].dropna())

        scor, load, expl = pca(df.set_index('Sample')[featfilter].dropna())
        st.write(biplot(scor, load, expl, sdd_iow.dropna(axis=1).dropna(), 'PC1', 'PC2', sc='Concentration', ntf=7, normalise='standard'))

        df2 = pd.concat([scor, df.set_index('Sample')[['Dist_WWTP', 'regio_sep', 'LON', 'LAT']]], axis=1)
        st.write(scatter_chart(df2.reset_index(), 'Dist_WWTP', 'PC1', 'regio_sep', 'Sample', width=800, height=600)[0])

#%%
    new_chap('Single predictor correlation and colinearity check')
    sample_chart_selections, cols = get_selections(featurelist, ('perc_MUD', 'Concentration', 'regio_sep'), key='_sdd')

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
        families = ['Gaussian', 'Poisson', 'Gamma', 'Tweedie', 'InverseGaussian', 'NegativeBinomial']
        family = col1.radio('Select distribution family:', families, index=families.index('NegativeBinomial'))
        # for neg.binom use CT-alpha-estimator from here: https://web.archive.org/web/20210303054417/https://dius.com.au/2017/08/03/using-statsmodels-glms-to-model-beverage-consumption/
        Config.glm_family = family
        tweedie_power = col1.number_input('Tweedie power (only applied if family "Tweedie" is chosen):', value=2.0, min_value=0.0, max_value=3.0, step=0.1)
        Config.glm_tweedie_power = tweedie_power

        links = [None, 'Power', 'identity', 'inverse_power', 'inverse_squared', 'sqrt', 'log']
        link = col2.selectbox('Select link function (use None for default link of family):', links, index=1)
        Config.glm_link = link
        power_power = col2.number_input('Power exponent (only applied if link "Power" is chosen):', value=-1.55, min_value=-10.0, max_value=10.0, step=0.1)
        Config.glm_power_power = power_power

        Config.glm_formula = st.text_input('GLM formula:', 'Concentration ~ Dist_WWTP + PC1')
        target_name = Config.glm_formula.split('~')[0].strip()

        # resp = st.sidebar.selectbox('Select Response', reponselist)
        glm_res = glm.glm(df)

        col1, col2, col3 = st.columns(3)
        col1.write(glm_res.summary())

        st.text("")  # empty line to make some distance

        df['yhat'] = glm_res.mu
        df['pearson_resid'] = glm_res.resid_pearson
        col2.write(scatter_chart(df, target_name, 'yhat',
                                 #color='regio_sep',
                                 identity=True, equal_axes=False,
                                 width=400, height=300,
                                 title='GLM --- yhat vs. y')[0])
        if col3.checkbox('log yhat?'):
            df.yhat = np.log(df.yhat)
        col3.write(scatter_chart(df, 'yhat', 'pearson_resid',
                                 #color='regio_sep', 
                                 title='GLM --- Pearson residuals')[0])

        resid = glm_res.resid_deviance.copy()
        col3.pyplot(qqplot(resid, line='r'))

        if st.checkbox('LOOCV'):
            loocv_predictions, metrics = cv.loocv(df)
            st.write(metrics)
            with st.expander("LOOCV predictions"):
                col1, col2 = st.columns((1, 2), gap='large')
                col1.dataframe(loocv_predictions, height=1000)
                loglog = col2.checkbox('LogLog?')
                col2.write(scatter_chart(loocv_predictions, target_name, 'pred',
                                 labels='Sample',
                                 identity=True,
                                 equal_axes=True if not loglog else False,
                                 xscale='log' if loglog else 'linear', yscale='log' if loglog else 'linear',
                                #  xtransform=True, ytransform=True,
                                 width=800, height=800,
                                 title='LOOCV --- yhat vs. y')[0]
                )

#%%
    new_chap()

if __name__ == "__main__":
    main()
