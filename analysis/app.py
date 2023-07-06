import pandas as pd
import numpy as np
from functools import reduce
from statsmodels.graphics.gofplots import qqplot
from sklearn.metrics import r2_score

import prepare_data
import glm
import cv
from components import pca, PCOA
from plots import scatter_chart, poly_comp_chart, poly_comp_pie, histograms, biplot, station_map, size_kde_combined_samples_dist_plot
from settings import Config, featurelist, sediment_data_filepaths

from app_loaders import (
    data_load_and_prep,
    load_grainsize_data,
    pdd2sdd,
    get_size_kde
)
from app_helpers import (
    general_settings,
    pdd_filters,
    sdd_filters,
    df_expander,
    new_chap,
    get_selections
)

import streamlit as st
st.set_page_config(layout="wide")

#%% Define variables to use
endogs = ['Concentration', 'MassConcentration','VolumeConcentration', 'MP_D50', 'MP_size_median_from_KDE', 'MP_size_mode1_from_KDE']  # endogs
endog_derivatives = [
    'ConcentrationA500', 'ConcentrationB500', 'ConcentrationA500_div_B500',  # endog derivatives
    # 'pred_Ord_Poly_ConcentrationA500', 'pred_TMP_ConcentrationA500','pred_Paint_ConcentrationA500',  # more endog derivatives
    # 'Concentration_paint', 'Concentration_PS_Beads', 'Concentration_ord_poly', 'Concentration_irregular',  # even more endog derivatives
]
additional_exogs = [
    'LON', 'LAT', 'X', 'Y', 'Depth', 'Dist_Land', 'Dist_Marina', 'Dist_WWTP', 'Dist_WWTP2', 'regio_sep', 'OM_D50', 'Split', 'TOC', 'SED_MODE1', 'SED_medians_from_grainsizes', 'SED_mode1s_from_grainsizes'
]
featurelist = endogs + endog_derivatives + [f for f in additional_exogs if f not in featurelist] + featurelist + ['Sample']

#%%
def main():
    st.title('Microplastics and sediment analysis')
    new_chap()
    
    raw_data_checkbox, vertical_merge = general_settings()  # loads sidebar widgets to control data calculations and app behaviour
    Config.vertical_merge = vertical_merge
    mp_pdd = data_load_and_prep()  # load data
    
    if raw_data_checkbox:
        df_expander(mp_pdd, "Original MP particle domain data")            

    mp_pdd = pdd_filters(mp_pdd)  # provide side bar menus and filter data
    sdd_iow = pdd2sdd(mp_pdd)
    sdd_iow = sdd_filters(sdd_iow)  # additional filters in siedbar can be used to limit which samples are included

    sed_scor, grainsize_iow, boundaries_dict = load_grainsize_data()
    KDE_medians, KDE_mode1, mp_sed_melt, grainsize_iow = get_size_kde(mp_pdd, boundaries_dict, grainsize_iow)
    sed_medians_from_grainsizes = prepare_data.get_medians_from_size_dist(grainsize_iow, boundaries_dict, 'SED_medians_from_grainsizes')
    sed_mode1s_from_grainsizes = prepare_data.get_mode1s_from_size_dist(grainsize_iow, boundaries_dict, 'SED_mode1s_from_grainsizes')
    dfs_to_merge = [
        sed_scor.rename_axis(index='Sample').reset_index(),
        KDE_medians,
        KDE_mode1,
        sed_medians_from_grainsizes,
        sed_mode1s_from_grainsizes,
        ]
    df = reduce(lambda  left,right: pd.merge(
        left,right,on=['Sample'], how='left'),
        dfs_to_merge,
        sdd_iow,
        )
    
    if raw_data_checkbox:
        df_expander(mp_pdd, "Filtered MP particle domain data")
        with st.expander("Plot particle properties"):
            particle_chart_selections, cols = get_selections(mp_pdd.columns.tolist(), ('size_geom_mean', 'Size_3_µm', 'Shape'), key='pddPlot_')
            particle_scatters, particle_reg_params = scatter_chart(
            mp_pdd, **particle_chart_selections,
            title='', width=800, height=600)
            cols[0].write(particle_scatters)
            cols[2].markdown('___', unsafe_allow_html=True)
            cols[2].text("")  # empty line to make some distance
            cols[2].write('Regression parameters:')
            cols[2].write(particle_reg_params)
        df_expander(df, "MP sample domain data", height=1000)       
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
    sample_chart_selections, cols = get_selections(featurelist, ('perc_MUD', 'Concentration', 'regio_sep'), key='sddPlot_')

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
        if family == 'Tweedie':
            tweedie_power = col1.number_input('Tweedie power:', value=2.0, min_value=0.0, max_value=3.0, step=0.1)
            Config.glm_tweedie_power = tweedie_power

        links = [None, 'Power', 'identity', 'inverse_power', 'inverse_squared', 'sqrt', 'log']
        link = col2.selectbox('Select link function (use None for default link of family):', links, index=1)
        Config.glm_link = link
        if link == 'Power':
            power_power = col2.number_input('Exponent of power link function:', value=-1.55, min_value=-10.0, max_value=10.0, step=0.1)
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
                                 width=500, height=300,
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

    # st.session_state  # activate to show stored widget variables in a dict