import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# from streamlit_vega_lite import altair_component
# alt.renderers.enable('altair_viewer')  # use to display altair charts externally in browser instead of inline (only activate in non-vega-compatible IDE like pycharm)
alt.data_transformers.disable_max_rows()

try:  # if on phy-server local modules will not be found if their directory is not added to PATH
    import sys

    sys.path.append("/silod7/lenz/MPSchleiSediments/analysis/")
    import os
    os.chdir("/silod7/lenz/MPSchleiSediments/analysis/")
except Exception:
    pass

import prepare_data
import glm
from pcoa import sed_pcoa
from settings import Config

st.set_page_config(layout="wide")

predictorlist = ['PC1', 'PC2', 'D50', 'smaller63', 'TOC']
reponselist = ['Concentration', 'ConcentrationA500', 'ConcentrationB500', 'MP_D50']


@st.cache()
def data_load_and_prep():
    # What happened so far: DB extract and blank procedure. Now import resulting MP data from csv
    mp_pdd = pd.read_csv('../csv/env_MP_clean_list_SchleiSediments.csv', index_col=0)

    # The following is a hotfix for missing data on sampling weight.
    # TODO: Correct error in S29 in MPDB and reomove this line!
    mp_pdd.loc[mp_pdd.Sample == 'Schlei_S29', 'Sampling_weight_[kg]'] = 0.25

    # Also import sediment data (sediment frequencies per size bin from master sizer export)
    sed_sdd = pd.read_csv('../csv/Enders_export_10µm_linear_noR_RL.csv')
    # Get the binning structure of the imported sediment data and optionally rebin it (make binning coarser) for faster computation
    sed_sdd, _ = prepare_data.sediment_preps(sed_sdd)
    sedpco = sed_pcoa(sed_sdd, num_coords = 2)
    
    return mp_pdd, sedpco


def pdd2sdd(mp_pdd, shapes, polymers):
    
    # filter mp_pdd based on selected response features
    mp_pdd = mp_pdd.loc[mp_pdd.Shape.isin(shapes) & mp_pdd.polymer_type.isin(polymers)]
    
    # ...some data wrangling to prepare particle domain data and sample domain data for MP and combine with certain sediment aggregates.
    mp_sdd = prepare_data.aggregate_SDD(mp_pdd)
    mp_added_sed_sdd = prepare_data.add_sediment(mp_sdd)

    return mp_added_sed_sdd


def scatter_chart(df, x, y, title='Title'):
    df = df.reset_index()
    scatter = alt.Chart(df).mark_point().encode(
        x=x,
        y=y,
        color='regio_sep',
        tooltip='index'
    )

    RegLine = scatter.transform_regression(
        x, y, method="linear",
    ).mark_line(
        color="brown"
    )

    RegParams = scatter.transform_regression(
        x, y, method="linear", params=True
    ).mark_text(align='left', lineBreak='\n').encode(
        x=alt.value(120),  # pixels from left
        y=alt.value(20),  # pixels from top
        text='params:N'
    ).transform_calculate(
        params='"r² = " + round(datum.rSquared * 100)/100 + \
        "      y = " + round(datum.coef[1] * 100)/100 + "x" + " + " + round(datum.coef[0] * 10)/10'
    )

    return alt.layer(scatter, RegLine, RegParams).properties(width= 900, height= 450).properties(title=title)


def main():
    mp_pdd, sedpco = data_load_and_prep()
    
    col1, col2 = st.columns(2)

    st.title('Microplastics and sediment analysis')
    # st.write('')
    # st.write("Some text that describes what's going on here", unsafe_allow_html=True)
    st.text("")  # empty line to make some distance
    st.markdown('___', unsafe_allow_html=True)
    
    st.subheader('GLM')
    shapefilter = st.multiselect('Select shapes:', ['irregular', 'fibre'], default=['irregular', 'fibre'])
    polymerfilter = st.multiselect('Select polymers:', mp_pdd.polymer_type.unique(), default=mp_pdd.polymer_type.unique())
    
    mp_added_sed_sdd = pdd2sdd(mp_pdd, shapefilter, polymerfilter)
    df = sedpco.merge(mp_added_sed_sdd, left_index=True, right_on='Sample')
    
    Config.glm_formula = st.text_input('GLM formula:', 'Concentration ~ Dist_WWTP + D50 + PC2')
    
    # resp = st.sidebar.selectbox('Select Response', reponselist)
    glm_res = glm.glm(df)
    st.write(glm_res.summary())
    st.text("")  # empty line to make some distance
    
    df['yhat'] = glm_res.mu
    df['pearson_resid'] = glm_res.resid_pearson
    
    col1.write(scatter_chart(df, 'yhat', Config.glm_formula.split(' ~')[0], title='GLM - y vs. yhat'))
    st.text("")  # empty line to make some distance
    
    from matplotlib import pyplot as plt
    from statsmodels.graphics.api import abline_plot
    import statsmodels.api as sm
    plt.rc("figure", figsize=(16,8))
    plt.rc("font", size=10)
    fig, ax = plt.subplots()
    ax.scatter(df.yhat, df.Concentration)
    line_fit = sm.OLS(df.Concentration, sm.add_constant(df.yhat, prepend=True)).fit()
    abline_plot(model_results=line_fit, ax=ax)

    ax.set_title('Model Fit Plot')
    ax.set_ylabel('Observed values')
    ax.set_xlabel('Fitted values')
    
    col2.pyplot(fig)


    
    
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance
    st.subheader('Predictor colinearity check')
    predx = st.selectbox('Predictor 1', predictorlist)
    predy = st.selectbox('Predictor 2', predictorlist, index=1)
    

    st.write(scatter_chart(df, predx, predy, title=''))
    
    st.markdown('___', unsafe_allow_html=True)

    
    
if __name__ == "__main__":
    main()
