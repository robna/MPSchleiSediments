import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk

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

featurelist = ['Concentration', 'ConcentrationA500', 'ConcentrationB500', 'MP_D50',
                 'PC1', 'PC2', 'Mass', 'GPS_LONs', 'GPS_LATs', 'Split',
                 'MP_D50', 'D50', 'smaller63', 'TOC', 'Hg', 'Dist_WWTP']

@st.cache()
def data_load_and_prep():
    # What happened so far: DB extract and blank procedure. Now import resulting MP data from csv
    mp_pdd = pd.read_csv('../csv/env_MP_clean_list_SchleiSediments.csv', index_col=0)

    # Also import sediment data (sediment frequencies per size bin from master sizer export)
    sed_sdd = pd.read_csv('../csv/Enders_export_10µm_linear_noR_RL.csv')
    # Get the binning structure of the imported sediment data and optionally rebin it (make binning coarser) for faster computation
    sed_sdd, _ = prepare_data.sediment_preps(sed_sdd)
    sedpco = sed_pcoa(sed_sdd, num_coords = 2)
    
    return mp_pdd, sedpco

 
def pdd2sdd(mp_pdd, regions):
    # ...some data wrangling to prepare particle domain data and sample domain data for MP and combine with certain sediment aggregates.
    mp_sdd = prepare_data.aggregate_SDD(mp_pdd)
    
    mp_added_sed_sdd = prepare_data.add_sediment(mp_sdd)
    mp_added_sed_sdd = mp_added_sed_sdd.loc[mp_added_sed_sdd.regio_sep.isin(regions)]  # filter based on selected regions

    return mp_added_sed_sdd


def map(data):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": 54.5770,
            "longitude": 9.8124,
            "zoom": 11,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "ColumnLayer",
                data=data,
                get_position=["GPS_LONs", "GPS_Lats"],
                get_elevation="Concentration",
                radius=100,
                elevation_scale=100,
                #elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                auto_highlight=True
            )
        ]
    ).to_html())


def poly_comp_chart(mp_pdd):
    
    poly_comp = prepare_data.aggregate_SDD(mp_pdd.groupby(['Sample', 'polymer_type']))
    
    sample_order = ['Schlei_S1_15cm', 'Schlei_S2','Schlei_S2_15cm', 'Schlei_S3',
                    'Schlei_S4', 'Schlei_S4_15cm', 'Schlei_S5', 'Schlei_S6',
                    'Schlei_S7', 'Schlei_S8', 'Schlei_S10', 'Schlei_S10_15cm',
                    'Schlei_S11', 'Schlei_S13', 'Schlei_S14', 'Schlei_S15',
                    'Schlei_S16', 'Schlei_S17', 'Schlei_S19', 'Schlei_S20',
                    'Schlei_S21', 'Schlei_S22', 'Schlei_S23', 'Schlei_S24',
                    'Schlei_S25', 'Schlei_S26', 'Schlei_S27', 'Schlei_S29',
                    'Schlei_S30', 'Schlei_S31', 'Schlei_S32']
    
    selection = alt.selection_multi(fields=['polymer_type'], bind='legend')
    
    chart = alt.Chart(poly_comp.reset_index()).mark_bar().encode(
        x= alt.X('Sample',sort = sample_order),
        y= alt.Y('Concentration',scale = alt.Scale(type ='linear')),
        color= 'polymer_type',
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip = ['polymer_type', 'Concentration']
    ).add_selection(
        selection
    )
    
    return chart | chart.encode(y=alt.Y('Concentration',stack='normalize'))


def scatter_chart(df, x, y, title='Title'):
    df = df.reset_index()
    scatter = alt.Chart(df).mark_point().encode(
        x=x,
        y=y,
        color='regio_sep',
        tooltip='Sample'
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
        params='"R² = " + round(datum.rSquared * 100)/100 + \
        "      y = " + round(datum.coef[1] * 100)/100 + "x" + " + " + round(datum.coef[0] * 10)/10'
    )

    return alt.layer(scatter, RegLine, RegParams).properties(width= 600, height= 450).properties(title=title)


def main():
    mp_pdd, sedpco = data_load_and_prep()  # load data
    
    shapefilter = st.sidebar.multiselect('Select shapes:', ['irregular', 'fibre'], default=['irregular', 'fibre'])
    polymerfilter = st.sidebar.multiselect('Select polymers:', mp_pdd.polymer_type.unique(), default=mp_pdd.polymer_type.unique())
        
    st.title('Microplastics and sediment analysis')
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance
    
    st.subheader('Polymer composition')
    # st.markdown("Some text that describes what's going on here", unsafe_allow_html=True)
    
    mp_pdd = mp_pdd.loc[mp_pdd.Shape.isin(shapefilter) & mp_pdd.polymer_type.isin(polymerfilter)]  # filter mp_pdd based on selected response features
    st.write(poly_comp_chart(mp_pdd))
            
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance
    
    st.subheader('GLM')
    regionfilter = st.multiselect('Select regions:', ['inner', 'outer', 'outlier', 'river'], default=['inner', 'outer', 'outlier', 'river'])
    mp_added_sed_sdd = pdd2sdd(mp_pdd, regionfilter)
    df = sedpco.merge(mp_added_sed_sdd, left_index=True, right_on='Sample')
    # map(df)
    
    col1, col2 = st.columns(2)
    family = col1.radio('Select ditribution family:', ['Gaussian', 'Poisson', 'Gamma', 'Tweedie'], index=2)
    Config.glm_family = family
    
    link = col2.selectbox('Select link function (use None for default link of family):',
                          [None, 'identity', 'Power', 'inverse_power', 'sqrt', 'log'], index=0)
    Config.glm_link = link
    
    Config.glm_formula = st.text_input('GLM formula:', 'Concentration ~ Dist_WWTP + D50 + PC2')
    
    # resp = st.sidebar.selectbox('Select Response', reponselist)
    glm_res = glm.glm(df)
    
    col1, col2, col3= st.columns(3)
    col1.write(glm_res.summary())
    
    st.text("")  # empty line to make some distance
    
    df['yhat'] = glm_res.mu
    df['pearson_resid'] = glm_res.resid_pearson
    col2.write(scatter_chart(df, 'yhat', Config.glm_formula.split(' ~')[0], title='GLM --- y vs. yhat'))
    col3.write(scatter_chart(df, 'yhat', 'pearson_resid', title='GLM --- Pearson residuals'))
    
    resid = glm_res.resid_deviance.copy()
    from statsmodels import graphics
    col3.pyplot(graphics.gofplots.qqplot(resid, line='r'))
    
    st.text("")  # empty line to make some distance
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance
    st.subheader('Single predictor correlation and colinearity check')
    
    predx = st.selectbox('x-Values:', featurelist)
    predy = st.selectbox('y-Values:', featurelist, index=1)
    
    st.write(scatter_chart(df, predx, predy, title=''))
    
        
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance
           
    
if __name__ == "__main__":
    main()
