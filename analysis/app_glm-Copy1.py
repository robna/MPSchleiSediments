import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
# import pydeck as pdk

# from streamlit_vega_lite import altair_component
# alt.renderers.enable('altair_viewer')  # use to display altair charts externally in browser instead of inline (only activate in non-vega-compatible IDE like pycharm)
alt.data_transformers.disable_max_rows()

import prepare_data
import glm
from pcoa import sed_pcoa
from settings import Config

st.set_page_config(layout="wide")

featurelist = ['Concentration', 'ConcentrationA500', 'ConcentrationB500', 'MP_D50', 'PC1', 'PC2', 'Mass', 'GPS_LONs', 'GPS_LATs', 'Split', 'MP_D50', 'Depth',
               'MoM_ari_MEAN', 'MoM_ari_SORTING', 'MoM_ari_SKEWNESS', 'MoM_ari_KURTOSIS',
               'MoM_geo_MEAN', 'MoM_geo_SORTING', 'MoM_geo_SKEWNESS', 'MoM_geo_KURTOSIS',
               'MoM_log_MEAN', 'MoM_log_SORTING', 'MoM_log_SKEWNESS', 'MoM_log_KURTOSIS',
               'FW_geo_MEAN', 'FW_geo_SORTING', 'FW_geo_SKEWNESS', 'FW_geo_KURTOSIS',
               'FW_log_MEAN', 'FW_log_SORTING', 'FW_log_SKEWNESS', 'FW_log_KURTOSIS',
               'MODE 1 (µm)', 'MODE 2 (µm)', 'MODE 3 (µm)',
               'D10 (µm)', 'D50 (µm)', 'D90 (µm)', '(D90 div D10) (µm)', '(D90 - D10) (µm)', '(D75 div D25) (µm)', '(D75 - D25) (µm)',
               'perc GRAVEL', 'perc SAND', 'perc MUD',
               'perc V COARSE SAND', 'perc COARSE SAND', 'perc MEDIUM SAND', 'perc FINE SAND', 'perc V FINE SAND',
               'perc V COARSE SILT', 'perc COARSE SILT', 'perc MEDIUM SILT', 'perc FINE SILT', 'perc V FINE SILT',
               'perc CLAY', 'OM_D50', 'TOC', 'Hg', 'Dist_WWTP', 'regio_sep']

@st.cache()
def data_load_and_prep():
    # What happened so far: DB extract and blank procedure. Now import resulting MP data from csv
    mp_pdd = pd.read_csv('../data/env_MP_clean_list_SchleiSediments_NoIOWblindRemoval.csv', index_col=0)

    # Also import sediment data (sediment frequencies per size bin from master sizer export)
    grainsize_iow = pd.read_csv('../data/sediment_grainsize_IOW_vol_log-cau_not-closed.csv')
    # Get the binning structure of the imported sediment data and optionally rebin it (make binning coarser) for faster computation
    grainsize_iow, _ = prepare_data.sediment_preps(grainsize_iow)
    sedpco = sed_pcoa(grainsize_iow, num_coords = 2)
    
    return mp_pdd, sedpco

 
@st.cache()
def pdd2sdd(mp_pdd, regions):
    # ...some data wrangling to prepare particle domain data and sample domain data for MP and combine with certain sediment aggregates.
    mp_sdd = prepare_data.aggregate_SDD(mp_pdd)
    
    mp_added_sed_sdd = prepare_data.add_sediment(mp_sdd)
    mp_added_sed_sdd = mp_added_sed_sdd.loc[mp_added_sed_sdd.regio_sep.isin(regions)]  # filter based on selected regions

    return mp_added_sed_sdd


# def station_map(data):
#     data = data.loc[:,['Sample', 'GPS_LON', 'GPS_LAT']]
#     st.write(pdk.Deck(
#         map_style="mapbox://styles/mapbox/light-v9",
#         initial_view_state=pdk.data_utils.compute_view(data[['GPS_LON','GPS_LAT']]),  # {"latitude": 54.5770,"longitude": 9.8124,"zoom": 11,"pitch": 50},
#         layers=[
#             pdk.Layer(
#                 "HexagonLayer",
#                 data=data,
#                 get_position=["GPS_LON", "GPS_LAT"],
#                 radius=100,
#                 elevation_scale=100,
#                 elevation_range=[0, 1000],
#                 pickable=True,
#                 extruded=True,
#                 auto_highlight=True)]))
    

    # Define a layer to display on a map

#     layer = pdk.Layer(
#         "GridLayer", data, pickable=True, extruded=True, cell_size=200, elevation_scale=4, get_position=["GPS_LONs", "GPS_Lats"],
#     )

#     view_state = pdk.ViewState(latitude=54.5770, longitude=9.8124, zoom=11, bearing=0, pitch=45)

#     # Render
#     r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{position}\nCount: {count}"},)
#     st.write(r)


def poly_comp_chart(mp_pdd, mp_added_sed_sdd):
    poly_comp = prepare_data.aggregate_SDD(mp_pdd.groupby(['Sample', 'polymer_type']))
    poly_comp = poly_comp.merge(mp_added_sed_sdd[['Sample', 'Dist_WWTP', 'perc MUD']], on='Sample')
    
    selection = alt.selection_multi(fields=['polymer_type'], bind='legend')
    
    chart = alt.Chart(poly_comp.reset_index()).mark_bar().encode(
        x= alt.X('Dist_WWTP', scale=alt.Scale(type='linear')), #, sort = alt.SortField(field='Dist_WWTP', order='ascending')),
        y= alt.Y('Concentration', scale=alt.Scale(type='linear')),
        color= alt.Color('polymer_type', scale=alt.Scale(scheme='rainbow')),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip = ['Sample', 'polymer_type', 'Concentration', 'Dist_WWTP']
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
        tooltip=['Sample', x, y]
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
    
    samplefilter = st.sidebar.multiselect('Select samples:', mp_pdd.Sample.unique(), default=mp_pdd.Sample.unique())
    shapefilter = st.sidebar.multiselect('Select shapes:', ['irregular', 'fibre'], default=['irregular', 'fibre'])
    polymerfilter = st.sidebar.multiselect('Select polymers:', mp_pdd.polymer_type.unique(), default=mp_pdd.polymer_type.unique())
    mp_pdd = mp_pdd.loc[mp_pdd.Shape.isin(shapefilter)
                        & mp_pdd.polymer_type.isin(polymerfilter)
                        & mp_pdd.Sample.isin(samplefilter)]  # filter mp_pdd based on selected response features
    
    regionfilter = st.sidebar.multiselect('Select regions:', ['inner', 'outer', 'outlier', 'river'], default=['inner', 'outer', 'outlier', 'river'])
    mp_added_sed_sdd = pdd2sdd(mp_pdd, regionfilter)
           
    st.title('Microplastics and sediment analysis')
    st.markdown('___', unsafe_allow_html=True)
    # station_map(mp_pdd)  # plot map
    st.text("")  # empty line to make some distance
    
    st.subheader('Polymer composition')
    # st.markdown("Some text that describes what's going on here", unsafe_allow_html=True)
    
    st.write(poly_comp_chart(mp_pdd, mp_added_sed_sdd))
            
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance
    
    df = sedpco.merge(mp_added_sed_sdd, left_index=True, right_on='Sample')
    
    st.subheader('GLM')
    if st.checkbox('Calculate GLM'):
        col1, col2 = st.columns(2)
        family = col1.radio('Select ditribution family:', ['Gaussian', 'Poisson', 'Gamma', 'Tweedie', 'NegativeBinomial'], index=2)  # for neg.binom use CT-alpha-estimator from here: https://web.archive.org/web/20210303054417/https://dius.com.au/2017/08/03/using-statsmodels-glms-to-model-beverage-consumption/
        Config.glm_family = family

        link = col2.selectbox('Select link function (use None for default link of family):',
                              [None, 'identity', 'Power', 'inverse_power', 'sqrt', 'log'], index=0)
        Config.glm_link = link

        Config.glm_formula = st.text_input('GLM formula:', 'Concentration ~ Dist_WWTP + "D50 (µm)" + PC2')

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

    predx = st.selectbox('x-Values:', featurelist, index=11)
    predy = st.selectbox('y-Values:', featurelist, index=0)

    st.write(scatter_chart(df, predx, predy, title=''))
    
        
    st.markdown('___', unsafe_allow_html=True)
    st.text("")  # empty line to make some distance
           
    
if __name__ == "__main__":
    main()
