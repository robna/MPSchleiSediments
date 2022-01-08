import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns

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
import KDE_utils
import correlations
from settings import Config

st.set_page_config(layout="wide")


@st.cache()
def data_load_and_prep():
    pdd_MP = pd.read_csv('../csv/env_MP_clean_list_SchleiSediments.csv', index_col=0)
    sed_size_freqs = pd.read_csv('../csv/sediment_grainsize_IOW.csv')
    sed_size_freqs, sed_x_d = prepare_data.sediment_preps(sed_size_freqs, rebinning=False)

    sdd_MP = prepare_data.aggregate_SDD(pdd_MP)
    sdd_MP_sed = prepare_data.add_sediment(sdd_MP)
    pdd_sdd_MP = prepare_data.sdd2pdd(sdd_MP_sed, pdd_MP)

    size_pdfs = KDE_utils.per_sample_kde(pdd_sdd_MP, x_d=sed_x_d)
    MP_size_conc = KDE_utils.probDens2conc(size_pdfs, sdd_MP_sed)
    MP_size_conc, sed_size_freqs, MPsedMelted = prepare_data.equalise_mp_and_sed(MP_size_conc, sed_size_freqs)

    return MP_size_conc, sed_size_freqs, sed_x_d


def scatter_chart(df):
    df = df.T.reset_index().rename(columns={df.index.name: 'sample'})
    scatter = alt.Chart(df).mark_point().encode(
        x='SED',
        y='MP',
        tooltip='sample'
    )

    RegLine = scatter.transform_regression(
        'SED', 'MP', method="linear",
    ).mark_line(
        color="brown"
    )

    RegParams = scatter.transform_regression(
        'SED', 'MP', method="linear", params=True
    ).mark_text(align='left', lineBreak='\n').encode(
        x=alt.value(120),  # pixels from left
        y=alt.value(20),  # pixels from top
        text='params:N'
    ).transform_calculate(
        params='"r² = " + round(datum.rSquared * 100)/100 + \
        "      y = " + round(datum.coef[1] * 100)/100 + "x" + " + " + round(datum.coef[0] * 10)/10'
    )

    return alt.layer(scatter, RegLine, RegParams).properties(width= 600, height= 450)


def main():
    MP_size_conc, sed_size_freqs, sed_x_d = data_load_and_prep()

    st.title('Microplastics and sediment analysis')
    # st.write('')
    # st.write("Some text that describes what's going on here", unsafe_allow_html=True)

    MPslider = st.sidebar.slider(
        'MP size range',
        0, 2000, [50, 250])

    SEDslider = st.sidebar.slider(
        'SED size range',
        0, 2000, [50, 250])

    MPlow = min(sed_x_d, key=lambda x: abs(x - MPslider[0]))
    MPup = min(sed_x_d[sed_x_d > MPlow], key=lambda x: abs(x - MPslider[1]))
    SEDlow = min(sed_x_d, key=lambda x: abs(x - SEDslider[0]))
    SEDup = min(sed_x_d[sed_x_d > SEDlow], key=lambda x: abs(x - SEDslider[1]))

    MPnow = MP_size_conc[(MP_size_conc.index.str.split('_').str[0].astype(int) >= MPlow) &
                         (MP_size_conc.index.str.split('_').str[1].astype(int) <= MPup)]
    SEDnow = sed_size_freqs[(sed_size_freqs.index.str.split('_').str[0].astype(int) >= SEDlow) &
                          (sed_size_freqs.index.str.split('_').str[1].astype(int) <= SEDup)]

    df = pd.DataFrame([MPnow.sum(), SEDnow.sum()], index=['MP', 'SED'])

    st.write(scatter_chart(df))
    
    MP_bin_width = st.select_slider('Choose MP bin width', options=sed_x_d)
    SED_bin_width = st.select_slider('Choose SED bin width', options=sed_x_d)
    
#     mp_size_conc['bin_widths'] = mp_size_conc.index.str.split('_').str[1].astype(int) - mp_size_conc.index.str.split('_').str[0].astype(int)
#     grainsize_iow.bin_widths = grainsize_iow.index.str.split('_').str[1].astype(int) - grainsize_iow.index.str.split('_').str[0].astype(int)
    
    MPbins = MP_size_conc.groupby(MP_size_conc.reset_index(drop=True).index // MP_bin_width).sum()
    SEDbins = MP_size_conc.groupby(MP_size_conc.reset_index(drop=True).index // MP_bin_width).sum()
    
    df2 = correlations.rangecorr(MPbins, SEDbins)
    
    MPbins
    MP_bin_width
    MP_size_conc
       
    st.write(sns.heatmap(df2, cmap ='RdYlGn'))
    
    
    
    

    

if __name__ == "__main__":
    main()
