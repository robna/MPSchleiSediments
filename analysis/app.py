import pandas as pd
import streamlit as st
import altair as alt
from streamlit_vega_lite import altair_component
import prepare_data
import KDE_utils
from settings import Config

# alt.renderers.enable('altair_viewer')  # use to display altair charts externally in browser instead of inline (only activate in non-vega-compatible IDE like pycharm)
alt.data_transformers.disable_max_rows()

try:  # if on phy-server local modules will not be found if their directory is not added to PATH
    import sys
    sys.path.append("/silod7/lenz/MPSchleiSediments/analysis/")
    import os
    os.chdir("/silod7/lenz/MPSchleiSediments/analysis/")
except Exception:
    pass

import os
os.chdir("/analysis")

st.set_page_config(layout="wide")


@st.cache()
def data_load_and_prep():
    pdd_MP = pd.read_csv('../csv/env_MP_clean_list_SchleiSediments.csv', index_col=0)
    sed_size_freqs = pd.read_csv('../csv/Enders_export_10µm_linear_noR_RL.csv')
    sed_size_freqs, sed_x_d = KDE_prepare_data.sediment_preps(sed_size_freqs, rebinning=False)

    sdd_MP = KDE_prepare_data.aggregate_SDD(pdd_MP)
    sdd_MP_sed = KDE_prepare_data.add_sediment(sdd_MP)
    pdd_sdd_MP = KDE_prepare_data.sdd2pdd(sdd_MP_sed, pdd_MP)

    size_pdfs = KDE_utils.per_sample_kde(pdd_sdd_MP, x_d=sed_x_d)
    MP_size_conc = KDE_utils.probDens2conc(size_pdfs, sdd_MP_sed)
    MP_size_conc, sed_size_freqs, MPsedMelted = KDE_prepare_data.equalise_mp_and_sed(MP_size_conc, sed_size_freqs)

    return MP_size_conc, sed_size_freqs, sed_x_d


def filter_data(MP_size_conc, sed_size_freqs):
    MPnow = MP_size_conc[(MP_size_conc.index.str.split('_').str[0].astype(int) >= Config.MPlow) &
                         (MP_size_conc.index.str.split('_').str[1].astype(int) <= Config.MPup)]
    SEDnow = sed_size_freqs[(sed_size_freqs.index.str.split('_').str[0].astype(int) >= Config.SEDlow) &
                            (sed_size_freqs.index.str.split('_').str[1].astype(int) <= Config.SEDup)]

    df = pd.DataFrame([MPnow.sum(), SEDnow.sum()], index=['MP', 'SED'])

    return df


@st.cache()
def select_charts1(df_MP):
    df_MP.index = df_MP.index.str.split('_').str[0]
    df_MP = df_MP.T.rename_axis('sample').reset_index().melt(id_vars='sample', var_name='lower_MP', value_name='MP')

    brush1 = alt.selection_interval(name="brush1", encodings=['y'])

    MP = alt.Chart(df_MP).mark_line().encode(
        x='mean(MP)',
        y=alt.X('lower_MP:Q', scale=alt.Scale(type='linear'))
    ).add_selection(
        brush1
    ).properties(
        height=300,
        width=100
    )

    return MP


@st.cache()
def select_charts2(df_SED):
    df_SED.index = df_SED.index.str.split('_').str[0]
    df_SED = df_SED.T.rename_axis('sample').reset_index().melt(id_vars='sample', var_name='lower_SED', value_name='SED')

    brush2 = alt.selection_interval(name="brush2", encodings=['x'])

    sed = alt.Chart(df_SED).mark_line().encode(
        x=alt.X('lower_SED:Q', scale=alt.Scale(type='linear')),
        y='mean(SED)'
    ).add_selection(
        brush2
    ).properties(
        height=100,
        width=400
    )

    return sed


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

    return scatter + RegLine + RegParams


def chart_composer(df_MP, df_SED, filtered_df):

    scats = scatter_chart(filtered_df)
    MP_select, SED_select = select_charts(df_MP, df_SED)

    return MP_select | scats & SED_select


def main():
    MP_size_conc, sed_size_freqs, sed_x_d = data_load_and_prep()

    st.title('Microplastics and sediment analysis')
    st.write('')
    st.write("Some text that describes what's going on here")

    df = filter_data(MP_size_conc, sed_size_freqs)

    # event_dict = altair_component(altair_chart=chart_composer(mp_size_conc.copy(), sed_sdd.copy(), df))
    event_dict1 = altair_component(altair_chart=select_charts1(MP_size_conc.copy()))
    event_dict2 = altair_component(altair_chart=select_charts2(sed_size_freqs.copy()))

    MPslider = event_dict1.get('y')
    SEDslider = event_dict2.get('x')

    if MPslider:
        Config.MPlow = min(sed_x_d, key=lambda x: abs(x - MPslider[0]))
        Config.MPup = min(sed_x_d[sed_x_d > Config.MPlow], key=lambda x: abs(x - MPslider[1]))

    if SEDslider:
        Config.SEDlow = min(sed_x_d, key=lambda x: abs(x - SEDslider[0]))
        Config.SEDup = min(sed_x_d[sed_x_d > Config.SEDlow], key=lambda x: abs(x - SEDslider[1]))

    MPslider
    SEDslider

    st.write(scatter_chart(df))

    # st.write(scatter_chart(df))
    # st.write(select_charts(mp_size_conc.copy(), sed_sdd.copy()))

if __name__ == "__main__":
    main()
