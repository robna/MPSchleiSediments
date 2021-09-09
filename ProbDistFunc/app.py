import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
# from streamlit_vega_lite import altair_component
# alt.renderers.enable('altair_viewer')  # use to display altair charts externally in browser instead of inline (only activate in non-vega-compatible IDE like pycharm)
alt.data_transformers.disable_max_rows()

try:  # if on phy-server local modules will not be found if their directory is not added to PATH
    import sys
    sys.path.append("/silod7/lenz/MPSchleiSediments/ProbDistFunc/")
    import os
    os.chdir("/silod7/lenz/MPSchleiSediments/ProbDistFunc/")
except Exception:
    pass

import KDE_prepare_data
import KDE_utils
import correlations
from KDE_settings import Config

st.set_page_config(layout="wide")

@st.cache()
def data_load_and_prep():
    pdd_MP = pd.read_csv('../csv/env_MP_clean_list_SchleiSediments.csv', index_col=0)
    sed_size_freqs, sed_x_d = KDE_prepare_data.sediment_preps(sed_size_freqs, rebinning=True)

    sdd_MP = KDE_prepare_data.aggregate_SDD(pdd_MP)
    sdd_MP_sed = KDE_prepare_data.add_sediment(sdd_MP)
    pdd_sdd_MP = KDE_prepare_data.sdd2pdd(sdd_MP_sed, pdd_MP)

    size_pdfs = KDE_utils.per_sample_kde(pdd_sdd_MP, x_d = sed_x_d)
    MP_size_conc = KDE_utils.probDens2conc(size_pdfs, sdd_MP_sed)
    MP_size_conc, sed_size_freqs, MPsedMelted = KDE_prepare_data.equalise_MP_and_Sed(MP_size_conc, sed_size_freqs)
    
    return MPsedMelted

def main():
    st.title('Microplastics and sediment analysis')
    st.write('')
    st.write("Some text that describes what's going on here",unsafe_allow_html=True)

if __name__ == "__main__":
    main()