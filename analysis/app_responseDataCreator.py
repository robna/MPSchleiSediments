import numpy as np
import pandas as pd
import streamlit as st

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
    return pdd_MP


