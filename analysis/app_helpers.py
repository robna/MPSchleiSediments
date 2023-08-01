import pandas as pd
from math import floor, ceil

from settings import Config, shortnames, regio_sep

import streamlit as st

def use_shortnames(df):
    return df.replace({'Sample': shortnames}).sort_values(by='Sample')


def general_settings():
    st.sidebar.write('**General controls**')
    raw_data_checkbox = st.sidebar.checkbox('Show raw data?')
    vertical_merge = st.sidebar.checkbox('Merge sediment corer samples (10 - 15 cm layer) ' \
                                                'with their respective sediment surface samples ' \
                                                '(0 - 5 cm layer) into single samples per station?',
                                                value=True, key='vertical_merge')
    st.sidebar.write('**Settings for particle size calculations (KDE, etc.)**')
    Config.sediment_grainsize_basis = st.sidebar.radio('Select basis for sediment grain size distributions', ['Volume_logscale', 'Volume_linscale', 'Counts_logscale'], index=0)
    Config.kde_weights = st.sidebar.radio('Select basis for MP size distributions ' \
                                          '(selecting "None" means distributions are particle count-based, ' \
                                          '"particle_volume_share" means volume distributions, ' \
                                          '"particle_mass_share" means mass distributions)',
                                          [None, 'particle_volume_share', 'particle_mass_share'], index=1)
    Config.fixed_bw = st.sidebar.number_input('Fixed bandwidth for MP size distribution KDEs (no optimisation)', value=18.0, min_value=0.0, max_value=200.0, step=10.0)
    Config.optimise_bw = st.sidebar.checkbox('Optimise KDE bandwidth for each sample? (very slow, check console output...)')
    Config.size_dim = st.sidebar.radio('Select size dimension', ['size_geom_mean', 'Size_1_µm', 'Size_2_µm', 'Size_3_µm',
                                                                 'vESD', 'particle_volume_µm3', 'particle_mass_µg'], index=4)
    return raw_data_checkbox, vertical_merge


def pdd_filters(mp_pdd):
    st.sidebar.write('**Filters for particle domain data**')
    Config.size_filter_on_sed_grainsizes = st.sidebar.checkbox('Apply set size limits also to sediment grain size data?')
    size_lims = floor(mp_pdd[Config.size_dim].min() / 10) * 10, ceil(mp_pdd[Config.size_dim].max() / 10) * 10
    Config.lower_size_limit = st.sidebar.number_input('Lower size limit (with respect to selected size dimension)',
                                                      value=size_lims[0],
                                                      min_value=0,
                                                      max_value=size_lims[1],
                                                      step=100)
    Config.upper_size_limit = st.sidebar.number_input('Upper size limit (with respect to selected size dimension)',
                                                      value=size_lims[1],
                                                      min_value=size_lims[0],
                                                      max_value=200_000_000,  # needs to be this large for particle_volume_µm3
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
    shapefilter = st.sidebar.multiselect('Select shapes:', mp_pdd.Shape.unique(), default=mp_pdd.Shape.unique())
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
    return mp_pdd


def sdd_filters(sdd):
    st.sidebar.write('**Filters for samples domain data**')
    regions = set(list(regio_sep.values()) + ['warnow']) if Config.warnow else set(list(regio_sep.values()))
    regionfilter = st.sidebar.multiselect('Select regions (works on sample domain data):', regions,
                                          default=regions)
    prop_filter = st.sidebar.selectbox('Select property for custom filtering:', sdd.select_dtypes('number').columns)
    prop_range = st.sidebar.slider(f'Select lower and upper limit for {prop_filter}:',
                                   float(sdd[prop_filter].min()), float(sdd[prop_filter].max()),
                                  [float(sdd[prop_filter].min()), float(sdd[prop_filter].max())]
                                  )
    sdd = sdd.loc[sdd.regio_sep.isin(regionfilter)
                  & (sdd[prop_filter] >= prop_range[0])
                  & (sdd[prop_filter] <= prop_range[1])
                 ]
    return sdd


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
    sel.y = col1.selectbox('y-Values:', optionlist, index=optionlist.index(defaults[1]), key=str(key)+'y')
    sel.x = col1.selectbox('x-Values:', optionlist, index=optionlist.index(defaults[0]), key=str(key)+'x')
    sel.color = col1.selectbox('Color:', [None, *optionlist], index=optionlist.index(defaults[2])+1, key=str(key)+'color')
    sel.xtransform = col2. checkbox('Log transform x-data', key=str(key)+'xtransform')
    sel.ytransform = col2. checkbox('Log transform y-data', key=str(key)+'ytransform')
    sel.reg = col2.radio('Regression type:', [None, 'linear', 'log', 'exp', 'pow'], index=0, key=str(key)+'reg')
    sel.reg_groups = col2.checkbox('Calculate separate regressions by color?', key=str(key)+'reg_groups')
    sel.xscale = col3.radio('X-Axis type:', ['linear', 'log', 'sqrt'], index=0, key=str(key)+'xscale')
    sel.yscale = col3.radio('Y-Axis type:', ['linear', 'log', 'sqrt'], index=0, key=str(key)+'yscale')
    sel.equal_axes = col3.checkbox('Equal axes?', key=str(key)+'equal_axes')
    sel.identity = col3.checkbox('Show identity line (dashed)?', key=str(key)+'identity')
    sel.linref = col3.checkbox('Show linear reference line (dotted)?', key=str(key)+'linref')
    if sel.linref is True:
        sel.linref_slope = col3.number_input('Line slope:', value=1.0, min_value=0.0, max_value=6.0, step=0.1, key=str(key)+'linref_slope')
        sel.linref_intercept = col3.number_input('Line offset:', value=0.0, min_value=-100.0, max_value=100.0, step=10.0, key=str(key)+'linref_intercept')
    sel.mix_lines = col3.checkbox('Show conservative mixing lines?', key=str(key)+'mix_lines')
    sel.labels = col3.selectbox('Labels:', [None, *optionlist], index=0, key=str(key)+'labels')
    cols = (col1, col2, col3)
    return sel.__dict__, cols
