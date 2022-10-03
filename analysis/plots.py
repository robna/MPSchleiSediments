from cv2 import norm
import numpy as np
import pandas as pd

import altair as alt
alt.data_transformers.disable_max_rows()
# alt.renderers.enable('altair_viewer')  # use to display altair charts externally in browser instead of inline (only activate in non-vega-compatible IDE like pycharm)
import altair_transform

import seaborn as sns
sns.set_style('whitegrid')

from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
    
import pydeck as pdk
import streamlit as st
from sklearn.metrics import r2_score
# from streamlit_vega_lite import altair_component

from settings import Config
import prepare_data


def conserv_mixin_lines(df, predx, predy, color):
    """
    As an addon for scatter_chart, this plots
    straight lines between the data points with
    the smallest and largest values of the
    x-variable within coloured groups.
    Can be used to show conservative mixing
    lines of property-property plots.

    :param df: dataframe with x and y variables
    :param predx: name of x variable
    :param predy: name of y variable
    :param color: variable which is used for grouping
    :return: chart object
    """

    if color:
        lst = []
        for groupname, group in df.groupby([color]):
            lsmin = {
                color: groupname,
                predx: group[predx].min(),
                predy: float(group.loc[group[predx] == group[predx].min(), predy].values)
                }
            lsmax = {
                color: groupname,
                predx: group[predx].max(),
                predy: float(group.loc[group[predx] == group[predx].max(), predy].values)
                }
            lst.append(lsmin)
            lst.append(lsmax)
            
        df2 = pd.DataFrame(lst)

    else:
        df2 = pd.DataFrame({
            predx: [
                df[predx].min(),
                df[predx].max()
                ],
            predy: [
                float(df.loc[df[predx] == df[predx].min(), predy].values),
                float(df.loc[df[predx] == df[predx].max(), predy].values)
                ]
            })
      
    conserv_mix = alt.Chart(df2).mark_line(strokeDash=[3,8]).encode(
        x=predx,
        y=predy,
        color=alt.Color(color, legend=None) if color else alt.value('black') # all below is not working, eg. no groupby in altair filter or calculate transform, hence the workaround via pandas groupby above...
    # ).transform_joinaggregate(
    #     x = f'if(datum.{predx} < max(datum.{predx}), min(datum.{predx}), max(datum.{predx}))',
    #     y = f'if(datum.{predx} < max(datum.{predx}), min(datum.{predy}), max(datum.{predy}))',
    #     groupby = [c]
    # ).transform_filter(
    #     alt.FieldOneOfPredicate(field=predx, oneOf=[df[predx].min(), df[predx].max()])
    # ).transform_calculate(
    #     # x = f'datum.{predx} < max(datum.{predx}) ? min(datum.{predx}) : max(datum.{predx})',
    #     # y = f'datum.{predx} < max(datum.{predx}) ? min(datum.{predy}) : max(datum.{predy})'
    #     x = f'if(datum.{predx} < max(datum.{predx}), min(datum.{predx}), max(datum.{predx}))',
    #     y = f'if(datum.{predx} < max(datum.{predx}), min(datum.{predy}), max(datum.{predy}))'
    )

    return conserv_mix


def scatter_chart(
    df, x, y,
    color=False, labels=None,
    reg=None, reg_groups=False,
    equal_axes=False,
    identity=False,
    mix_lines=False,
    xtransform=False, ytransform=False, xscale='linear', yscale='linear',
    title='', width=400, height=300):

    """
    Create a scatter plot with optional regression line and equation.
    :param df: dataframe with x and y columns
    :param x: x column name
    :param y: y column name
    :param color: color column name, default: False (means no colored groups)
    :param labels: label column name (prints label on each point), None if no label (default)
    :param reg: None (or False) for no regression line (default), 'linear', 'log', 'exp' or 'pow'
    :param reg_groups: True for regression line for each colored group (default: False)
    :param equal_axes: True for x and y axes ranging from 0 to their higher maximum (useful for predicted vs. observed plots) (default=False)
    :param identity: True for showing identity line (default=False)
    :param mix_lines: True for showing conservative mixing lines (default=False)
    :param xtransform: when True take np.log10 of x-values before plotting, be carful when also using non-linear axis scales (default=False)
    :param ytransform: when True take np.log10 of y-values before plotting, be carful when also using non-linear axis scales (default=False)
    :param xscale: scale to use on x axis, str, any of
                 ['linear', 'log', 'pow', 'sqrt', 'symlog', 'identity', 'sequential', 'time', 'utc', 'quantile', 'quantize', 'threshold', 'bin-ordinal', 'ordinal', 'point', 'band']
    :param yscale: scale to use on y axis, str, see xscale for options
    :param title: plot title
    :param width: plot width
    :param height: plot height
    :return: altair chart and df of regression parameter (None if reg=None)
    """
    
    df = df.copy()  # avoid changing original dataframe

    if xtransform:
        df[x] = np.log10(df[x])
    
    if ytransform:
        df[y] = np.log10(df[y])

    maxX = df[x].max()
    maxY = df[y].max()
    domain_max = max(maxX, maxY) * 1.05

    minX = df[x].min()
    minY = df[y].min()

    df['common_group'] = 'all'  # generate a common group for all points
    df = df.reset_index()

    base = alt.Chart(df).mark_circle(size=100).encode(
        x=alt.X(x, scale=alt.Scale(type=xscale), axis= alt.Axis(title = f'log10 of {x}') if xtransform else alt.Axis(title = x)),
        y=alt.Y(y, scale=alt.Scale(type=yscale), axis= alt.Axis(title = f'log10 of {y}') if ytransform else alt.Axis(title = y)),
        tooltip=[df.columns[1], x, y])

    if equal_axes:
        base = base.encode(
            x=alt.X(x, scale=alt.Scale(domain=[0, domain_max])),
            y=alt.Y(y, scale=alt.Scale(domain=[0, domain_max])))

    if color:
        scatter = base.encode(
            alt.Color(
                color,
                # scale=alt.Scale(scheme='cividis')
                )
                )  # TODO: only ordinal works with categorical and quantiative data and also shows regrassion, when not specifying data type it works fully for categorical data, but stops showing the regression when quantitative data is chosen for color.
    else:
        scatter = base

    if labels:
        scatter = scatter + scatter.mark_text(
            align='left',
            baseline='middle',
            dx=7
        ).encode(
            text=f'{labels}:N'
        )

    if reg:

        if reg_groups:
            R2_string = '"y = "'
            coef0_string = '"a"'
            coef1_string = '"b"'

        else:
            R2_string = '"R² = " + round(datum.rSquared * 100)/100 + "      y = "'
            coef0_string = 'round(datum.coef[0] * 1000)/1000'
            coef1_string = 'round(datum.coef[1] * 1000)/1000'

        reg_eq = {'linear': f'{R2_string} + {coef1_string} + " * x + " + {coef0_string}',
                  'log': f'{R2_string} + {coef1_string} + " * log(x) + " + {coef0_string}',
                  'exp': f'{R2_string} + {coef0_string} + " * e ^ (" + {coef1_string} + " x)"',
                  'pow': f'{R2_string} + {coef0_string} + " * x^" + {coef1_string}'}
        
        RegLine = base.mark_line().transform_regression(
            x, y, method=reg, groupby=[color if reg_groups else 'common_group']
        ).encode(
        color=alt.Color(color if reg_groups else 'common_group', legend=None)
        )

        RegParams = base.transform_regression(
            x, y, method=reg, groupby=[color if reg_groups else 'common_group'], params=True
        )
        
        RegEq = RegParams.mark_text(align='left', lineBreak='\n', fontSize = 18).encode(
            x=alt.value(width / 5),  # pixels from left
            y=alt.value(height / 20),  # pixels from top
            color=alt.Color(color if reg_groups else 'common_group', legend=None),
            text='params:N'
        ).transform_calculate(
            params=reg_eq[reg]
        )

        chart = alt.layer(scatter, RegLine, RegEq)
        params = altair_transform.extract_data(RegParams)
        params = pd.concat([pd.DataFrame(params.coef.tolist(), columns=['a', 'b']), params[['rSquared', 'keys']]], axis=1)

    else:
        chart = alt.layer(scatter)
        params= None

    if identity:
        identityLine = base.mark_line(
            color= 'black',
            strokeDash=[3,8],
            strokeWidth=0.6,
            clip=True
        ).encode(
            x=alt.X(x, scale=alt.Scale(domain=[minX, maxX])),
            y=alt.Y(x, scale=alt.Scale(domain=[minY, maxY]))
        )

        identityR2 = r2_score(df[x], df[y])
        
        identityR2Text = alt.Chart(df).mark_text(  # TODO: could use empty dataset "{'values':[{}]}" instead of "df" to print text only once?
            fontSize=12,
            align="left", baseline="top"
        ).encode(
            x=alt.value(width / 1.8),  # pixels from left
            y=alt.value(height / 100),  # pixels from top
            text=alt.value(f'identity_R² = {identityR2:.2f}')
        )
        
        chart = alt.layer(chart, identityLine, identityR2Text)

    if mix_lines:
        chart = alt.layer(chart, conserv_mixin_lines(df, x, y, color)).resolve_scale(
        color='independent'
    )
    
    chart = chart.resolve_scale(
        color='independent'
    ).configure_axis(
        labelFontSize=18,
        titleFontSize=20
    ).configure_legend(
        orient='top'
    ).properties(
        width=width,
        height=height,
        title=title
    ).interactive()

    # chart.save('../plots/scatter_chart.html')  # activate save chart to html file

    return chart, params


def poly_comp_chart(mp_pdd, sdd_iow):
    poly_comp = prepare_data.aggregate_SDD(mp_pdd.groupby(['Sample', 'polymer_type']))
    poly_comp = poly_comp.merge(sdd_iow[['Sample', 'Dist_WWTP', 'perc MUD']], on='Sample')
    selection = alt.selection_multi(fields=['polymer_type'], bind='legend')

    chart_abs = alt.Chart(poly_comp.reset_index()).mark_bar().encode(
        x=alt.X('Dist_WWTP', scale=alt.Scale(type='linear')),
        # , sort = alt.SortField(field='Dist_WWTP', order='ascending')),
        y=alt.Y('Concentration', stack=True),
        color=alt.Color('polymer_type', scale=alt.Scale(scheme='rainbow')),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.3)),
        tooltip=['Sample', 'polymer_type', 'Concentration', 'Dist_WWTP']
    ).add_selection(
        selection
    )

    chart_rel = chart_abs.encode(y=alt.Y('Concentration', stack='normalize'))

    chart_tot = chart_rel.mark_bar().encode(
        x=alt.value(10),
        y=alt.Y('all_stations_summed:Q', stack='normalize'),
        tooltip=['polymer_type', 'all_stations_summed:Q']
    ).transform_aggregate(
        all_stations_summed='sum(Concentration)',
        groupby=['polymer_type']
    )

    chart = alt.hconcat(chart_abs.interactive(), chart_rel, chart_tot).resolve_scale('independent')
    chart.save('../plots/poly_comp_chart.html')  # activate save chart to html file

    return chart  # | chart.encode(y=alt.Y('Concentration',stack='normalize'))


def station_map(data):
    data = data.loc[:, ['Sample', 'GPS_LON', 'GPS_LAT']]
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.data_utils.compute_view(data[['GPS_LON', 'GPS_LAT']]),
        # {"latitude": 54.5770,"longitude": 9.8124,"zoom": 11,"pitch": 50},
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["GPS_LON", "GPS_LAT"],
                radius=100,
                elevation_scale=100,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                auto_highlight=True)]))

    # Define a layer to display on a map

    layer = pdk.Layer(
        "GridLayer", data, pickable=True, extruded=True, cell_size=200, elevation_scale=4,
        get_position=["LON", "LAT"],
    )

    view_state = pdk.ViewState(latitude=54.5770, longitude=9.8124, zoom=11, bearing=0, pitch=45)

    # Render
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{position}\nCount: {count}"}, )
    st.write(r)


def histograms(df):
    brush = alt.selection_interval(encodings=['x'])
    base = alt.Chart(df).properties(width=1000, height=200)

    bar = base.mark_bar().encode(
        x=alt.X(Config.size_dim, bin=alt.Bin(maxbins=50)),
        y='count():Q',
        tooltip='count():Q'
    )

    rule = base.mark_rule(color='red').encode(
        x=f'median({Config.size_dim}):Q',
        size=alt.value(3)
    )

    chart = alt.vconcat(
        # bar.encode(alt.X(Config.size_dim,
        #                  bin=alt.Bin(maxbins=50, extent=brush),
        #                  scale=alt.Scale(domain=brush)
        #                  ),
        bar.add_selection(brush) + rule
    )

    return chart


def biplot(scor, load, expl, discr, x, y, sc, ntf=5, normalise=None,
           figsize=(800, 600)):  # TODO: normalisation not yet implemented
    """
    Create the Biplot based on the PCoA or PCA scores and loadings.

    Parameters
    ----------
    scor : pandas.DataFrame containing the scores of the PCoA
    load : pandas.DataFrame containing the loadings of the PCoA
    expl : pandas.Series containing the proportion explained by each PCoA
    discr : pandas.DataFrame containing discriminators of samples
    x : component to plot on the x-axis
    y : component to plot on the y-axis
    sc : component to plot on the color scale for scores
    ntf : number of top features to plot, default = 5
    normalise : normalise scores and loadings: 'maxabs' = [-1,1], 'standard' = [mean=0, std=1], default: None
    figsize : (float, float), optional, default: 800, 600

    Returns
    -------
    altair figure, as html and inline
    """
    load.rename_axis('features', inplace=True)

    if normalise == 'maxabs': # MaxAbsScaler
        load = load.apply(lambda x: x / np.max(np.abs(x)))
        scor = scor.apply(lambda x: x / np.max(np.abs(x)))

    elif normalise == 'standard': # StandardScaler
        load = load.apply(lambda x: (x - x.mean()) / x.std())
        scor = scor.apply(lambda x: (x - x.mean()) / x.std())

    dfl = load.head(ntf).append(load.head(ntf) - load.head(ntf)).reset_index()
    # dfl = (dfl / dfl.max(numeric_only=True).max(numeric_only=True))  # normalise values to range [-1,1]

    dfl2 = load.head(ntf).reset_index()
    # dfl2 = (dfl2 / dfl2.max().max())

    dfs = scor.reset_index().rename(columns={'index': 'Sample'}).merge(discr[['Sample', 'Dist_WWTP', 'regio_sep', 'Concentration']])
    # dfs = (dfs / dfs.max(numeric_only=True).max(numeric_only=True))

    lines = alt.Chart(dfl).mark_line(opacity=0.3).encode(
        x=x,
        y=y,
        detail='features'
        # color=alt.Color('features', scale=alt.Scale(scheme='warmgreys'), legend=None)
    )

    heads = alt.Chart(dfl2).mark_circle(size=80, opacity=0.5).encode(
        x=x,
        y=y,
        detail='features'
        # color=alt.Color('features', scale=alt.Scale(scheme='warmgreys'), legend=None)
    )

    text = heads.mark_text(dx=-25, dy=19, fontSize=12).encode(
        text='features'
    )

    scatter = alt.Chart(dfs).mark_point(strokeWidth=5, size=50).encode(
        x=alt.X(x, axis=alt.Axis(title=f'{x}  ({expl[x]:.1%})')),
        y=alt.Y(y, axis=alt.Axis(title=f'{y}  ({expl[y]:.1%})')),
        color=alt.Color(sc, scale=alt.Scale(scheme='turbo', type='log')),
        tooltip='Sample'
    )

    info = alt.Chart(dfs).mark_text(  # TODO: could use empty dataset "{'values':[{}]}" instead of "dfs" to print text only once?
        fontSize=12
    ).encode(
        text=alt.value([f'Explained: {expl.sum():.1%}', f'Values scaled: {normalise}']),
        x=alt.value(figsize[0] / 5),  # pixels from left
        y=alt.value(figsize[1] / 20),  # pixels from top
    )

    figure = alt.layer(lines, heads, text, scatter, info
                       ).resolve_scale(color='independent'
                                       ).interactive(
    ).properties(
        width=figsize[0], height=figsize[1]
    )

    # figure.save('../plots/biplot.html')

    return figure


def dist_hist(prop, particle_type, dist_name, x, pdf, r, plot_bins):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, pdf, 'r-', lw=5, alpha=0.6, label=f'{dist_name} pdf, {prop} of {particle_type}')
    ax.hist(r, bins=plot_bins, density=True, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.show()


def sns_contour_plot(data,x,y,hue,xlim=False,ylim=False,log=(False,False),figsize=(5,5)):
    """
    Create a contour plot using seaborn.

    Parameters
    ----------
    data : pandas.DataFrame containing the data
    x : column for x-axis (continuous data)
    y : column for y-axis (continuous data)
    hue : column for color (categorical data)
    xlim : (float, float), optional
    ylim : (float, float), optional
    log : boolean tuple, (xlog,ylog), optional, default: False
    figsize : (float, float), optional, default: 5, 5
    
    Returns
    -------
    seaborn contour + scatter plot with marginal histograms
    """

    p = sns.jointplot(
        x=x, y=y, data=data,
        hue=hue, kind='scatter',
        alpha=0.4, s=1
        ).plot_joint(
            sns.kdeplot,
            alpha=0.8,
            fill=False,
            bw_adjust=.85,
            thresh=0.05,
            levels=4,  # [0.0001, 0.25, 0.5, 0.75, 0.9999],
            log_scale=log,
            common_norm=False)
    if xlim:
        p.ax_joint.set_xlim(xlim)
    if ylim:
        p.ax_joint.set_ylim(ylim)
    if log[0]:
        p.ax_joint.set_xscale('log')
    if log[1]:
        p.ax_joint.set_yscale('log')
    p.fig.set_size_inches(figsize[0], figsize[1])

    p.savefig('../plots/sns_contour_plot.svg')

    return p


def plotly_contour_plot(df, x, y, color, nbins=100, ncontours=10, figsize=(800, 600)):
    """
    Create a contour plot of size vs. density.

    Parameters
    ----------
    df : pandas.DataFrame containing the data
    x_var : column of df to plot on the x-axis
    y_var : column of df to plot on the y-axis
    nbins : number of bins to use, default = 100
    ncontours : number of contours to use, default = 10
    figsize : (float, float), optional, default: 800, 600

    Returns
    -------
    plotly figure, as html and inline
    """
    
    fig = go.Figure()
    fig.add_trace(px.scatter(df,
    x=x,
    y=y,
    color=color,
    color_discrete_sequence=px.colors.qualitative.D3
    ))
    fig.add_trace(go.Histogram2dContour(
            x = df[x],
            y = df[y],
            colorscale = 'Blues',
            reversescale = False,
            xaxis = x,
            yaxis = 'y'
        ))
    fig.add_trace(go.Histogram(
            y = y,
            xaxis = 'x2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            )
        ))
    fig.add_trace(go.Histogram(
            x = x,
            yaxis = 'y2',
            marker = dict(
                color = 'rgba(0,0,0,1)'
            )
        ))

    fig.update_layout(
        autosize = False,
        xaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        yaxis = dict(
            zeroline = False,
            domain = [0,0.85],
            showgrid = False
        ),
        xaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        yaxis2 = dict(
            zeroline = False,
            domain = [0.85,1],
            showgrid = False
        ),
        height = 600,
        width = 600,
        bargap = 0,
        hovermode = 'closest',
        showlegend = False
    )

    #fig.update_yaxes(type="log", range=[0,4])  # log range: 10^0=1, 10^5=100000

    return fig


def size_kde_combined_samples_dist_plot(mp_sed_melt):
    """
    KDE plots of MP and Sediment for all stations combined
    (i.e. KDE calculated on separate samples,
    then averageing probabilities for particle occurrenc
    in the size bins for all samples)
    """
    
    mp_sed_size_dist_AllSamplesAveraged = mp_sed_melt.melt(
        id_vars=['Sample','lower','upper'],
        var_name='particle_type'
    ).groupby(
        ['particle_type', 'lower', 'upper']
    ).value.mean().reset_index()

    dists = alt.Chart(mp_sed_size_dist_AllSamplesAveraged).mark_area(
        #point=True,
        opacity=0.3
    ).encode(
        x=alt.X('lower', scale=alt.Scale(domain=[0.1, 1500], clamp=True, type='linear')),
        y='value',
        color=alt.Color('particle_type'),
        tooltip=['lower', 'upper', 'value', 'particle_type']
    )

    cumsum = dists.mark_line(strokeWidth=5, opacity=1).transform_window(
        frame=[None, 0],
        cumsum='sum(value)',
        groupby=['particle_type']
    ).encode(
        y='cumsum:Q',
        color=alt.Color('particle_type')
    )

    return alt.layer(dists, cumsum).resolve_scale(y='independent').properties(width=800, height=400)
    
