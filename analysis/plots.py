import altair as alt
alt.data_transformers.disable_max_rows()
# alt.renderers.enable('altair_viewer')  # use to display altair charts externally in browser instead of inline (only activate in non-vega-compatible IDE like pycharm)

import seaborn as sns
sns.set_style('darkgrid')

from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
    
import pydeck as pdk
import streamlit as st
# from streamlit_vega_lite import altair_component

from settings import Config
import prepare_data


def scatter_chart(df, x, y, c=False, l=None, reg=None, equal_axes=False, title='', width=400, height=300):
    """
    Create a scatter plot with optional regression line and equation.
    :param df: dataframe with x and y columns
    :param x: x column name
    :param y: y column name
    :param c: color column name
    :param l: label column name (prints label on each point), None if no label (default)
    :param reg: None for no regression line (default), 'linear', 'log', 'exp' or 'pow'
    :param equal_axes: True for x and y axes ranging from 0 to their higher maximum (useful for predicted vs. observed plots) (default)
    :param title: plot title
    :param width: plot width
    :param height: plot height
    :return: altair chart
    """

    maxX = df[x].max()
    maxY = df[y].max()
    domain_max = max(maxX, maxY) * 1.05

    df = df.reset_index()
    base = alt.Chart(df).mark_point().encode(
        x=x,
        y=y,
        tooltip=[df.columns[1], x, y])

    if c:
        scatter = base.encode(alt.Color(c, scale=alt.Scale(
            scheme='cividis')))  # TODO: only ordinal works with categorical and quantiative data and also shows regrassion, when not specifying data type it works fully for categorical data, but stops showing the regression when quantitative data is chosen for color.
    else:
        scatter = base

    if l:
        scatter = scatter + scatter.mark_text(
            align='left',
            baseline='middle',
            dx=7
        ).encode(
            text=f'{l}:N'
        )

    if equal_axes:
        base = base.encode(
            x=alt.X(x, scale=alt.Scale(domain=[0, domain_max])),
            y=alt.X(y, scale=alt.Scale(domain=[0, domain_max])))

    if reg is not None:
        RegLine = base.transform_regression(
            x, y, method=reg,
        ).mark_line()

        R2_string = '"R² = " + round(datum.rSquared * 100)/100 + "      y = "'
        coef0_string = 'round(datum.coef[0] * 10)/10'
        coef1_string = 'round(datum.coef[1] * 10)/10'
        reg_eq = {'linear': f'{R2_string} + {coef1_string} + " * x + " + {coef0_string}',
                  'log': f'{R2_string} + {coef1_string} + " * log(x) + " + {coef0_string}',
                  'exp': f'{R2_string} + {coef0_string} + " * e ^ (" + {coef1_string} + " x)"',
                  'pow': f'{R2_string} + {coef0_string} + " * x^" + {coef1_string}'}

        RegParams = base.transform_regression(
            x, y, method=reg, params=True
        ).mark_text(align='left', lineBreak='\n', fontSize = 18).encode(
            x=alt.value(width / 4),  # pixels from left
            y=alt.value(height / 20),  # pixels from top
            text='params:N'
        ).transform_calculate(
            params=reg_eq[reg]
        )

        chart = alt.layer(scatter, RegLine, RegParams).properties(width=width, height=height).properties(title=title).configure_axis(labelFontSize = 18, titleFontSize = 28)
    else:
        chart = alt.layer(scatter).properties(width=width, height=height, title=title)

    chart.save('../plots/scatter_chart.html')  # activate save chart to html file

    return chart


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


def biplot(scor, load, expl, discr, x, y, sc, lc, ntf=5, normalise=False,
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
    lc : component to plot on the color scale for loadings
    ntf : number of top features to plot, default = 5
    normalise : boolean, whether normalise scores and loadings to [-1,1], default = False
    figsize : (float, float), optional, default: 800, 600

    Returns
    -------
    altair figure, as html and inline
    """

    dfl = load.head(ntf).append(load.head(ntf) - load.head(ntf)).reset_index()
    # dfl = (dfl / dfl.max(numeric_only=True).max(numeric_only=True))  # normalise values to range [-1,1]

    dfl2 = load.head(ntf).reset_index()
    # dfl2 = (dfl2 / dfl2.max().max())

    dfs = scor.reset_index().rename(columns={'index': 'Sample'}).merge(discr[['Sample', 'Dist_WWTP', 'regio_sep']])
    # dfs = (dfs / dfs.max(numeric_only=True).max(numeric_only=True))

    lines = alt.Chart(dfl).mark_line(opacity=0.3).encode(
        x=x,
        y=y,
        color=alt.Color('polymer_type', legend=None)
    )

    heads = alt.Chart(dfl2).mark_circle(size=50, opacity=0.5).encode(
        x=x,
        y=y,
        color=alt.Color('polymer_type', legend=None)
    )

    text = heads.mark_text(dx=-25, dy=19, fontSize=12).encode(
        text='polymer_type'
    )

    scatter = alt.Chart(dfs).mark_point(strokeWidth=3).encode(
        x=x,
        y=y,
        color=alt.Color('regio_sep', scale=alt.Scale(scheme='dark2')),
        tooltip='Sample'
    )

    figure = alt.layer(lines, heads, text, scatter
                       ).resolve_scale(color='independent'
                                       ).interactive(
    ).properties(
        width=figsize[0], height=figsize[1]
    )

    figure.save('../plots/biplot.html')

    return figure


## Create a function that uses plotly graphobjects Histogram2dContour to create a contour plot of size vs. density
def plotly_contour_plot(x,y, nbins=100, ncontours=10, figsize=(800, 600)):
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
    fig.add_trace(go.Histogram2dContour(
            x = x,
            y = y,
            colorscale = 'Blues',
            reversescale = False,
            xaxis = 'x',
            yaxis = 'y'
        ))
    fig.add_trace(go.Scatter(
            x = x,
            y = y,
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = dict(
                color = 'rgba(0,0,0,0.3)',
                size = 3
            )
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


def px_contour_plot(df,x,y):
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
    plotly figure
    """

    fig = px.density_contour(df, x=x, y=y, log_y=False)
    fig.update_traces(contours_coloring="fill", contours_showlabels = True)
    return fig


## Create a seaborn contour plot
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

    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.set(ylim=(10, 1000))

    # sns.kdeplot(x=x, y=y, log_scale=(xlog,ylog), cmap=cmap, fill=True, thresh=thresh, levels=5, alpha=0.8) #, norm=LogNorm())
    # return sns.scatterplot(x=x, y=y, s=10, color='black')

    # sns.kdeplot(ax=ax, data=data, x=x, y=y, hue=hue, log_scale=(xlog,ylog), fill=True, thresh=thresh, levels=5, alpha=0.8) #, norm=LogNorm())
    # p = sns.JointGrid(data=data, x=x, y=y, hue=hue)

    # p.plot(sns.scatterplot, sns.histplot)

    p = sns.jointplot(x=x, y=y, data=data, hue=hue, kind='scatter', alpha=0.4, s=2).plot_joint(sns.kdeplot, levels=5, alpha=0.8, thresh=0, log_scale=log)
    if xlim:
        p.ax_joint.set_xlim(xlim)
    if ylim:
        p.ax_joint.set_ylim(ylim)
    if log[0]:
        p.ax_joint.set_xscale('log')
    if log[1]:
        p.ax_joint.set_yscale('log')
    p.fig.set_size_inches(figsize[0], figsize[1])

    return p


def dist_hist(prop, particle_type, dist_name, x, pdf, r, plot_bins):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, pdf, 'r-', lw=5, alpha=0.6, label=f'{dist_name} pdf, {prop} of {particle_type}')
    ax.hist(r, bins=plot_bins, density=True, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.show()