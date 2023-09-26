import warnings
warnings.filterwarnings('ignore')  # ignore warnings to avoid flooding the gridsearch output with repetitive messages (works for single cpu)

import numpy as np
import pandas as pd
import xarray as xr

import altair as alt
alt.data_transformers.disable_max_rows()
# alt.renderers.enable('altair_viewer')  # use to display altair charts externally in browser instead of inline (only activate in non-vega-compatible IDE like pycharm)
import altair_transform

import seaborn as sns
sns.set_style('whitegrid')

from matplotlib import pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
    
import pydeck as pdk
import streamlit as st
from sklearn.metrics import r2_score

from settings import Config, target
import prepare_data
import geo_io

df_0 = pd.DataFrame({'x': [0], 'y': [0]})


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
        for groupname, group in df.groupby(color):
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
    linref=False,
    linref_slope=1,
    linref_intercept=0,
    mix_lines=False,
    xtransform=False, ytransform=False, xscale='linear', yscale='linear',
    title='', width=400, height=300,
    incl_params=True):

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
    :param linref: True for showing a linear reference line (default=False)
    :param linref_slope: Slope of the linear reference line (default=1)
    :param linref_intercept: Intercept of the linear reference line (default=0)
    :param mix_lines: True for showing conservative mixing lines (default=False)
    :param xtransform: when True take np.log10 of x-values before plotting, be carful when also using non-linear axis scales (default=False)
    :param ytransform: when True take np.log10 of y-values before plotting, be carful when also using non-linear axis scales (default=False)
    :param xscale: scale to use on x axis, str, any of
                 ['linear', 'log', 'pow', 'sqrt', 'symlog', 'identity', 'sequential', 'time', 'utc', 'quantile', 'quantize', 'threshold', 'bin-ordinal', 'ordinal', 'point', 'band']
    :param yscale: scale to use on y axis, str, see xscale for options
    :param title: plot title
    :param width: plot width
    :param height: plot height
    :param incl_params: if True (default) return a tuple (chart, params_df), otherwise return chart only
    :return: altair chart and df of regression parameter (None if reg=None), or chart only if incl_params=False
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
            x=alt.X(x, scale=alt.Scale(domain=[0, domain_max]), axis= alt.Axis(title= f'log10 of {x}') if xtransform else alt.Axis(title = x)),
            y=alt.Y(y, scale=alt.Scale(domain=[0, domain_max]), axis= alt.Axis(title= f'log10 of {y}') if ytransform else alt.Axis(title = y)))

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
            strokeDash=[10,20],
            strokeWidth=0.6,
            clip=True
        ).encode(
            x=alt.X(x, scale=alt.Scale(domain=[minX, maxX])),
            y=alt.Y(x, scale=alt.Scale(domain=[minY, maxY]))
        )

        identityR2 = r2_score(df[x], df[y])
        
        identityR2Text = alt.Chart(df_0).mark_text(
            fontSize=12,
            align="left", baseline="top"
        ).encode(
            x=alt.value(width / 1.8),  # pixels from left
            y=alt.value(height / 100),  # pixels from top
            text=alt.value(f'R² to identity = {identityR2:.2f}')
        )
        
        chart = alt.layer(chart, identityLine, identityR2Text)


    if linref:
        linrefLine = base.mark_line(
            color= 'black',
            strokeDash=[3,3],
            strokeWidth=0.6,
            clip=True
        ).encode(
            x=alt.X(x, scale=alt.Scale(domain=[minX, maxX])),
            y=alt.Y('xx:Q', scale=alt.Scale(domain=[minY, maxY]))
        ).transform_calculate(
            xx=f'{linref_slope} * datum.{x} + {linref_intercept}'
        )

        linrefR2 = r2_score(linref_slope * df[x] + linref_intercept, df[y])
        
        linrefR2Text = alt.Chart(df_0).mark_text(
            fontSize=12,
            align="left", baseline="top"
        ).encode(
            x=alt.value(width / 1.8),  # pixels from left
            y=alt.value(height / 100 * 5),  # pixels from top
            text=alt.value(f'R² to linear reference = {linrefR2:.2f}')
        )
        
        chart = alt.layer(chart, linrefLine, linrefR2Text)

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

    # chart.save('../data/exports/plots/scatter_chart.html')  # activate save chart to html file
    if not incl_params:
        return chart
    return chart, params


def poly_comp_chart(poly_comp, color='polymer_type'):
    selection = alt.selection_multi(fields=[color], bind='legend')

    chart_abs = alt.Chart(poly_comp.reset_index()).mark_bar().encode(
        x=alt.X('Dist_WWTP', scale=alt.Scale(type='linear')),
        # , sort = alt.SortField(field='Dist_WWTP', order='ascending')),
        y=alt.Y('Concentration', stack=True),
        color=alt.Color(color, scale=alt.Scale(scheme='rainbow')),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.3)),
        tooltip=['Sample', color, 'Concentration', 'Dist_WWTP']
    ).add_selection(
        selection
    )

    chart_rel = chart_abs.encode(y=alt.Y('Concentration', stack='normalize'))

    chart_tot = chart_rel.mark_bar(size=160).encode(
        x=alt.value(10),
        y=alt.Y('total_share:Q', stack='normalize', axis=alt.Axis(format='.0%')),
        tooltip=alt.Tooltip([color, 'total_share:Q'],),# format='.0%'),  # TODO: how to make a tooltip with several entries of different formats, i.e. polymer type as string and total_share as '.0%'?
    ).transform_joinaggregate(
        total='sum(Concentration)',
    ).transform_joinaggregate(
        all_stations_summed='sum(Concentration)',
        groupby=[color]
    ).transform_calculate(
        share="datum.all_stations_summed / datum.total"
    ).transform_aggregate(
        total_share='mean(share)',
        groupby=[color]
    ).properties(width=80)

    chart = alt.hconcat(chart_abs.interactive(), chart_rel, chart_tot.interactive()).resolve_scale('independent')
    chart.save('../data/exports/plots/poly_comp_chart.html')  # activate save chart to html file

    return chart  # | chart.encode(y=alt.Y('Concentration',stack='normalize'))


def poly_comp_pie(com2):
    """
    Pie chart of polymer composition, transferred from analysis notebook for use in app.
    Selection handled by streamlit widgets!
    """
    base = alt.Chart(com2).encode(
        theta=alt.Theta('Concentration', stack=True),
        color=alt.Color('ranked_polymer_type:N'),
        tooltip=['polymer_type', 'ShareOfTotal:N']
    ).transform_window(
        rank='row_number()',
        sort=[alt.SortField("Concentration", order="descending")],
    ).transform_calculate(
        ranked_polymer_type="datum.rank < 6 ? datum.polymer_type : ''"
    ).transform_joinaggregate(
        total='sum(Concentration)',
    ).transform_calculate(
        ShareOfTotal="datum.Concentration / datum.total"
    )

    pie = base.mark_arc(outerRadius=100, innerRadius=30, padAngle=0.01, cornerRadius=4)
    text = base.mark_text(radius=120, size=16).encode(text=alt.Text("ShareOfTotal:N", format='.0%'))
    
    return alt.layer(
        pie, text
    )


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


def histograms(df, title=''):
    brush = alt.selection_interval(encodings=['x'])
    base = alt.Chart(df).properties(width=1000, height=200)

    bar = base.mark_bar().encode(
        x=alt.X(Config.size_dim, bin=alt.Bin(maxbins=50)),
        y='count():Q',
        tooltip='count():Q'
    )

    cum_size = base.mark_line(interpolate='step-after', color='black').encode(
        x=Config.size_dim,
        y='cumulative_count:Q',
        tooltip=[Config.size_dim, 'cumulative_count:Q']
    ).transform_window(
        cum="count()",
        sort=[{"field": Config.size_dim}]
    ).transform_calculate(
        cumulative_count=f'datum.cum / {len(df)}'
    )

    rule = base.mark_rule(color='red').encode(
        x=f'median({Config.size_dim}):Q',
        tooltip=alt.Tooltip([f'median({Config.size_dim}):Q'], format=".2f"),
        size=alt.value(3)
    )

    chart = alt.vconcat(
        # bar.encode(alt.X(Config.size_dim,
        #                  bin=alt.Bin(maxbins=50, extent=brush),
        #                  scale=alt.Scale(domain=brush)
        #                  ),
        alt.layer(bar.add_selection(brush), cum_size, rule
    ).resolve_axis(y='independent').resolve_scale(y='independent')
    )

    return chart.properties(title=title).configure_title(fontSize=20, offset=5, orient='top', anchor='middle')


def biplot(scor, load, expl, discr, x, y, sc, ntf=5, normalise=None,
           figsize=(800, 600)):
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

    dfl = pd.concat([load.head(ntf), load.head(ntf) - load.head(ntf)]).reset_index()
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

    info = alt.Chart(df_0).mark_text(
        fontSize=12
    ).encode(
        text=alt.value([f'Explained: {expl.loc[[x,y]].sum():.1%}', f'Values scaled: {normalise}']),
        x=alt.value(figsize[0] / 5),  # pixels from left
        y=alt.value(figsize[1] / 20),  # pixels from top
    )

    figure = alt.layer(lines, heads, text, scatter, info
                       ).resolve_scale(color='independent'
                                       ).interactive(
    ).properties(
        width=figsize[0], height=figsize[1]
    )

    # figure.save('../data/exports/plots/biplot.html')

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

    p.savefig('../data/exports/plots/sns_contour_plot.svg')

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


def size_kde_combined_samples_dist_plot(mp_sed_melt, title=''):
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

    # medians = cumsum.mark_rule(strokeWidth=3).encode(
    #     y='median(value):Q',
    #     # y=alt.value(0),
    #     # y2=''
    # )

    return alt.layer(dists, cumsum).resolve_scale(y='independent').properties(width=800, height=400, title=title)#.configure_title(fontSize=20, offset=5, orient='top', anchor='middle')
    

def repNCV_score_plots(scored_multi, return_df=False, ncv_mode=Config.ncv_mode, width=400, height=300):
    '''
    Takes score df of multi-repetition simulations,
    re-arranges it to suitable long-form and plots
    a chart of score evoloution against number of
    repetitions, facetted for scorer type and model run.
    '''
    df = scored_multi.unstack().reset_index().melt(id_vars=['NCV_repetitions', 'run_with'], var_name=['Scorer', 'Aggregation']).dropna()
    df.Aggregation = df.Aggregation.str.split('_\d', expand=True)[0].str.strip('_of').str.rstrip('s').str.lower()

    df = pd.concat([df, df.Aggregation.str.split('_of_', expand=True).rename(columns={0: 'rep_aggregator', 1: 'fold_aggregator'})], axis=1).drop(columns='Aggregation')
    idx = df.loc[df.fold_aggregator == 'all'].index
    df.fold_aggregator.loc[idx] = df.rep_aggregator.loc[idx]
    df.rep_aggregator.loc[idx] = 'none'

    dfs = df.loc[df.rep_aggregator == 'stdev'].rename(columns={'value': 'Score_stdev'})
    dfs.drop(columns='rep_aggregator', inplace=True)
    
    df = df.loc[df.rep_aggregator != 'stdev']
    df.sort_values(['run_with', 'NCV_repetitions', 'rep_aggregator', 'fold_aggregator', 'Scorer'], ascending=[True,True,False,False,False], inplace=True)
    df = df[['run_with', 'NCV_repetitions', 'rep_aggregator', 'fold_aggregator', 'Scorer', 'value']].rename(columns={'value': 'Score_value'})
    df = df.merge(dfs, how='left')
    df.Score_stdev.loc[df.rep_aggregator == 'none'] = np.nan
    
    sel_rep_aggregator = alt.selection_point(
        fields=['rep_aggregator'],
        bind=alt.binding_radio(options=np.append(df.rep_aggregator.unique(), None), labels=np.append(df.rep_aggregator.unique(), 'all')),
        name='Select',
        # clear=False,
        value='mean',
    )
    toggle_stdev = alt.param(
        bind=alt.binding_checkbox(name='Show standard deviations as areas? ')
    )
    
    base = alt.Chart(df).encode(
        x='NCV_repetitions',
        color = alt.Color('fold_aggregator', sort=['median', 'iqm', 'mean']),
    ).properties(width=width, height=height,
    )
    
    score = base.mark_line(
        point=True
    ).encode(
        y = alt.Y('Score_value', title=None),
        strokeDash = alt.StrokeDash('rep_aggregator', sort=['median', 'iqm', 'mean']),
        tooltip = ['NCV_repetitions', 'rep_aggregator', 'fold_aggregator', 'Score_value', 'Score_stdev'],
    ).transform_filter(
        sel_rep_aggregator
    )
    
    stdev = base.mark_area(opacity=0.2).encode(
        y = 'lower:Q',
        y2 = 'upper:Q',
    ).transform_calculate(
        lower = 'datum.Score_value - 0.5 * datum.Score_stdev',
        upper = 'datum.Score_value + 0.5 * datum.Score_stdev',
    ).transform_filter(
        sel_rep_aggregator,
    ).transform_filter(
        toggle_stdev,
    )
    
    cols = (stdev + score).facet(
        column = alt.Column(
            'Scorer',
            sort=[k for k in Config.scorers.keys()],  # ['R2', 'MedAPE', 'MAPE', 'MedAE', 'MAE'],
            title=[f"Modelled response variable: {target}" \
                 "   ---   " \
                 "Evolution of scores with increasing number of shuffle-repeated NCV runs" \
                 "   ---   " \
                 "Best model candidates in grid search CV were determined by the best " \
                 f"{Config.select_best} {Config.refit_scorer} scores on inner folds' test sets"]
            )
    ).resolve_scale(
        y='independent'
    ).interactive(
    )    
    chart = alt.hconcat()
    for model_class in df.run_with.unique():
        text = alt.Chart(df_0).mark_text(
                angle=270,
                lineBreak=', ',
                fontSize=16,
                align="center", baseline="middle"
            ).encode(
                x=alt.value(0),  # pixels from left
                y=alt.value(height / 2),  # pixels from top
                text=alt.value(model_class)
            )
        chart &= (text | cols.transform_filter(alt.FieldEqualPredicate(field='run_with', equal=model_class)))
    chart = chart.configure_view(
        strokeWidth=0
    ).add_params(
        sel_rep_aggregator,
        toggle_stdev,
    )
    return (chart, df) if return_df else chart


def ensemble_pred_histograms(members_df, pred_df, truth):
    df = pd.concat([
        members_df.iloc[:, 1:], pred_df
        ],
        axis=1).melt(
        id_vars='regressor',
        value_vars=pred_df.columns,
        var_name='Sample',
        value_name=f'{target}_predicted'
    )
    df2 = pd.DataFrame(truth).rename(columns={target: f'{target}_observed'}).reset_index()

    input_dropdown = alt.binding_select(options=df.Sample.unique(), name='Sample ')
    selection = alt.selection_point(fields=['Sample'], bind=input_dropdown)

    histx = f'{target}_predicted'
    hist = alt.Chart(df).mark_bar().encode(
        # alt.X(f'{target}_predicted').bin(maxbins=30),
        alt.X(histx).bin(step=100, maxbins=20), #.scale(domain=[f'min({histx}):Q', f'max({histx}):Q']),
        y='count()',
        color='regressor',
    ).add_params(
        selection
    ).transform_filter(
        selection
    )

    rule_of_truth = alt.Chart(df2).mark_rule(color='red').encode(
        x=f'{target}_observed',
        opacity=alt.value(0.4),
        tooltip=['Sample', f'{target}_observed'],
        size=alt.value(3)
    ).transform_filter(
        selection
    )

    return alt.layer(
        hist, rule_of_truth
    ).properties(
        width=800
    ).interactive(
    )


def ncv_pie(df, cols_select = ['regressor', 'test_set_samples', 'features'], show_top=4):
    '''
    Makes pie charts of how often certain elements are occuring in an NCV ensemble model
    :param cols_select: columns in NCV df to make a plot of (default: ['regressor', 'features'])
    :param show_top: number of groups to show in the plots: eg. show_top = 4  --> only the top 4 groups are shown, the rest is summarised as 'Other'.
    '''
    df = df.copy()
    # df = df.loc[df.index.get_level_values('run_with').str.startswith('#'), cols_select]
    members = len(df)
    reps = len(df.index.get_level_values('NCV_repetition').unique())
    df = df.loc[:, cols_select]
    df.rename(columns={'features': 'feature_sets'}, inplace=True)
    feature_frame = pd.DataFrame([[item] for sublist in df.feature_sets.to_list() for item in sublist], columns=['features'])
    testset_frame = pd.DataFrame([[item] for sublist in df.test_set_samples.to_list() for item in sublist], columns=['test_set_samples'])

    if 'regressor__booster' in df.columns:
        df.regressor[~df.regressor__booster.isna()] = df.regressor[~df.regressor__booster.isna()] + ' ' + df.regressor__booster[~df.regressor__booster.isna()]

    var_list = cols_select + ['feature_sets']
    var_list = [{v: f"datum.rank <= {show_top} ? datum.{v} : 'Other'"} for v in var_list]  # turn into list of dicts with var_name: some-string-for-altairs_windowtransform

    pies = []
    for var in var_list:
        var_name = next(iter(var))
        source = feature_frame if var_name == 'features' else testset_frame if var_name == 'test_set_samples' else df
        base = alt.Chart(source.reset_index()).encode(
            alt.Theta('counted:Q').aggregate("sum").stack(True),
            alt.Color(f'{var_name}:N', sort=alt.SortField('rank:O')),
            alt.Order('rank:O').aggregate("sum"),
            tooltip=[f'{var_name}:N', 
                     alt.Tooltip('sum(counted):Q', title='count'),
                     alt.Tooltip('sum(rank):O', title='rank'),
                    ],
        ).transform_aggregate(
            counted='count()',
            groupby=[var_name],
        ).transform_calculate(
            rel_counted=f"datum.counted / {source.shape[0]}",
        ).transform_window(
            rank='row_number()',
            sort=[alt.SortField('counted', order="descending")],
        ).transform_calculate(
            **var,
        )
        pie = base.mark_arc(outerRadius=100)
        text = base.mark_text(radius=140, size=20).encode(
            alt.Text("rel_counted:Q", format='.0%').aggregate("sum")
        )
        pies.append(pie+text)

    plot = alt.vconcat(*pies    
    ).resolve_scale(
        color='independent',
    ).configure_legend(
        orient='top',
        labelLimit=0,
    ).properties(
        title={
      "text": [f'Using ensemble with {members} members: {int(members/reps)} member(s) per repetition.'], 
      "subtitle": [
          f'There are {len(df.regressor.unique())} different model classes: {df.regressor.unique()}',
          f'There are {len(df.feature_sets.apply(lambda x: tuple(x)).unique())} different feature sets, composed of {len(feature_frame.features.unique())} different features.',
                  ]
        },
    )
    return plot, df


def model_pred_bars(df, target='Concentration', domain=None):
    r_df = df.copy()
    r_df[f'{target}_predicted'] = (r_df[f'{target}_predicted'] - r_df[f'{target}_observed']) / r_df[f'{target}_observed']
    r_df[f'{target}_interpolated'] = (r_df[f'{target}_interpolated'] - r_df[f'{target}_observed']) / r_df[f'{target}_observed']

    return alt.Chart(r_df.loc[df.Type=='observed']).mark_bar(clip=True).encode(
        x=alt.X('key:N', sort=None, title=None),
        y=alt.Y('value:Q', scale=alt.Scale(domain=[-domain, domain] if domain else [-1, 5]), axis=alt.Axis(format='.0%'), title=None),
        color=alt.Color('key:N', sort=None),
        column='Sample',
        tooltip=['value:Q'],
    ).transform_fold(
        [f'{target}_predicted', f'{target}_interpolated']
    ).properties(title='Deviation of predicted and interpolated values from obsereved.'
    # ).resolve_scale(y='independent'
    ).interactive(
    )


def per_err_agg_bar(pred_agg_df):
    agg_dev_df = pd.DataFrame(  # caluclates the deviations of the aggregated predictions relative to the observed values
        np.array([
            (pred_agg_df[s] - pred_agg_df[target])/pred_agg_df[target] * 100 # relative difference in percent
            for s in Config.aggregators.keys()
        ]).T,
        columns=Config.aggregators.keys(),
        index=pred_agg_df.index)
    agg_dev_df[target] = pred_agg_df[target]

    return agg_dev_df.reset_index().melt(id_vars=['Sample', target]).sort_values(by=target).plot.barh(
        x='Sample', xlabel='',
        y='value', ylabel='Error [%]',
        by='variable',
        title=f'Percentage error of predictions per aggregation\nSamples are sorted by descending observed {target}',
        width=600, height=1000,
    )


def gridplot(a, x, y, poly=None, rasterize=False, project=False, tiles=False):
    '''
    Plot grid data in a numpy 2D array as an hvplot.
    :param a: numpy array with data values
    :param x: Either array of same size as a, holding x coordinate values, or 1D array of length a.shape[1]
    :param y: Either array of same size as a, holding y coordinate values, or 1D array of length a.shape[0]
    :param poly: Plot a polygon outline ontop of the grid. Provided as a one-row geopandas df of polygon or multipolygon geometry. If not provided, default Schlei coastline will be loaded. Set to False for no outline.
    :param rasterize: If True, use datashader for faster loading (default=False)
    :param project: Whether to project the data before plotting (adds initial overhead but avoids projecting data when plot is dynamically updated) (default=False)
    :param tiles: True to show OSM background for map, default False
    :return: hvplot
    '''
    if poly is True or poly is None:
        poly = geo_io.get_schlei()

    if len(x.shape) == 2:  # if x is supplied as an xgrid
        x= x[0,:]  # take first row as x-coodinates vector
    if len(y.shape) == 2:  # if x is supplied as an xgrid
        y= y[:,0]  # take first column as y-coodinates vector
    
    data_array = xr.DataArray(a, coords={'x': x, 'y': y}, dims=('y', 'x'))
    grid_plot = data_array.hvplot(
        x='x', y='y',
        rasterize=rasterize,
        cmap='spectral_r',
        cnorm='log',#'eq_hist',
        tiles=tiles,
        project=project,
        crs=Config.baw_epsg,
        frame_width=700, frame_height=500,  # set width and height to 1/5th of the grid dimensions, but not smaller than 400x300
        )
    if poly is not False:
        grid_plot = grid_plot * poly.hvplot(
            fill_alpha=0,
            line_color='red',
            crs=Config.baw_epsg,
            project=project,
            )
    return grid_plot


def variogram_plot(cov_model, bin_center, gamma):
    ax = cov_model.plot(x_max=max(bin_center))
    ax.scatter(bin_center, gamma)