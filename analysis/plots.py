import altair as alt

alt.data_transformers.disable_max_rows()
# alt.renderers.enable('altair_viewer')  # use to display altair charts externally in browser instead of inline (only activate in non-vega-compatible IDE like pycharm)
import pydeck as pdk
import streamlit as st
# from streamlit_vega_lite import altair_component

from settings import Config
import prepare_data


def scatter_chart(df, x, y, c=False, equal_axes=False, title='', width=400, height=300):
    maxX = df[x].max()
    maxY = df[y].max()
    domain_max = max(maxX, maxY) * 1.05

    df = df.reset_index()
    base = alt.Chart(df).mark_point().encode(
        x=x,
        y=y,
        tooltip=['Sample', x, y])
    
    if equal_axes:
        base = base.encode(
            x=alt.X(x, scale=alt.Scale(domain=[0, domain_max])),
            y=alt.X(y, scale=alt.Scale(domain=[0, domain_max])))

    if c:
        scatter = base.encode(alt.Color(c, scale=alt.Scale(scheme='cividis')))  # TODO: only ordinal works with categorical and quantiative data and also shows regrassion, when not specifying data type it works fully for categorical data, but stops showing the regression when quantitative data is chosen for color.
    else: scatter = base

    RegLine = base.transform_regression(
        x, y, method="linear",
    ).mark_line()

    RegParams = base.transform_regression(
        x, y, method="linear", params=True
    ).mark_text(align='left', lineBreak='\n').encode(
        x=alt.value(width / 4),  # pixels from left
        y=alt.value(height / 20),  # pixels from top
        text='params:N'
    ).transform_calculate(
        params='"R² = " + round(datum.rSquared * 100)/100 + \
        "      y = " + round(datum.coef[1] * 100)/100 + "x" + " + " + round(datum.coef[0] * 10)/10'
    )

    chart = alt.layer(scatter, RegLine, RegParams).properties(width=width, height=height).properties(title=title)
    
    chart.save('../plots/scatter_chart.html')  # activate save chart to html file
    
    return chart


def poly_comp_chart(mp_pdd, mp_added_sed_sdd):
    poly_comp = prepare_data.aggregate_SDD(mp_pdd.groupby(['Sample', 'polymer_type']))
    poly_comp = poly_comp.merge(mp_added_sed_sdd[['Sample', 'Dist_WWTP', 'perc MUD']], on='Sample')
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
    
    chart_rel = chart_abs.encode(y=alt.Y('Concentration',stack='normalize'))

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
        get_position=["GPS_LONs", "GPS_Lats"],
    )

    view_state = pdk.ViewState(latitude=54.5770, longitude=9.8124, zoom=11, bearing=0, pitch=45)

    # Render
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{position}\nCount: {count}"}, )
    st.write(r)


def histograms(df):

    brush = alt.selection_interval(encodings=['x'])
    #
    # base = alt.Chart(df).properties(
    #     width=800,
    #     height=200
    # )
    #
    # bar = base.mark_bar().encode(
    #     y='count():Q'
    # )
    #
    # rule = bar.mark_rule(color='red', y='height').encode(  # TODO: rule not yet scaling with changing y-axis
    #     x=f'median({Config.size_dim}):Q',
    #     size=alt.value(5)
    # )
    #
    # chart = alt.vconcat(
    #     # bar.encode(
    #     #     alt.X(Config.size_dim,
    #     #           bin=alt.Bin(maxbins=30, extent=brush),
    #     #           scale=alt.Scale(domain=brush)
    #     #           ), tooltip='count():Q',
    #     # ), #+ rule,
    #     bar.encode(
    #         alt.X(Config.size_dim, bin=alt.Bin(maxbins=30)),
    #     ).add_selection(brush) + rule
    # )

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


def biplot(scor, load, expl, discr, x, y, sc, lc, ntf=5, normalise=False, figsize=(800,600)):  # TODO: normalisation not yet implemented
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

    dfl = load.head(ntf).append(load.head(ntf)-load.head(ntf)).reset_index()
    # dfl = (dfl / dfl.max(numeric_only=True).max(numeric_only=True))  # normalise values to range [-1,1]

    dfl2 = load.head(ntf).reset_index()
    # dfl2 = (dfl2 / dfl2.max().max())

    dfs = scor.reset_index().rename(columns={'index':'Sample'}).merge(discr[['Sample', 'Dist_WWTP', 'regio_sep']])
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
    
    text = heads.mark_text(dx=-25,dy=19, fontSize=12).encode(
        text='polymer_type'
    )

    scatter = alt.Chart(dfs).mark_point(strokeWidth=3).encode(
        x=x,
        y=y,
        color=alt.Color('regio_sep',scale=alt.Scale(scheme='dark2')),
        tooltip='Sample'
    )

    figure = alt.layer(lines, heads, text, scatter
    ).resolve_scale(color='independent'
    ).interactive(
    ).properties(
        width=figsize[0],height=figsize[1]
    )

    figure.save('../plots/biplot.html')

    return figure
    