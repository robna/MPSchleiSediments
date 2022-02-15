import altair as alt
alt.data_transformers.disable_max_rows()
# alt.renderers.enable('altair_viewer')  # use to display altair charts externally in browser instead of inline (only activate in non-vega-compatible IDE like pycharm)
import pydeck as pdk
import streamlit as st
# from streamlit_vega_lite import altair_component


import prepare_data


def scatter_chart(df, x, y, c=False, equal_axes=False, title='', width=400, height=300):
    maxX = df[x].max()
    maxY = df[y].max()
    domain_max = max(maxX, maxY) * 1.05

    df = df.reset_index()
    scatter = alt.Chart(df).mark_point().encode(
        x=x,
        y=y,
        tooltip=['Sample', x, y])

    if c:
        scatter = scatter.encode(color=c)

    if equal_axes:
        scatter = scatter.encode(
            x=alt.X(x, scale=alt.Scale(domain=[0, domain_max])),
            y=alt.X(y, scale=alt.Scale(domain=[0, domain_max])))

    RegLine = scatter.transform_regression(
        x, y, method="linear",
    ).mark_line()

    RegParams = scatter.transform_regression(
        x, y, method="linear", params=True
    ).mark_text(align='left', lineBreak='\n').encode(
        x=alt.value(width / 4),  # pixels from left
        y=alt.value(height / 20),  # pixels from top
        text='params:N'
    ).transform_calculate(
        params='"R² = " + round(datum.rSquared * 100)/100 + \
        "      y = " + round(datum.coef[1] * 100)/100 + "x" + " + " + round(datum.coef[0] * 10)/10'
    )

    return alt.layer(scatter, RegLine, RegParams).properties(width=width, height=height).properties(title=title)


def poly_comp_chart(mp_pdd, mp_added_sed_sdd):
    poly_comp = prepare_data.aggregate_SDD(mp_pdd.groupby(['Sample', 'polymer_type']))
    poly_comp = poly_comp.merge(mp_added_sed_sdd[['Sample', 'Dist_WWTP', 'perc MUD']], on='Sample')
    selection = alt.selection_multi(fields=['polymer_type'], bind='legend')

    chart1 = alt.Chart(poly_comp.reset_index()).mark_bar().encode(
        x=alt.X('Dist_WWTP', scale=alt.Scale(type='linear')),
        # , sort = alt.SortField(field='Dist_WWTP', order='ascending')),
        y=alt.Y('Concentration', stack='normalize'),
        color=alt.Color('polymer_type', scale=alt.Scale(scheme='rainbow')),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=['Sample', 'polymer_type', 'Concentration', 'Dist_WWTP']
    ).add_selection(
        selection
    )

    chart2 = chart1.mark_bar().encode(
        x=alt.value(10),
        y=alt.Y('all_stations_summed:Q', stack='normalize'),
        tooltip=['polymer_type', 'all_stations_summed:Q']
    ).transform_aggregate(
        all_stations_summed='sum(Concentration)',
        groupby=['polymer_type']
    )

    chart = chart1 | chart2
    chart.save('poly_comp_chart.html')

    return chart  # | chart.encode(y=alt.Y('Concentration',stack='normalize'))


def station_map(data):
    data = data.loc[:,['Sample', 'GPS_LON', 'GPS_LAT']]
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.data_utils.compute_view(data[['GPS_LON','GPS_LAT']]),  # {"latitude": 54.5770,"longitude": 9.8124,"zoom": 11,"pitch": 50},
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
        "GridLayer", data, pickable=True, extruded=True, cell_size=200, elevation_scale=4, get_position=["GPS_LONs", "GPS_Lats"],
    )

    view_state = pdk.ViewState(latitude=54.5770, longitude=9.8124, zoom=11, bearing=0, pitch=45)

    # Render
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{position}\nCount: {count}"},)
    st.write(r)

