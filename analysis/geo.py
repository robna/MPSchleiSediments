import geopandas as gpd




def get_distance_to_shore(LON, LAT, polygon=None, epsg=32632):
    """
    Calculates the distance to the shore for each sample station.
    :param LON: longitude
    :param LAT: latitude
    :param polygon: polygon to be used for the distance calculation
    :param epsg: epsg code of the polygon
    """
    
    if polygon is None:
        # read ../data/SchleiCoastline_from_OSM.geojson into a GeoDataFrame
        polygon = gpd.read_file('../data/SchleiCoastline_from_OSM.geojson').to_crs(f"EPSG:{epsg}")
    
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(LON, LAT))
    gdf.crs = 4326
    gdf.to_crs(f"EPSG:{epsg}", inplace=True)

    # gdf.apply(lambda x: x.geometry.distance(schlei_coast.boundary), axis=1)
    # schlei_coast.boundary.distance(gdf)

    d = gdf.geometry.apply(lambda x: polygon.boundary.distance(x))

    return d