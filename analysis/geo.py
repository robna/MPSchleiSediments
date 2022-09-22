from settings import Config
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import LineString




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


def get_BAW_traces(epsg=25832):
    df = pd.read_fwf('../data/insel_Bahnen.dat', names=['simPartID', 'X', 'Y', 'depth'])
    df.loc[df.simPartID == 'Part', 'simPartID'] = df.loc[df.simPartID == 'Part', 'X']
    df.simPartID.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    # add a new column to df called 'time_step' which starts from one for each simPartID and increases by one for each row
    df['time_step'] = df.groupby('simPartID').cumcount() + 1

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=f"EPSG:{epsg}").drop(columns=['X', 'Y'])
    return gdf


def get_wwtp_influence(sdd, gdf=None, buffer_radius=Config.station_buffers, col_name='WWTP_influence'):
    """
    Calculates the sum of simulated particles that were encountered in a sample's buffer zone (irrespective of the time step).
    :param sdd: df with sample domain data
    :param gdf: GeoDataFrame containing the particle tracks as points geometries per time step
    :return: sdd df with cummulated residence time in the buffer zone
    """
    sdd = sdd.copy()
    if gdf is None:
        gdf = get_BAW_traces()
    # turn sdd into a GeoDataFrame with conversion from Lat/Lon to to epsg of gdf
    sdd_gdf = gpd.GeoDataFrame(sdd, geometry=gpd.points_from_xy(sdd.LON, sdd.LAT), crs=4326).to_crs(gdf.crs)
    sdd[col_name+'_as_tracer_mean_dist'] = sdd_gdf.geometry.apply(lambda x: gdf.distance(x).mean())  # mean distance of all particles at all time steps to each sample station
    # create a buffer around each sample station
    sdd_gdf['geometry'] = sdd_gdf.buffer(buffer_radius)
    # spatially join the sample domain data with the particle tracks, where the buffer zone of a sample contains a particle location at any time step
    dfsjoin = gpd.sjoin(sdd_gdf, gdf, how='left', predicate='contains')
    # group sdd by Sample and sum the number of particles that were encountered in the buffer zone
    sdd[col_name+'_as_cumulated_residence'] = dfsjoin.reset_index().groupby('index').time_step.count()  # Robins approach
    sdd[col_name+'_as_mean_time_travelled'] = dfsjoin.reset_index().groupby(['index', 'simPartID']).nth(0).groupby('index').time_step.mean()  # Kristinas approach
    sdd.drop(columns=['geometry'], inplace=True)
    return sdd

##TODO: the following two functions are not complete yet
# def build_paths(start, polygon, lines=None):
#     """
#     Generates a line between start and each polygon vertex if it does not intersect with the polygon.
#     :param start: start point
#     :param polygon: polygon action as a boundary
#     :return: list of lines between start and each polygon vertex
#     """
#     if lines is None:
#         lines = []
#     for vertex in polygon.exterior.coords:
#         line = LineString([start, vertex])
#         if not line.intersects(polygon):
#             lines.append(line)
#     return lines


# def dist_within(start, dest, polygon):
#     """
#     Calculates the distance between two points within a polygon using the Shapely library.
#     :param start: start point
#     :param dest: destination point
#     :param polygon: polygon action as a boundary
#     :return: distance between start and dest within polygon
#     """
#     # check if start and dest are within the polygon, and raise an error if not
#     if not polygon.contains(start) or not polygon.contains(dest):
#         raise ValueError('start and dest must be within the polygon')
#     # create a line between start and dest
#     line = LineString([start, dest])
#     # check if line lies completely within the polygon, and if so: return its length
#     if polygon.contains(line):
#         return line.length
#     # generate a line between start and each polygon vertex if it does not intersect with the polygon
#     lines = build_paths(start, dest, polygon)
#     # while lines have not reached dest, extend them to the next polygon vertex


def tracer_sedimentation_points(gdf, dem=None, dist=Config.sed_contact_dist, dur=Config.sed_contact_dur):
    """
    Calculates the sedimentation points of the tracer particles.
    :param gdf: GeoDataFrame containing the particle tracks as points geometries per time step
    :param dem: Digital Elevation Model
    :param dist, dur: when a particle gets closer to the sediment than dist for longer than dur, it is considered to have sedimented
    :return: same GeoDataFrame with a new column 'sediment_contacts' starting at 0 for each particle and increasing by 1 for each contact to sediment
    """
    if dem is None:
        dem = rasterio.open('../data/DGM_Schlei_1982-2002_5m-Raster_UTM32.xyz')
    # TODO: this function is not complete yet
    return pd.DataFrame()
