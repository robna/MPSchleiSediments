from settings import Config
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
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


def get_BAW_traces(file, epsg=25832):
    # first, find the line number of the first occurence of the string 'EGRUPPE' in the file (lines from there on will not be read)
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if 'EGRUPPE' in line:
                break
    num_lines = sum(1 for _ in open(file))  # count number of lines in file
    footer_length = num_lines - i  # number of lines in the footer
    
    # read and wrangle file into pandas df
    df = pd.read_fwf(file, names=['X', 'Y', 'tracer_depth', 'simPartID'], skipfooter=footer_length)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    
    # add a new column to df called 'time_step' which starts from one for each simPartID and increases by one for each row
    df['time_step'] = df.groupby('simPartID').cumcount() + 1
    
    # correct unrealistic tracer depths
    # df.loc[(df.tracer_depth > Config.max_depth_allowed) |
    #        (df.tracer_depth < 0 ), 'tracer_depth'] = np.nan
    # df.tracer_depth.interpolate(method='linear', inplace=True)

    # create a GeoDataFrame from df
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=f"EPSG:{epsg}").drop(columns=['X', 'Y'])
    return gdf


def get_wwtp_influence(sdd, gdf=None, buffer_radius=Config.station_buffers, col_name='WWTP_influence'):
    """
    Calculates different versions of an influence factor of the WWTP based on simulated tracers.
    :param sdd: df with sample domain data
    :param gdf: GeoDataFrame containing the particle tracks as points geometries per time step
    :return: sdd df with added columns for WWTP inluence
    """
    sdd = sdd.copy()
    if gdf is None:  # if gdf is provided use as is, other load and prepare
        gdf = get_BAW_traces()  # load from .dat file
        gdf = tracer_sedimentation_points(gdf)  # determine tracer particle sedimentation events
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


def sparse_xyz_to_grid(xyz, sep=' ', resolution=Config.dem_resolution, nodata=-9999):  # FIXME: not working yet, gives: TypeError: 'set' type is unordered, in line df_missing = ...
    """
    Converts a sparse regularly spaced xyz dataset into a dense grid. Adapted from https://www.gpxz.io/blog/fixing-sparse-xyz-files
    :param xyz: path to .xyz file, no header, projected CRS
    :param sep: separator between columns in xyz file
    :param resolution: resolution of the grid in meters
    :param nodata: value to use for cells not covered by the sparse dataset
    :return: regular grid as a numpy array
    """
    
    df = pd.read_csv(xyz, sep=sep, header=None, names=['x', 'y', 'z'])
    # Figure out which x and y values are needed.
    x_vals = set(np.arange(df.x.min(), df.x.max() + resolution, resolution))
    y_vals = set(np.arange(df.y.min(), df.y.max() + resolution, resolution))
    # For each x value, find any missing y values, and add a NODATA row.
    dfs = [df]
    for x in x_vals:
        y_vals_missing = y_vals - set(df[df.x == x].y)
        if y_vals_missing:
            df_missing = pd.DataFrame({'x': x, 'y': y_vals_missing, 'z': nodata})
            dfs.append(df_missing)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['y', 'x'])
    # Check.
    assert len(df) == len(x_vals) * len(y_vals)

    return df.to_numpy()


def load_zipped_grid(path2zip, path2grid=None, epsg=25832):
    """
    Loads a grid from a .grd file inside a .zip file
    :param path2zip: path to .zip file
    :param path2grid: path to .grd file inside the .zip file (if None, it tries to load a .grd file with the same name as the .zip file)
    :return: grid as a numpy array
    """

    if not isinstance(path2zip, Path):
        path2zip = Path(path2zip)
    rasta = rasterio.open(f'zip://{path2zip}!{path2zip.stem + ".grd" if path2grid is None else path2grid}', 'r+')
    rasta.crs = rasterio.crs.CRS.from_epsg(epsg)
    return rasta


def tracer_sedimentation_points(gdf, dem=None, dist=Config.sed_contact_dist, dur=Config.sed_contact_dur):
    """
    Calculates the sedimentation points of the tracer particles.
    :param gdf: GeoDataFrame containing the particle tracks as points geometries per time step
    :param dem: Digital Elevation Model
    :param dist: distance a tracer particle needs to get under to make a valid sediment contact
    :param dur: duration in time steps a tracer particle needs to be within 'dist' from sediment to make a valid sediment contact
    :return: same GeoDataFrame with a column for 'water_depth',
             a new column called 'sediment_contact' being True where a particle comes closer than 'dist' to 'water_depth',
             and another column called 'contact_count'starting at 0 for each particle and increasing by 1 for each contact to sediment which is at least 'dur' timesteps long
    """
    if dem is None:
        dem = load_zipped_grid('../data/DGM_Schlei_1982_bis_2002_UTM32.zip')  # load grid from zip file
    coord_list = [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]  # create list of coordinate tuples, because rasterio.sample can't handle geopandas geometries
    gdf['water_depth'] = [x[0] for x in dem.sample(coord_list)]  # sample the grid at the locations of the tracer particles
    gdf['sediment_contact'] = False  # intitialize sediment_contact column: False = no contact
    gdf.loc[gdf['tracer_depth'].abs() > gdf['water_depth'].abs()-dist, 'sediment_contact'] = True  # set sediment_contact to True where the particle comes closer than 'dist' to 'water_depth'

    gdf['contact_id'] = (gdf.sediment_contact != gdf.sediment_contact.shift(1)).cumsum()  # create an intermediate column with a new unique id for each time the state of sediment_contact changes from False to True or vice versa
    
    contact_dur = gdf.loc[gdf.sediment_contact].groupby(['contact_id']).sediment_contact.count().rename('contact_dur')  # create a series with the duration of each contact
    contact_dur[contact_dur < dur] = np.nan  # exclude all contacts which are shorter than 'dur'
    gdf = gdf.join(contact_dur, on=['contact_id'])  # join the series of the duration of each contact to the gdf
    
    for group_name, group in gdf.groupby('simPartID'):  # per tracer particle, loop through the rows of the gdf and count how many times up to each row a particle has been is valid contact to sediment
        completed_row_contact_dur = np.nan
        counter = 0
        for row_index, row in group.iterrows():
            if ~np.isnan(row.contact_dur) and np.isnan(completed_row_contact_dur):  # for row where contact_dur is not nan but completed_row_contact_dur was nan (i.e. at every time step where a new sedimentation starts), increase counter
                counter += 1
            gdf.loc[row_index, 'contact_count'] = counter
            completed_row_contact_dur = row.contact_dur  # save what was in this rows contact_dur for comparison in next iteration
    gdf.drop(columns=['contact_id', 'contact_dur'], inplace=True)  # drop the intermediate columns

    if Config.truncate_on_nth_sedimentation > 0:
        gdf = gdf.loc[gdf.contact_count < Config.truncate_on_nth_sedimentation]
    
    return gdf
