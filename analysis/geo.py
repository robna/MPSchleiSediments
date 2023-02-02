from settings import Config, baw_tracer_reduction_factors
from pathlib import Path
from zipfile import ZipFile
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from shapely.geometry import Point, LineString
from joblib import Parallel, delayed, cpu_count

def get_schlei(epsg=Config.baw_epsg):
    """
    Loads the Schlei polygon from a shapefile
    :return: GeoDataFrame with the Schlei polygon
    """
    
    return gpd.read_file('../data/SchleiCoastline_from_OSM.geojson').to_crs(f"EPSG:{epsg}")


def load_zipped_grid(path2zip, path2grid=None, epsg=Config.baw_epsg):
    """
    Loads a grid from a .grd file inside a .zip file
    :param path2zip: path to .zip file
    :param path2grid: path to .grd file inside the .zip file (if None, it tries to load a .grd file with the same name as the .zip file)
    :return: grid as a numpy array
    """

    if not isinstance(path2zip, Path):
        path2zip = Path(path2zip)
    rasta = rasterio.open(f'zip://{path2zip}!{path2zip.stem.lstrip(".") + ".grd" if path2grid is None else path2grid}', 'r+')
    rasta.crs = rasterio.crs.CRS.from_epsg(epsg)
    return rasta


def get_distance_to_shore(LON, LAT, polygon=None, epsg=Config.baw_epsg):
    """
    Calculates the distance to the shore for each sample station.
    :param LON: longitude
    :param LAT: latitude
    :param polygon: polygon to be used for the distance calculation
    :param epsg: epsg code of the polygon
    """
    
    if polygon is None:
        polygon = get_schlei()
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(LON, LAT))
    gdf.crs = 4326
    gdf.to_crs(f"EPSG:{epsg}", inplace=True)
    return gdf.geometry.apply(lambda x: polygon.boundary.distance(x))


def get_BAW_traces(file, polygon=None, save_tracks=True):
    """
    Utility function to load either a single .dat file
    or a .zip file containing multiple .dat files
    :param file: path to .dat or .zip file
    :param cached: path to cached .geojson file, to long run loading from .dat or .zip files again
    :param polygon: polygon to be used crop points outside the relevant area
    :param save_tracks: whether to save the tracks as a .geojson file
    :return: GeoDataFrame with the tracer particle simulation run(s) as points per time step
    """

    file = Path(file)
    if file.suffix == '.geojson':
        return gpd.read_file(file)
    elif file.suffix == '.zip':
        tracks = extract_zipped_traces(file)
    elif file.suffix == '.dat':
        tracks = load_single_BAW_run(file)
    else:
        raise ValueError(f"File {file} has an unsupported file extension.")

    if polygon is None:
        polygon = get_schlei()
    tracks = gpd.clip(tracks, polygon.buffer(10))  # keep only points in 'tracks' that are within the polygon (incl. a 10 m buffer to allow for minor deviations of the coastline between the polygon used here and the bouandaries of the tracer model)
    
    tracks.sort_index(inplace=True)
    tracks = tracer_sedimentation_points(tracks)  # determine tracer particle sedimentation events
    
    tracklines = tracer_points_to_lines(tracks)
    if save_tracks:
        tracks.to_file('../data/exports/BAW_tracer_points.geojson', driver='GeoJSON')
        tracklines.to_file('../data/exports/BAW_tracer_lines.geojson', driver='GeoJSON')
    return tracks, tracklines


def load_single_BAW_run(file, epsg=Config.baw_epsg):
    """
    Loads a single BAW tracer particle simulation run from a .dat file
    :param file: path to .dat file
    :param epsg: epsg code of the projection
    :return: GeoDataFrame with the tracer particle simulation run
    """

    # read and wrangle file into pandas df
    if isinstance(file, str):
        file = Path(file)
    df = pd.read_fwf(file, names=['X', 'Y', 'tracer_depth', 'simPartID'], widths=[15,15,15,6])
    df['season'] = file.name.split('_')[0]
    df['tracer_ESD'] = int(file.name.split('_')[1].strip('.dat'))
    df = df.iloc[:(df.X=='EGRUPPE').argmax()]  # find the first row which has "EGRUPPE" in its index and drop it and all rows below
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['simPartID'] = df['simPartID'].astype(int)
    
    # add a new column to df called 'time_step' which starts from one for each simPartID and increases by one for each row
    df['time_step'] = df.groupby('simPartID').cumcount() + 1
    
    # correct unrealistic tracer depths
    if Config.restrict_tracers_to_depth:
        df.loc[(df.tracer_depth > Config.restrict_tracers_to_depth) |  # find all rows where tracer_depth is larger than Config.restrict_tracers_to_depth
            (df.tracer_depth < 0 ), 'tracer_depth'] = np.nan  # find all rows where tracer_depth is smaller than zero ...and set them to NaN
        df.tracer_depth.interpolate(method='linear', inplace=True)  # interpolate the NaN values from their neighbours

    # create a GeoDataFrame from df
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=f"EPSG:{epsg}").drop(columns=['X', 'Y'])


def extract_zipped_traces(path2zip):
    with ZipFile(path2zip) as zipObj:
        # get a list of all .dat files in the zip
        datfiles = [f for f in zipObj.namelist() if f.endswith('.dat')]
        # only keep .dat files in the list which are from seasons named in Config.use_seasons
        datfiles = [f for f in datfiles if f.split('_')[0] in Config.use_seasons]
        # read the .dat files using geo.get_BAW_traces and concatenate them into one dataframe
        gdfs = [load_single_BAW_run(zipObj.open(f)) for f in datfiles]  # normal way, using one cpu core
        # gdfs = Parallel(n_jobs=cpu_count(), verbose=2, backend='multiprocessing')(delayed(load_single_BAW_run)(zipObj.open(f)) for f in datfiles)  # attempt to parallelize, but returns error: cannot pickle '_io.BufferedReader' object
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)


def get_wwtp_influence(sdd, trackpoints=None, tracks_file=None, buffer_radius=Config.station_buffers, col_prefix='WWTP_influence', file_postfix=''):
    """
    Calculates different versions of an influence factor of the WWTP based on simulated tracers.
    :param sdd: df with sample domain data
    :param tracks: GeoDataFrame containing the particle tracks as points geometries per time step
    :param tracks_file: path to a .dat file (or zip of .dats) containing the particle tracks
    :param buffer_radius: radius of the buffer around the WWTP
    :param col_prefix: column name prefix for the different versions of the influence factor
    :return: sdd df with added columns for WWTP inluence
    """

    try:  # because this calculation takes a while, we want to be able to save the tracks to a file and load them later
        return pd.merge(sdd, pd.read_csv(f'../data/{col_prefix + file_postfix}.csv', index_col=0), how='left', left_on='Sample', right_index=True)
    except:
        print('Need to calculate WWTP influence based on simulated particle tracks. This may take a while...')
        sdd = sdd.copy()
        if trackpoints is None:  # if gdf is provided use as is, other load and prepare
            if tracks_file is None:
                raise ValueError('Either tracks or tracks_file must be provided.')
            else:
                trackpoints, tracklines = get_BAW_traces(tracks_file)  # load from .dat file
        # turn sdd into a GeoDataFrame with conversion from Lat/Lon to to epsg of gdf
        sdd_gdf = gpd.GeoDataFrame(sdd, geometry=gpd.points_from_xy(sdd.LON, sdd.LAT), crs=4326).to_crs(trackpoints.crs)
        # approach 1: mean distance based influence factor
        sdd[col_prefix+'_as_tracer_mean_dist'] = sdd_gdf.geometry.apply(lambda x: trackpoints.distance(x).mean())  # mean straight-line distance of all drifters at all time steps to each sampling station
        # approach 2: mean distance based influence factor, but only distances to each tracer end point
        sdd[col_prefix+'_as_endpoints_mean_dist'] = sdd_gdf.geometry.apply(lambda x: tracklines.apply(lambda y: Point(y.coords[-1])).distance(x).mean())
        # approach 3 and 4: buffer based influence factor
        sdd_gdf['geometry'] = sdd_gdf.buffer(buffer_radius)
        # spatially join the sample domain data with the particle tracks, where the buffer zone of a sample contains a particle location at any time step
        dfsjoin = gpd.sjoin(sdd_gdf, trackpoints, how='left', predicate='contains')
        # group sdd by Sample and sum the number of particles that were encountered in the buffer zone
        sdd[col_prefix+'_as_cumulated_residence'] = dfsjoin.reset_index().groupby('index').time_step.count()  # summed occurrences of presence of all drifters at all time steps inside station buffer zones
        sdd[col_prefix+'_as_mean_time_travelled'] = dfsjoin.reset_index().groupby(['index', 'simPartID']).nth(0).groupby('index').time_step.mean()  # mean of time steps of all drifters at first entrance to station buffer zones
        sdd[col_prefix+'_as_mean_time_travelled'].fillna(Config.tracer_mean_time_fillna, inplace=True)
        sdd.drop(columns=['geometry'], inplace=True)
        # save calculated WWTP influence factors (the last 3 columns of sdd) to a csv file
        sdd.set_index('Sample').iloc[:, -4:].to_csv(f'../data/{col_prefix + file_postfix}.csv', index_label='Sample')
        return sdd


def tracer_sedimentation_points(tracks, dem=None, dist=Config.sed_contact_dist, dur=Config.sed_contact_dur):
    """
    Calculates the sedimentation points of the tracer particles.
    :param tracks: GeoDataFrame containing the particle tracks as points geometries per time step
    :param dem: Digital Elevation Model
    :param dist: distance a tracer particle needs to get under to make a valid sediment contact
    :param dur: duration in time steps a tracer particle needs to be within 'dist' from sediment to make a valid sediment contact
    :return: same GeoDataFrame with a column for 'water_depth',
             a new column called 'sediment_contact' being True where a particle comes closer than 'dist' to 'water_depth',
             and another column called 'contact_count'starting at 0 for each particle and increasing by 1 for each contact to sediment which is at least 'dur' timesteps long
    """
    if dem is None:
        dem = load_zipped_grid('../data/.DGM_Schlei_1982_bis_2002_UTM32.zip')  # load grid from zip file
    coord_list = [(x,y) for x,y in zip(tracks['geometry'].x , tracks['geometry'].y)]  # create list of coordinate tuples, because rasterio.sample can't handle geopandas geometries
    tracks['water_depth'] = [-1 * x[0] for x in dem.sample(coord_list)]  # sample the grid at the locations of the tracer particles (multiply by -1 because the grid uses negative depth values)  
    tracks['sediment_contact'] = False  # intitialize sediment_contact column: False = no contact
    tracks.loc[tracks['tracer_depth'] > tracks['water_depth'] - dist, 'sediment_contact'] = True  # set sediment_contact to True where the particle comes closer than 'dist' to 'water_depth'

    tracks['contact_id'] = (tracks.sediment_contact != tracks.sediment_contact.shift(1)).cumsum()  # create an intermediate column with a new unique id for each time the state of sediment_contact changes from False to True or vice versa
    
    contact_dur = tracks.loc[tracks.sediment_contact].groupby(['contact_id']).sediment_contact.count().rename('contact_dur')  # create a series with the duration of each contact
    contact_dur[contact_dur < dur] = np.nan  # exclude all contacts which are shorter than 'dur'
    tracks = tracks.join(contact_dur, on=['contact_id'])  # join the series of the duration of each contact to the gdf
    tracks = pd.concat(Parallel(n_jobs=cpu_count())(delayed(arrest_tracks)(group, group_name) for group_name, group in tracks.groupby(['simPartID', 'season', 'tracer_ESD'])))
    tracks.sort_index(inplace=True)
    # tracks.drop(columns=['contact_id', 'contact_dur'], inplace=True)  # drop the intermediate columns    
    return tracks


def arrest_tracks(group, group_name):
    """
    Per tracer particle, loop through the rows of the gdf and count
    how many times up to each row a particle has been in valid contact to sediment.
    After a valid sedimentation set coordinates of all following time steps of
    that particle to what they were at the time of sedimentation.
    """
    simPartID, tracer_ESD = group_name[0], group_name[2]
    if baw_tracer_reduction_factors[tracer_ESD] == 0:  # if baw_tracer_reduction_factor is 0, return nothing, i.e. no particles of this type will be included
        return
    reducer = round(1 / baw_tracer_reduction_factors[tracer_ESD])
    if (simPartID + reducer - 1) % reducer == 0:
        completed_row_contact_dur = np.nan
        counter = 0
        for row_index, row in group.iterrows():
            if ~np.isnan(row.contact_dur) and np.isnan(completed_row_contact_dur):  # for row where contact_dur is not nan but completed_row_contact_dur was nan (i.e. at every time step where a new sedimentation starts), increase counter
                counter += 1
            completed_row_contact_dur = row.contact_dur  # save what was in this rows contact_dur for comparison in next iteration
            group.loc[row_index, 'contact_count'] = counter
        if group.contact_count.max() >= Config.arrest_on_nth_sedimentation > 0:
            last_valid_coords = group.loc[group.contact_count == Config.arrest_on_nth_sedimentation, 'geometry'].values[0]
            last_valid_x, last_valid_y = last_valid_coords.x, last_valid_coords.y
            group.loc[group.contact_count >= Config.arrest_on_nth_sedimentation, 'geometry'] = Point(last_valid_x, last_valid_y)
        return group


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


def tracer_points_to_lines(tracks):
    """
    Convert the points of the tracer particles to lines
    """
    return tracks.groupby(['season', 'tracer_ESD', 'simPartID'])['geometry'].apply(lambda x: LineString(x.unique().tolist()*2 if len(x.unique().tolist())==1 else x.unique().tolist()))


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

