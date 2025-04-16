from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rio
from shapely.geometry import Point, LineString
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from settings import Config, baw_tracer_reduction_factors
import interpol, geo_io
tqdm.pandas()

HERE = Path(__file__).resolve().parent          # analysis/ folder
ROOT = HERE.parent                              # project root
DATA_DIR = ROOT / "data"


def get_distance_to_shore(LON, LAT, polygon=None, epsg=Config.baw_epsg):
    """
    Calculates the distance to the shore for each sample station.
    :param LON: longitude
    :param LAT: latitude
    :param polygon: polygon to be used for the distance calculation
    :param epsg: epsg code of the polygon
    :return: numpy array of shortest distance of each point to polygon boundary
    """
    
    if polygon is None:
        polygon = geo_io.get_schlei()
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(LON, LAT))
    gdf.crs = 4326
    gdf.to_crs(f"EPSG:{epsg}", inplace=True)
    return gdf.geometry.apply(lambda x: polygon.boundary.distance(x)).to_numpy()


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
        tracks = geo_io.extract_zipped_traces(file)
    elif file.suffix == '.dat':
        tracks = geo_io.load_single_BAW_run(file)
    else:
        raise ValueError(f"File {file} has an unsupported file extension.")

    if polygon is None:
        polygon = geo_io.get_schlei()
    tracks = gpd.clip(tracks, polygon.buffer(10))  # keep only points in 'tracks' that are within the polygon (incl. a 10 m buffer to allow for minor deviations of the coastline between the polygon used here and the bouandaries of the tracer model)
    
    tracks.sort_index(inplace=True)
    tracks = tracer_sedimentation_points(tracks)  # determine tracer particle sedimentation events
    
    tracklines = tracer_points_to_lines(tracks)
    if save_tracks:
        tracks.to_file('../data/exports/geo/BAW_tracer_points.geojson', driver='GeoJSON')
        tracklines.to_file('../data/exports/geo/BAW_tracer_lines.geojson', driver='GeoJSON')
    return tracks, tracklines


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
        return pd.merge(sdd, pd.read_csv(DATA_DIR / f'{col_prefix + file_postfix}.csv', index_col=0), how='left', left_on='Sample', right_index=True)
    except:
        print('Need to calculate WWTP influence based on simulated particle tracks. This may take a while...')
        sdd = sdd.copy()
        if trackpoints is None:  # if gdf is provided use as is, other load and prepare
            if tracks_file is None:
                raise ValueError('Either tracks or tracks_file must be provided.')
            else:
                print('Loading trackpoints...')
                trackpoints, tracklines = get_BAW_traces(tracks_file)  # load from .dat file
        # turn sdd into a GeoDataFrame with conversion from Lat/Lon to to epsg of gdf
        print('Building gdf')
        sdd_gdf = gpd.GeoDataFrame(sdd, geometry=gpd.points_from_xy(sdd.LON, sdd.LAT), crs=4326).to_crs(trackpoints.crs)
        # approach 1: mean distance based influence factor
        print('Starting tracer_mean_dist...')
        sdd[col_prefix+'_as_tracer_mean_dist'] = sdd_gdf.geometry.progress_apply(lambda x: trackpoints.distance(x).mean())  # mean straight-line distance of all drifters at all time steps to each sampling station
        # approach 2: mean distance based influence factor, but only distances to each tracer end point
        print('Starting endpoints_mean_dist...')
        sdd[col_prefix+'_as_endpoints_mean_dist'] = sdd_gdf.geometry.progress_apply(lambda x: tracklines.apply(lambda y: Point(y.coords[-1])).distance(x).mean())
        # approach 3 and 4: buffer based influence factor
        print('Start making buffers...')
        sdd_gdf['geometry'] = sdd_gdf.buffer(buffer_radius)
        # spatially join the sample domain data with the particle tracks, where the buffer zone of a sample contains a particle location at any time step
        print('Start joining points and buffers...')
        dfsjoin = gpd.sjoin(sdd_gdf, trackpoints, how='left', predicate='contains')
        # group sdd by Sample and sum the number of particles that were encountered in the buffer zone
        print('Starting cumulated_residence...')
        sdd[col_prefix+'_as_cumulated_residence'] = dfsjoin.reset_index().groupby('index').time_step.count()  # summed occurrences of presence of all drifters at all time steps inside station buffer zones
        print('Starting mean_time_travelled...')
        sdd[col_prefix+'_as_mean_time_travelled'] = dfsjoin.reset_index().groupby(['index', 'simPartID']).nth(0).groupby('index').time_step.mean()  # mean of time steps of all drifters at first entrance to station buffer zones
        sdd[col_prefix+'_as_mean_time_travelled'].fillna(Config.tracer_mean_time_fillna, inplace=True)
        sdd.drop(columns=['geometry'], inplace=True)
        # save calculated WWTP influence factors (the last 3 columns of sdd) to a csv file
        sdd.set_index('Sample').iloc[:, -4:].to_csv(DATA_DIR / f'{col_prefix + file_postfix}.csv', index_label='Sample')
        return sdd


def make_coord_tuples(gdf):
    '''
    Create a list of coordinate tuples from a geopandas df,
    because rasterio.sample can't handle geopandas geometries
    '''
    return [(x,y) for x,y in zip(gdf['geometry'].x , gdf['geometry'].y)]


def get_depth(gdf, dem=None, label='water_depth'):
    '''
    Calculates water depth at points of gdf.
    `label` is used as column name for sampled
    depths and same df (with new column) is returned
    '''
    if dem is None:
        dem = geo_io.load_zipped_grid(Config.dem_path, Config.dem_filename)  # load grid from zip file
    coord_list = make_coord_tuples(gdf)
    gdf[label] = [-1 * x[0] for x in dem.sample(coord_list)]  # sample the grid at the locations of the tracer particles (multiply by -1 because the grid uses negative depth values) 
    return gdf


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
    tracks = get_depth(tracks)
    tracks['sediment_contact'] = False  # intitialize sediment_contact column: False = no contact
    tracks.loc[tracks['tracer_depth'] > tracks['water_depth'] - dist, 'sediment_contact'] = True  # set sediment_contact to True where the particle comes closer than 'dist' to 'water_depth'

    tracks['contact_id'] = (tracks.sediment_contact != tracks.sediment_contact.shift(1)).cumsum()  # create an intermediate column with a new unique id for each time the state of sediment_contact changes from False to True or vice versa
    
    contact_dur = tracks.loc[tracks.sediment_contact].groupby('contact_id').sediment_contact.count().rename('contact_dur')  # create a series with the duration of each contact
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


def tracer_points_to_lines(tracks):
    """
    Convert the points of the tracer particles to lines
    """
    return tracks.groupby(['season', 'tracer_ESD', 'simPartID'])['geometry'].apply(lambda x: LineString(x.unique().tolist()*2 if len(x.unique().tolist())==1 else x.unique().tolist()))


def make_grid(poly, res, round_grid_coords=False):
    '''
    Generate two arrays, x-coordinates (H x W) and y-coordinates (H x W) from the total bounds of a polygon (single row geopandas df).
    '''
    poly_bounds = poly.total_bounds  # get the extent of the Schlei polygon
    print('polygon bounds: ', [b for b in zip(['xmin', 'ymin', 'xmax', 'ymax'], poly_bounds)])
    if round_grid_coords:
        xmin, ymin = np.floor(poly_bounds[:2] / res) * res  # rounding down to next multiple of res        
        xmax, ymax = np.ceil(poly_bounds[2:] / res) * res  # rounding up to next multiple of res
    else:
        xmin, ymin, xmax, ymax = poly_bounds
    xgrid, ygrid = np.meshgrid(np.arange(xmin, xmax, res),
                            np.arange(ymax, ymin, -res),  # y is upside down so higher values end up at the top (north) of the grid
                            )
    return xgrid, ygrid, xmin, ymin, xmax, ymax


def grid_clip(values, poly, xgrid, ygrid):
    '''
    Clips raster layers (ndarray) with a polygon, by converting into geodataframe and using its clip method.
    The returned raster array has the original values in areas that are within the polygon and np.nan for areas outside.
    '''
    grid_gdf = gpd.GeoDataFrame({'vals': values.ravel()}, 
                                geometry=gpd.points_from_xy(xgrid.ravel(), ygrid.ravel()),
                                crs=Config.baw_epsg,
                                )
    clipper = grid_gdf.clip(poly)  # takes about 0.5 min
    ## old method:
    # clipper = gpd.overlay(grid_gdf, poly, how='intersection')  # takes about 15 min
    # clipper = clipper.loc[grid_gdf.intersects(poly.geometry[0])]  # takes about 11 min
    grid_gdf.loc[~grid_gdf.index.isin(clipper.index), 'vals'] = np.nan
    return grid_gdf['vals'].values.reshape(values.shape)


def interclip(data, name, xgrid, ygrid, poly, tool, clip=True, plot='nochange'):
    '''
    Interpolate data from point geo-dataframe to a regular grid
    and clip the grid to a shape given in a second geo-dataframe,
    cotaining one polygon.
    It is just a wrapper to run grid_interp and grid_clip in one go.
    '''
    x = data.geometry.x
    y = data.geometry.y
    z = data[name]
    Config.interpolation_methods[tool]['var_name'] = name
    if plot != 'nochange':
        Config.interpolation_methods[tool]['plot'] = plot
    ipolator = getattr(interpol, tool)
    values = ipolator(x, y, z, xgrid, ygrid, params=Config.interpolation_methods[tool])
    if not clip:
        return values
    return grid_clip(values, poly, xgrid, ygrid)


def sample_array_raster(raster, xmin, ymax, res, points_gdf, **kwargs):
    """
    Sample a raster (ndarray) at points from a geopandas df.
    Adapted from here: https://gis.stackexchange.com/a/387772
    Args:
        raster (numpy.ndarray): raster to be masked with dim: [H, W]
        xmin, ymax, res: coordinates of upper left corner of raster array,
                         and resolution in raster units.
        points_gdf, **kwargs: passed to rasterio.mask.mask

    Returns:
        list of values sampled from raster at points
    """
    transform = rio.transform.from_origin(xmin, ymax, res, res)
    coord_list = make_coord_tuples(points_gdf)
    with rio.io.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            dataset.write(raster, 1)
        with memfile.open() as dataset:
            output = [val[0] for val in dataset.sample(coord_list, **kwargs)]
    return output


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

