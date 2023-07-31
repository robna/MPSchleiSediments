from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.transform import from_origin
from settings import Config


def get_schlei(epsg=Config.baw_epsg):
    """
    Loads the Schlei polygon from a shapefile
    :return: GeoDataFrame with the Schlei polygon
    """    
    return gpd.read_file('../data/SchleiCoastline_from_OSM.geojson').to_crs(f"EPSG:{epsg}")


def load_zipped_grid(path2zip, path2grid=None):
    """
    Loads a grid file from inside a .zip file
    :param path2zip: path to .zip file
    :param path2grid: path to grid file inside the .zip file (if None, it tries to load a non-hidden grd-file with the same name as the .zip file)
    :return: grid as a numpy array
    """

    if not isinstance(path2zip, Path):
        path2zip = Path(path2zip)
    rasta = rio.open(f'zip://{path2zip}!{path2zip.stem.lstrip(".") + ".grd" if path2grid is None else path2grid}', 'r+')
    rasta.crs = rio.crs.CRS.from_epsg(Config.baw_epsg)
    return rasta


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


def grid_save(a, f, left_upper, res=None, tags=None):
    '''
    Save a grid from a numpy 2D array to a tif file.
    :param a: array to be saved
    :param f: file name
    :param left_upper: list-like of upper left coordinate (xmin, ymax)
    :param res: resolution of the grid (pixel size)
    '''
    if not res:
        res = Config.interpolation_resolution
    transform = from_origin(left_upper[0], left_upper[1], res, res)
    new_dataset = rio.open(f, 'w', driver='GTiff',
                                height = a.shape[0], width = a.shape[1],
                                count=1, dtype=str(a.dtype),
                                crs=Config.baw_epsg,
                                transform=transform)
    new_dataset.write(a, 1)
    if tags:
        new_dataset.update_tags(**tags)  # Update tags if provided
    new_dataset.close()


def grid_load(src_filename):
    '''
    Load a saved grid from a tiff file into ndarray.
    '''
    with rio.open(src_filename, "r") as src:
        print('\n\nLoading raster with Tiff tags:')
        for k,v in src.meta.items():
            print(f'{k}: {v}')
        print(src.bounds, '\n\n')
        grid = src.read(1)
    grid[grid==src.nodata] = np.nan
    return grid


def zero_padding(array, xx, yy):
    """
    Add padding to a numpy array to make it fit specific dimensions.
    Adapted from here: https://stackoverflow.com/a/59241336
    :param array: numpy array
    :param xx: desired height
    :param yy: desired width
    :param kwargs: kwargs passed to np.pad
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)))


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
