from ipywidgets import widgets
from IPython.display import display
import numpy as np
import numba as nb
import pandas as pd
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import gstools as gs
import pykrige
import skgstat
import pygmt
from settings import Config
from geo_io import grid_load, zero_padding


def scipy_griddata(x, y, z, xgrid, ygrid, params):
    '''
    Interpolates point data from geopandas geoseries, using scipy.interpolat.griddata,
    to a numpy 2D-array of regularly spaced grid points (ndarray).
    Optional fillna: As some interpolation methods ('linear' and 'cubic') will result
    in nan outside of the convex hull of data points, these can be filled be
    re-interpolating them using 'nearest' method.
    '''

    points = np.vstack((x, y)).T
    values = griddata(
        points, z,
        (xgrid, ygrid),
        method=params['method'],  # 'linear' and 'cubic' will result in nan outside of the convex hull of data points
    )
    nan_mask = np.isnan(values)  # if there are any nan points re-interpolate them using method 'nearest'

    if np.any(nan_mask):
        values2 = griddata(
            points, z,
            (xgrid, ygrid), method='nearest',
        )
        values[nan_mask] = values2[nan_mask]
    return values


def numpy_simple_idw(x, y, z, xgrid, ygrid, params):
    '''
    Note: adapted from https://stackoverflow.com/a/3114117
    '''
    xi, yi = xgrid.ravel(), ygrid.ravel()
    dist = distance_matrix(
        np.array([list(c) for c in zip(x, y)]),
        np.array([list(c) for c in zip(xi, yi)])
        )
    weights = 1.0 / np.power(dist, params['power'])  # in IDW, weights are 1 / distance raised to the power
    weights /= weights.sum(axis=0)  # make weights sum to one
    zi = np.dot(weights.T, z)  # multiply the weights for each interpolated point by all observed Z-values
    return zi.reshape(xgrid.shape)


def numpy_rbf_idw(x, y, z, xgrid, ygrid, params):
    '''
    Note: adapted from https://stackoverflow.com/a/3114117
    '''
    xi, yi = xgrid.ravel(), ygrid.ravel()
    dist = distance_matrix(
        np.array([list(c) for c in zip(x, y)]),
        np.array([list(c) for c in zip(xi, yi)])
        )
    internal_dist = distance_matrix(
        np.array([list(c) for c in zip(x, y)]),  # Mutual pariwise distances between observations
        np.array([list(c) for c in zip(x, y)])
        )
    weights = np.linalg.solve(internal_dist, z)  # solve for the weights such that mistfit at the observations is minimized
    zi =  np.dot(dist.T, weights)  # Multiply the weights for each interpolated point by the distances
    return zi.reshape(xgrid.shape)


def scipy_rbf_idw(x, y, z, xgrid, ygrid, params):
    '''
    Note: adapted from https://stackoverflow.com/a/3114117
    :param x: x coordinates of point measurements
    :param y: y coordinates of point measurements
    :param z: values of point measurements
    :param xgrid: array of x-coordinates with shape of the final grid
    :param ygrid: array of y-coordinates with shape of the final grid
    :param params: dict or list or str of parameters specific to the interpolation algorithm
    '''
    interp = Rbf(x, y, z, function=params['function'])
    return interp(xgrid.ravel(), ygrid.ravel()).reshape(xgrid.shape)


def pykrige_ordkrig(x, y, z, xgrid, ygrid, params):
    '''
    Perform ordinary kriging using PyKrige module.
    :param x: x coordinates of point measurements
    :param y: y coordinates of point measurements
    :param z: values of point measurements
    :param xgrid: array of x-coordinates with shape of the final grid
    :param ygrid: array of y-coordinates with shape of the final grid
    :param params: dict or list or str of parameters specific to the interpolation algorithm
    
    params:
        
        nlags: The lag distance would be the maximum distance / nlag.
        
        weight: bool, optional,
            Flag that specifies if semivariance at smaller lags should be weighted
            more heavily when automatically calculating variogram model.
            The routine is currently hard-coded such that the weights are
            calculated from a logistic function, so weights at small lags are ~1
            and weights at the longest lags are ~0; the center of the logistic
            weighting is hard-coded to be at 70% of the distance from the shortest
            lag to the largest lag. Setting this parameter to True indicates that
            weights will be applied. Default is False. (Kitanidis suggests that the
            values at smaller lags are more important in fitting a variogram model,
            so the option is provided to enable such weighting.)
        
        model:
            variogram model function type (str or GSTools CovModel, default='linear')
            in-built models: 'linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect'
        
        variogram_params: list or dict, optional
            Parameters that define the specified variogram model. If not provided,
            parameters will be automatically calculated using a "soft" L1 norm
            minimization scheme. For variogram model parameters provided in a dict,
            the required dict keys vary according to the specified variogram
            model:
                # linear
                    {'slope': slope, 'nugget': nugget}
                # power
                    {'scale': scale, 'exponent': exponent, 'nugget': nugget}
                # gaussian, spherical, exponential and hole-effect:
                    {'sill': s, 'range': r, 'nugget': n}
                # OR
                    {'psill': p, 'range': r, 'nugget': n}
            Note that either the full sill or the partial sill
            (psill = sill - nugget) can be specified in the dict.
            
            OR FOR USE WITH gstools:
            # gstools  --> use gstools to fit a variogram to the emprical data
                {'h': <lag_distance>, 'max_dist': <maximum distance considered>}
    '''
    if params['model'] == 'gstools':  # TODO: using gstools to fit the variogram in combination with ipywidgets to interactively adjust the parameters, doesn't work yet...
        model_types = [
            'Gaussian',
            'Exponential',
            'Matern',
            'Integral',
            'Stable',
            'Rational',
            'Cubic',
            'Linear',
            'Circular',
            'Spherical',
            'HyperSpherical',
            'SuperSpherical',
            'JBessel',
            ]
        # Create interactive widgets for argument adjustment
        h_wid = widgets.BoundedIntText(value=params['h'], min=0, max=params['max_dist'], description='Lag distance (h):', disabled=False)
        max_wid = widgets.BoundedIntText(value=params['max_dist'], min=0, description='Max distance:', disabled=False)
        model_wid = widgets.RadioButtons(options=model_types, value=model_types[0], layout={'width': 'max-content'}, description='Variogram model:', disabled=False)
        # run_button = widgets.Button(description='Go Kriging Go!')
        
        display(h_wid, max_wid, model_wid)#, run_button)
        bin_center, gamma = gstools_variogram(x, y, z, h_wid.value, max_wid.value)
        
        nugget_wid = widgets.IntSlider(value=params['nugget'] if 'nugget' in params.keys() else 0, min=0, max=30000)  # also create a widget to change the nugget (up to max gamma)
        display (nugget_wid)

        tvm = getattr(gs, model_wid.value)
        cov_model = tvm(dim=2, nugget=nugget_wid.value)
        cov_model.fit_variogram(bin_center, gamma)
        print(type(cov_model))
        ax = cov_model.plot(x_max=max(bin_center))
        ax.scatter(bin_center, gamma)

        # def on_run_button_clicked(button):
        grid , vargrid = pykrige.ok.OrdinaryKriging(x, y, z, cov_model).execute("grid", xgrid[0, :], ygrid[:, 0])

        # run_button.on_click(on_run_button_clicked)
        return grid

    else:
        OK = pykrige.ok.OrdinaryKriging(
            x, y, z,
            nlags=params['nlags'],
            weight=params['weight'],
            variogram_model=params['model'],
            variogram_parameters= params['variogram_params'],
            exact_values=params['exact'] if 'exact' in params.keys() else True,
            enable_plotting=params['plot'] if 'plot' in params.keys() else True,
            enable_statistics=params['stats'] if 'stats' in params.keys() else True,
            coordinates_type='euclidean',
            verbose=True,
        )
        grid, vargrid = OK.execute("grid", xgrid[0, :], ygrid[:, 0])  # Creates the kriged grid and the variance grid
        return grid


def skgstat_ordkrig(x, y, z, xgrid, ygrid, params):
    xy= np.array([x, y]).T
    V = skgstat.Variogram(xy, z, 
                  maxlag=params['maxlag'],
                  n_lags=params['n_lags'],
                  model=params['model'],
                  normalize=params['normalize'] if 'normalize' in params.keys() else False,
                  use_nugget=params['use_nugget'] if 'use_nugget' in params.keys() else True
                  )
    
    if params['plot']:
        # print(f'Calculating {params["var_name"]}...)
        V.plot()
    ok = skgstat.OrdinaryKriging(V, min_points=params['min_points'], max_points=params['max_points'],
                         mode=params['mode'] if 'mode' in params.keys() else 'exact')
    grid = ok.transform(xgrid.ravel(), ygrid.ravel()).reshape(xgrid.shape)
    return grid


def pygmt_xyz2grd(x, y, z, xgrid, ygrid, params):  # TODO: pygmt interpolation not yet working
    res = Config.interpolation_resolution
    xmin, xmax, ymin, ymax = xgrid[0,0], xgrid[0,-1], ygrid[-1,0], xgrid[-1,-1], 
    surf = pygmt.surface(x, y, z,
                         region=[xmin, xmax, ymin, ymax],
                         spacing=res)

    ## https://www.pygmt.org/latest/api/generated/pygmt.xyz2grd.html
    data = gpd.GeoDataFrame({'values': z, 'geometry': (x,y)})
    data.crs = Config.baw_epsg
    grid = pygmt.xyz2grd(data=data,
                         region=[xmin, xmax, ymin, ymax],  # f'{xmin}/{xmax}/{ymin}/{ymax}+uM',
                         spacing=(res, res),
                         projection='U32N',  # f'EPSG:{Config.baw_epsg}',
                         verbose=None,  # t for timing, True for more output
                        )
    return grid


def load_external_interpol(x, y, z, xgrid, ygrid, params):
    grid = grid_load(params['path'] + params[params['var_name']])
    grid = zero_padding(grid, *xgrid.shape)
    return grid


@nb.njit((nb.float64[:, :], nb.float64[:, :]), parallel=True)
def distance_matrix(coords_a, coords_b):
    '''
    Calculates a distance matrix of two point arrays.
    Note: from https://stackoverflow.com/a/68598094 and https://stackoverflow.com/a/69333877
    :param coords_a: 1st array (shape: N, 2) of point coordinates: first col = x, second col = y
    :param coords_b: 2nd array (shape: M, 2) of point coordinates: first col = x, second col = y
    :return: array of distances (shape, N, M)
    '''
    res = np.empty((coords_a.shape[0], coords_b.shape[0]), dtype=coords_a.dtype)
    for i in nb.prange(coords_a.shape[0]):
        for j in range(coords_b.shape[0]):
            res[i, j] = np.sqrt((coords_a[i, 0] - coords_b[j, 0])**2 + (coords_a[i, 1] - coords_b[j, 1])**2)
    return res


def gstools_variogram(x,y,z,h,max_dist):
    '''
    '''
    bin_no = int(np.ceil(max_dist/h))
    bin_edges = gs.variogram.standard_bins((x, y), bin_no=bin_no, max_dist=max_dist)
    bin_center, gamma, counts = gs.vario_estimate((x, y), z, bin_edges=bin_edges,return_counts=True)
    return bin_center, gamma
