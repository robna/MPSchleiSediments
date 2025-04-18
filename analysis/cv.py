import numpy as np
import pandas as pd
from itertools import compress
import joblib
from datetime import datetime
import time
from tqdm import tqdm
import threading

from sklearn.model_selection import LeaveOneOut as sk_loo
from statsmodels.sandbox.tools.cross_val import LeaveOneOut as sm_loo
from sklearn.model_selection import GridSearchCV, cross_validate, KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from sklearn import clone, set_config
# set_config(transform_output='pandas')  # only works for sklearn >= 1.2
from sklearn.utils import parallel_backend

import glm, geo
from helpers import tqdm_joblib
from cv_helpers import iqm, median_absolute_percentage_error
from settings import Config, featurelist, getLogger, target


logger = getLogger()
            
            
def loocv(df):
    """
    Perform LOOCV on a dataframe.
    :param df: dataframe containing target and predictors
    :return pred: dataframes with predictions and metrics
    """
    pred = pd.DataFrame(columns=['Sample', 'pred'])  # create empty dataframe for predictions
    n = df.shape[0]  # number of samples
    p = Config.glm_formula.count('+') + Config.glm_formula.count('*') + 1  # number of predictors
    for train_index, test_index in sm_loo(df.shape[0]):
        train = df.loc[train_index, :]
        test = df.loc[test_index, :]
        glm_res = glm.glm(train)
        predi = pd.concat([test.Sample, glm_res.predict(test).rename('pred')], axis=1)
        pred = pd.concat([pred, predi])

    target = df.loc[:, Config.glm_formula.split(' ~')[0]]  # isolate target variable (observed values)
    target_name = Config.glm_formula.split(' ~')[0]
    pred.loc[:, target_name] = target
    metrics = performance(
        pred.set_index('Sample')[target_name],
        pred.set_index('Sample')['pred'],
        p)
    return pred, metrics


def loocv_interp(station_data, target, xgrid, ygrid, res, poly, tool, n_jobs=1, verbose=False):
    '''
    Perform LOOCV for interpolation
    '''
    xmin, ymax = xgrid[0,0], ygrid[0,0]
    loo = sk_loo()

    def interpolate_leftout(train_index, test_index):
        return [test_index[0],
            geo.sample_array_raster(
            geo.interclip(
                            station_data.loc[train_index],
                            target, xgrid, ygrid, poly, tool,
                            clip=False, plot=False,
                        ),
                xmin, ymax, res, station_data.loc[test_index]
                )[0]
            ]
    
    if n_jobs == 1:
        l = [
             interpolate_leftout(train_index, test_index) for
             train_index, test_index in 
             tqdm(loo.split(station_data), desc=f'LOOCV of {tool}', total=station_data.shape[0])
            ]
    else:
        with tqdm_joblib(tqdm(desc=f'Parallel LOOCV of {tool}', total=station_data.shape[0])) as progress:
            l = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
                        joblib.delayed(interpolate_leftout)(train_index, test_index
                        ) for
                            train_index, test_index in 
                            loo.split(station_data)
                    )

    time.sleep(1)  # wait a second, just to get print outputs in correct order
    return pd.Series({i[0]: i[1] for i in l})
    

def performance(observed, predicted, p=None):
    """
    Measure the performance of a prediction using a set of metrics.
    Series of observed and predicted values must have the same index.
    :param observed: pandas series of observed values with meaningful index, eg. sample names
    :param predicted: pandas series of predicted values with meaningful index, eg. sample names
    :param p: optional number of predictors used by the model which generated the prediction
    :return metrics: a df listing the metrics' values + explanations.
    """
    maxe = max_error(observed, predicted)
    medae = median_absolute_error(observed, predicted)
    medape = median_absolute_percentage_error(observed, predicted)
    mae = mean_absolute_error(observed, predicted)
    mape = mean_absolute_percentage_error(observed, predicted)
    rmse = np.sqrt(mean_squared_error(observed, predicted))
    r2 = r2_score(observed, predicted)
    if p is not None:
        adj_r2 = 1 - (1 - r2) * (len(predicted) - 1) / (len(predicted) - p)

    metrics = pd.DataFrame(columns=['Metric', 'Value', 'Info'])
    metrics.loc[0] = ['Max Error', maxe, f"Where: {np.abs(observed - predicted).idxmax()}"]
    metrics.loc[1] = ['Median Absolute Error', medae, 'In units of observed variable']
    metrics.loc[2] = ['Median Absolute Percentage Error', medape, 'Relative error: 0 (no error), 1 (100% misestimated), >1 (arbitrarily wrong)']
    metrics.loc[3] = ['Mean Absolute Error', mae, 'In units of observed variable']
    metrics.loc[4] = ['Mean Absolute Percentage Error', mape, 'Relative error: 0 (no error), 1 (100% misestimated), >1 (arbitrarily wrong)']
    metrics.loc[5] = ['Root Mean Square Error', rmse, 'In units of observed variable']
    metrics.loc[6] = ['R²', r2, 'From 1 (perfect prediction) and 0 (just as good as predicting the mean) to neg. infinity (arbitrarily wrong)']
    if p is not None:
        metrics.loc[7] = [f'Adjusted R² (n={len(predicted)}, p={p})', adj_r2, 'Like R² but takes into account sample number "n" and feature number "p". An additional feature is justified if it increases this metric.']
    return metrics


def get_performance(df, target, kind='predicted', with_outliers=False):
    '''
    Wrapper function to calculate yhat vs y perfomance scores of df columns.
    y-column: indicated by name <target>_observed
    yhat-column: named <target>_<kind>
    '''
    if 'Sample' in df.columns:
        df = df.copy().set_index('Sample')
    df_f = df.loc[(df.Type=='observed') & 1 if with_outliers else (df.outlier_excl==False), :]
    print(f'Number of ŷ-vs-y pairs: {df_f.shape[0]}')
    return performance(df_f.loc[:, f'{target}_observed'],
                       df_f.loc[:, f'{target}_{kind}'])
    
    
def make_setup_dict(**kwargs):
    setup = kwargs  # dict of lists of NCV parameter settings: first element for outer, second for inner CV    
    
    setup['cv_scheme'] = [
        ContinuousStratifiedKFold(), # using single StratifiedKFold in outer which will be repeated manually in loop to extract the test set indices at each iteration
        ContinuousStratifiedKFold()  # inner CV scheme with repetition
    ]
    
    if 'repeats' in setup.keys():
        if isinstance(setup['repeats'], (int, float)):
            setup['repeats'] = (setup['repeats'], setup['repeats'])
        else:
            setup['repeats'] = (setup['repeats'][0], max(int(setup['repeats'][1]), 1))
        setup['cv_scheme'][1].n_repeats = setup['repeats'][1]
            
    if 'folds' in setup.keys():
        if isinstance(setup['folds'], (int, float)):
            setup['folds'] = (setup['folds'], setup['folds'])
        setup['cv_scheme'][0].n_splits = setup['folds'][0]
        setup['cv_scheme'][1].n_splits = setup['folds'][1]
            
    if 'stratification_mode' in setup.keys():
        if isinstance(setup['stratification_mode'], (str, type(None))):
            setup['stratification_mode'] = (setup['stratification_mode'], setup['stratification_mode'])
        setup['cv_scheme'][0].mode = setup['stratification_mode'][0]
        setup['cv_scheme'][1].mode = setup['stratification_mode'][1]
    
    if 'strata' in setup.keys():
        if isinstance(setup['strata'], (int, float)):
            setup['strata'] = (setup['strata'], setup['strata'])
        setup['cv_scheme'][0].strata = setup['strata'][0]
        setup['cv_scheme'][1].strata = setup['strata'][1]
    
    return setup


class ContinuousStratifiedKFold:
    """
    Continuous stratified k-fold splitter.

    Provides train/test indices to split data into train/test sets. Data is first divided into strata based on the
    continuous target variable, and then splits are yielded with an as-equal-as-possible representation of each stratum in each
    fold. This ensures that each fold contains a representative sample of each stratum.

    There are two modes for stratification implemented: 'quantile' and 'interval'.
    
    In 'quantile' mode, the target variable is discretised using pandas.qcut, which bins the data based on quantiles.
    This means that the bins will be of (nearly) equal sample number,
    but not necessarily of equal bin width. If the target variable is not evenly distributed,
    this may result in some bins containing one more sample than others. If strata is a number it will dictate how many quantiles
    will be used. If strata is list-like of floats between 0 and one, its values will be used as quantile edges.
    
    In 'interval' mode, the target variable is discretised using pandas.cut, which bins the data into strata equally sized bins
    if strata is a number, or into the bins defined by the their edge values in strata, if it is list-like.
    This means that the bins will be of equal width, but not necessarily of equal sample number. This mode can only be used
    if the bin with the fewest samples has at least as many samples as the number of inteded folds.

    If mode is set to None or omitted, interval strafitication is attempted first, and if that fails, quantile stratification.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    strata : int or None, default=None
        Number of strata to create based on the target variable. If None, False or 0: strata = n_splits.
    n_repeats : int or None, default=None
        Number of times to repeat the cross-validation process with different random splits.
        If None, defaults to 1 (meaning single split into n_splits folds.
    shuffle : bool, default=False
        Whether or not to shuffle the data before splitting it into folds.
    mode : str, default='None'
        Mode to use for stratification. Must be one of 'None', 'quantile' or 'interval'.
    random_state : int or None, default=None
        Random seed to use for shuffling the data and generating random splits.

    Attributes
    ----------
    n_splits : int
        Number of folds.
    strata : int or list-like
        Number of strata if int. Otherwise values will be used for bin edges.
    n_repeats : int
        Number of repeats.
    shuffle : bool
        Whether or not to shuffle the data before splitting it into folds.
    mode : str
        Mode to use for creating bins for stratification.
    random_state : int or None
        Random seed to use for shuffling the data and generating random splits.

    Methods
    -------
    split(X, y[, groups])
        Generate indices to split data into training and test set.
    get_n_splits([X, y, groups])
        Returns the number of splitting iterations in the cross-validator.
        For repeated splits this equals n_splits * n_repeats

    Notes
    -----
    This cross-valdidation splitter is a variation of StratifiedKFold that supports continuous target variables.
    It works by first binning the target variable into P bins, where P is the number of strata, which is either
    specified by the user or defaults to n_splits. Setting strata to 1 will result in a single bin, which is
    equivalent to a non-stratified KFold.

    Examples
    --------
    >>> from cv import ContinuousStratifiedKFold
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13,14], [15,16], [17,18], [19,20], [21,22], [23,24]])
    >>> y = np.array([0.0, 0.5, 2.5, 2.0, 10.0, 9.9, 9.8, 9.7, 9.6, 9.5, 14.0, 14.5])
    >>> cv = ContinuousStratifiedKFold(n_splits=3, mode='quantile')
    >>> for train_index, test_index in cv.split(X, y):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [ 2  3  5  7  8  9 10 11] TEST: [ 0  1  4  6]
    TRAIN: [ 0  1  3  4  6  8  9 11] TEST: [ 2  5  7 10]
    TRAIN: [ 0  1  2  4  5  6  7 10] TEST: [ 3  8  9 11]
    """
    def __init__(self, n_splits=5, strata=None, n_repeats=None, shuffle=False, mode=None, random_state=None):
        self.n_splits = n_splits
        self.strata = strata if strata else n_splits
        self.n_repeats = n_repeats if n_repeats else 1
        self.shuffle = shuffle
        self.mode = mode
        self.random_state = random_state

    def split(self, X, y, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target variable.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples. Not used in this implementation.

        Yields
        ------
        train_index : ndarray
            The training set indices for that split.
        test_index : ndarray
            The testing set indices for that split.
        """

        # Check mode and set stratificator function
        if self.mode == 'quantile':
            self.stratificator = pd.qcut
        elif self.mode == 'interval' or self.mode is None:
            # test if when using pd.cut, the min number of samples in bins is at least n_splits
            self.stratificator = pd.cut
            min_smpls = self.stratificator(y, self.strata).value_counts()
            if all(min_smpls < self.n_splits):
                if self.mode == 'interval': 
                    raise ValueError(f'''
                                        Not enough samples in target variable to use interval stratification.
                                        Splitting into {self.strata} bins would result in the smallest bin having
                                        only {min_smpls} samples, but at least {self.n_splits} are required.
                                        Try lowering the number of strata or use quantile stratification instead.
                                        ''')
                else:
                    self.stratificator = pd.qcut  # fallback to quantile stratification
                    if isinstance(self.strata, (list, tuple)):
                        self.strata = len(self.strata)  # when using fallback but original strata was list-like of bin edges, define same number of quantile bins
        else:
            raise ValueError(f'''
                             Invalid mode. Must be one of 'quantile', 'interval' or None.
                             ''')
        # print(f'Control print from function: cv.ContinuousStratifiedKFold.split()  -> self.stratificator is "{self.stratificator.__name__}"')
        
        # Create P bins based on the target variable
        self.labels = [f'stratum_{s}' for s in range(len(self.strata)-1 if isinstance(self.strata, (list, tuple)) else self.strata)]
        strata, bins = self.stratificator(y, self.strata, retbins=True, labels=self.labels)
        # print({'bins': bins, 'strata': strata})  # turn on for diagnostics
        
        # Stratify the data based on the bins
        if self.n_repeats > 1:
            skf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        else:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for train_index, test_index in skf.split(X, strata, groups):  # groups is not used here. It just exists for compatibility reasons.
            yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Training data. Unused, present here for API consistency by convention.
        y : array-like of shape (n_samples,), default=None
            Target variable. Unused, present here for API consistency by convention.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples. Unused, present here for API consistency by convention.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator. For repeated splits this equals n_splits * n_repeats.
        """
        return self.n_splits * self.n_repeats

    
def ncv(pipe, params, model_X, model_y, scorers, setup):
    '''
    Performs nested cross-validation on a given pipeline and parameter grid.

    Parameters:
    -----------
    pipe: sklearn.pipeline.Pipeline object
        The pipeline to be optimized through cross-validation.

    params: dict
        Dictionary with hyperparameters to be tuned, as accepted by GridSearchCV.

    scorers: dict
        Dictionary of scoring functions for evaluating model performance. 
        Each scorer must follow the scikit-learn API for scoring functions.

    setup: dict
        Dictionary with configuration parameters for the cross-validation:
          first elements for outer, second for inner CV.
        It must contain the following keys:
            - 'cv_scheme': list-like of two sklearn CV splitters (like e.g. KFold or RepeatedKFold), 
                for outer and inner cross-validation folds, respectively.
            - 'verbosity': list-like of two integers, 
                with the verbosity level for outer and inner cross-validation, respectively.
            - 'n_jobs': list-like of two integers,
                with the number of parallel jobs to be run for outer and inner cross-validation, respectively.
            - 'repeats': list-like of two integers,
                with the number of repetitions for outer and inner cross-validation, respectively.

    Returns:
    --------
    outerCV: dict
        Dictionary with the results of the nested cross-validation. 
    '''
# with parallel_backend('threading'):#(setup['parallel_backend'][1], n_jobs=setup['n_jobs'][1]):
    innerCV = GridSearchCV(
        pipe,
        params,
        scoring=scorers,
        refit=best_scored,
        cv=setup['cv_scheme'][1],
        verbose=setup['verbosity'][1],
        n_jobs=setup['n_jobs'][1]
        )
# with parallel_backend(setup['parallel_backend'][0], n_jobs=setup['n_jobs'][0]):
    NCV = cross_validate(
        innerCV,
        model_X,
        model_y,
        scoring=scorers,
        cv=setup['cv_scheme'][0],
        return_train_score=False,
        return_estimator=True,
        verbose=setup['verbosity'][0],
        n_jobs=setup['n_jobs'][0]
        )
    return NCV


def rep_ncv(pipe, params, model_X, model_y, scorers, setup):
    '''
    The pipeline is run by searching the provided parameter space
    using scorings of a crossvalidation technique to find out how
    each model candidate performs.
    setup: dict of lists of setting values for cv_scheme,
           n_jobs, parallel_backend, verbosity
           [first element for outer CV, second for inner CV]
    '''
    
    repetitions = setup['repeats'][0]
    starttime = datetime.now()
    NCV = pd.DataFrame()

    try:
        for i in range(repetitions):
            repstart = datetime.now()
            logger.info(f'>>>>>>>>>>   Starting NCV repetition {i+1} at {repstart.strftime("%Y-%m-%d %H:%M:%S")}   <<<<<<<<<<')

            setup['cv_scheme'][0].shuffle = True  # activate shuffling to yield varying outer fold train / test splits
            setup['cv_scheme'][0].random_state = setup['cv_scheme'][1].random_state = i  # set different random states for each repetition

            outerCV = ncv(pipe, params, model_X, model_y, scorers, setup)

            for outer_fold in range(len(outerCV['estimator'])):  # iterate through the outer folds
                res = outerCV['estimator'][outer_fold].cv_results_  # get the dict of cv results (including test scores on each inner fold) for the current outer fold
                append_agg_cv_scores(res, agg='median')
                append_agg_cv_scores(res, agg='iqm')
                # get_median_cv_scores(res)  # append median scores to dict inplace
                # get_iqm_cv_scores(res)  # append iqm scores to dict inplace

            rep = {'NCV_repetition': np.repeat(i, len(list(outerCV.values())[0]))}

            outer_testsets = get_test_sets(setup['cv_scheme'][0], model_X, model_y, prefix='outer')
            # print(f'Outer test sets of outer Rep {i}: \n {outer_testsets}')
            
            # inner_testsets = {
            #     # 'inner_test_set_indices': [],  # test set indices refer to the reduced lists of samples (ots samples are missing, and vary between outer folds)
            #     'inner_test_set_samples': []
            # }
            # for ots in outer_testsets['outer_test_set_indices']:
            #     current_inner_testsets = get_test_sets(setup['cv_scheme'][1], model_X.iloc[~ots], model_y.iloc[~ots], prefix='inner')
            #     for k in inner_testsets:
            #         inner_testsets[k].append(current_inner_testsets[k])
            # # print(f'Inner test sets of outer Rep {i}: \n {inner_testsets}')
                              
            outerCV = {
                **rep,
                **outer_testsets,
                # **inner_testsets,  # TODO: had to take out the inner folds extraction, as it can throw "n_splits cannot be smaller than samples in each class" error. Needs checking.
                **outerCV}

            ## Print details of outerCV dict (uncomment for diagnostic reasons)
            # print([(
            #     k,
            #     type(outerCV[k]),
            #     outerCV[k].shape if isinstance(outerCV[k], np.ndarray) else len(outerCV[k])
            # )for k in outerCV.keys()])
            
            outerCV_df = pd.DataFrame(outerCV).rename_axis('OuterCV_fold')
            outerCV_df['Intra-Rep_rank'] = outerCV_df[f'test_{Config.refit_scorer}'].rank(ascending=False).astype(int)
            outerCV_df.set_index(['NCV_repetition', outerCV_df.index, 'Intra-Rep_rank'], inplace=True)
            NCV = pd.concat([NCV, outerCV_df])
            repdur = datetime.now() - repstart
            time_needed = datetime.now() - starttime
            time.sleep(1)  # just to finish the printings, not mixing up by already starting the next repetition
            logger.info(f'Duration for repetition {i+1}/{repetitions}:   {repdur}    (total time so far:   {time_needed})\n')

    except KeyboardInterrupt:
        setup['repeats'] = (i, setup['repeats'][1])  # change outer repeats to how many repeatitions were actually completed
        get_inner_test_scores(NCV)
        print_end(setup['repeats'][0], time_needed, msg=f'Stopped early after {setup["repeats"][0]} of {repetitions} repetitions.')
        return NCV, setup

    get_inner_test_scores(NCV)
    print_end(repetitions, time_needed)
    return NCV, setup


def print_start(starttime):
    logger.info(f'\n\n\n\nStarted new {Config.ncv_mode} model run with savestamp: {starttime.strftime("%Y%m%d_%H%M%S")}')

    
def print_end(r, t, msg=None):
    if msg:
        logger.info(msg)
    logger.info(f'##########   Total duration for {r} repetitions: {t}   ##########')


def compete_rep_ncv(pipe, params, model_X, model_y, scorers, setup):
    starttime = datetime.now()
    print_start(starttime)
    name_reg = [pg['regressor'][0].__class__.__name__ for pg in params]
    NCV, setup = rep_ncv(pipe, params, model_X, model_y, scorers, setup)
    NCV = NCV.assign(run_with = ', '.join(name_reg))
    NCV.set_index(['run_with', NCV.index], inplace=True)
    time_needed = datetime.now() - starttime
    return NCV, setup, starttime, time_needed


def compara_rep_ncv(pipe, params, model_X, model_y, scorers, setup):
    starttime = datetime.now()
    print_start(starttime)
    NCV = pd.DataFrame()
    for param_set in params:
        name_reg = param_set['regressor'][0].__class__.__name__
        if name_reg == 'XGBRegressor':
            name_reg = name_reg + f"_{param_set['regressor__booster'][0]}"
        logger.info(f'\n\n    Starting NCV run for model class {name_reg}\n\n')
        sNCV, setup = rep_ncv(pipe, param_set, model_X, model_y, scorers, setup)
        sNCV = sNCV.assign(run_with = name_reg)
        sNCV.set_index(['run_with', sNCV.index], inplace=True)
        NCV = pd.concat([NCV, sNCV])
    time_needed = datetime.now() - starttime
    logger.info(f'\n\n    Finished all {len(params)} comparative model runs in {time_needed}\n\n')
    return NCV, setup, starttime, time_needed


def best_scored(cv_results):
    """
    Find the best median / mean / interquartile mean score from a cross-validation result dictionary.
    :param cv_results: dictionary of cross-validation results
    :return: index of best scoring candidate accoring to the chosen refit scorer and aggregation method
    """
    
    inner_test_scores = np.array([
                                    scores for key, scores
                                    in cv_results.items()
                                    if key.startswith('split')
                                    and f'test_{Config.refit_scorer}'
                                    in key
                                ])
    # print(f'Control print from function: cv.best_scored()  -> Config.select_best is "{Config.select_best}"')
    # print(f'Control print from function: cv.best_scored()  -> Config.refit_scorer is "{Config.refit_scorer}"')
    # print(f'Control print from function: cv.best_scored()  -> inner_test_scores shape: {inner_test_scores.shape}')
    # print(f'Control print from function: cv.best_scored()  -> inner_test_scores: {inner_test_scores}')
    # inner_test_keys = np.array([
    #                                 key for key, scores
    #                                 in cv_results.items()
    #                                 if key.startswith('split')
    #                                 and f'test_{Config.refit_scorer}'
    #                                 in key
    #                             ])
    # print(f'Control print from function: cv.best_scored()  -> inner_test_scores:')
    # print(pd.DataFrame(inner_test_scores.T, columns=inner_test_keys).T)
    
    if Config.select_best == 'median':
        avg_inner_test_scores = np.median(inner_test_scores, axis=0)
    elif Config.select_best == 'mean':
        avg_inner_test_scores = np.mean(inner_test_scores, axis=0)
    elif Config.select_best == 'iqm':
        avg_inner_test_scores = np.array([iqm(s) for s in inner_test_scores.T])
    else:
        raise ValueError(f'Can only select for best mean, median or iqm score. You supplied "{Config.select_best}".')
    best_index = avg_inner_test_scores.argmax()
    # print(f'Control print from function: cv.best_scored()  -> avg_inner_test_scores shape = {avg_inner_test_scores.shape}, max = {avg_inner_test_scores.max()}, best_index = {best_index}')
    # print(f'Control print from function: cv.best_scored()  -> best_estimator: {cv_results["estimator"][best_index]}')
    return best_index
    

def append_agg_cv_scores(res, agg='median'):
    """
    Add median or interquartile mean cross-validation scores to the given CV results.

    Parameters
    ----------
    res : dict
        Result object of nested cross-validation.
    agg : str, optional
        Aggregation method to be used for cross-validation scores. Must be 'median' or 'iqm'. Default is 'median'.

    Notes
    -----
    The dictionary elements are modified inplace and not returned.
    The function expects that multiple scorers were used in the cross-validation
    and that they are listed in the Config.scorers dictionary.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `agg` is not 'median' or 'iqm'.
    """

    if agg == 'median':
        aggregator = np.median
    elif agg == 'iqm':
        aggregator = iqm
    else:
        raise ValueError(f'Can only aggregate for median or iqm score. You supplied "{agg}".')

    res_df = pd.DataFrame(res)
    for k, v in Config.scorers.items():
        if f'rank_test_{k}' in res:
            res[f'rank_by_mean_test_{k}'] = res.pop(f'rank_test_{k}')
        splits = res_df.filter(regex=f"^split\d+_test_{k}")  # get all columns of the df that are named like "split<some-number>_test_<scorer>"
        # print(splits)
        
        if splits.isna().any().any():
            print('There were NaN in some of the inner test scores: \n',
                  splits[splits.isna().any(axis=1)].rename_axis(index='param_gridpoint').T,
                  f'\n The NaN-splits will not be included in the calculation of the {agg}_test_{k} for the configurations of these param_gridpoints.')
            
        res_df[f'{agg}_test_{k}'] = [aggregator(s[~np.isnan(s)]) for s in splits.values]
        # print(f'Control print from function: cv.append_agg_cv_scores()  -> number of splits to aggregate: {len(splits.columns)}, aggregation: {aggregator}, scorer: {k}')
        
        if all(res_df[f'{agg}_test_{k}'].isna()):
            res_df[f'rank_by_{agg}_test_{k}'] = np.nan
        else:
            # with pd.option_context('display.max_rows', None):
                # print('\n Splits: \n', splits)
                # print(f'Ranks of {agg}_test_{k}: \n {res_df[f"{agg}_test_{k}"].rank(ascending=False)}')
            res_df[f'rank_by_{agg}_test_{k}'] = res_df[f'{agg}_test_{k}'].rank(ascending=False).astype(int)

        res[f'{agg}_test_{k}'] = res_df[f'{agg}_test_{k}'].to_numpy()
        res[f'rank_by_{agg}_test_{k}'] = res_df[f'rank_by_{agg}_test_{k}'].to_numpy()


# def get_median_cv_scores(res):
#     """
#     Add median cross-validation scores to CV results.
#     :param res: cv_results_ dict of cross-validation object
    
#     Note: the dict elements are modified inplace and not returned.
#     """

#     res_df = pd.DataFrame(res)
#     for k, v in Config.scorers.items():
#         if f'rank_test_{k}' in res:
#             res[f'rank_by_mean_test_{k}'] = res.pop(f'rank_test_{k}')
#         res_df[f'median_test_{k}'] = res_df.filter(regex=f'^split._test_{k}').median(axis=1)
#         if all(res_df[f'median_test_{k}'].isna()):
#             res_df[f'rank_by_median_test_{k}'] = np.nan
#         else:
#             res_df[f'rank_by_median_test_{k}'] = res_df[f'median_test_{k}'].rank(ascending=False).astype(int)

#         res[f'median_test_{k}'] = res_df[f'median_test_{k}'].to_numpy()
#         res[f'rank_by_median_test_{k}'] = res_df[f'rank_by_median_test_{k}'].to_numpy()


# def get_iqm_cv_scores(res):
#     """
#     Add iqm cross-validation scores to CV results.
#     :param res: cv_results_ dict of cross-validation object
    
#     Note: the dict elements are modified inplace and not returned.
#     """

#     res_df = pd.DataFrame(res)
#     for k, v in Config.scorers.items():
#         if f'rank_test_{k}' in res:
#             res[f'rank_by_mean_test_{k}'] = res.pop(f'rank_test_{k}')
#         res_df[f'iqm_test_{k}'] = [iqm(s) for s in res_df.filter(regex=f'^split._test_{k}').values]
#         if all(res_df[f'iqm_test_{k}'].isna()):
#             res_df[f'rank_by_iqm_test_{k}'] = np.nan
#         else:
#             res_df[f'rank_by_iqm_test_{k}'] = res_df[f'iqm_test_{k}'].rank(ascending=False).astype(int)

#         res[f'iqm_test_{k}'] = res_df[f'iqm_test_{k}'].to_numpy()
#         res[f'rank_by_iqm_test_{k}'] = res_df[f'rank_by_iqm_test_{k}'].to_numpy()


def get_inner_test_scores(NCV):
    # print(f'Control print: cv.get_inner_test_scores()  -> Config.select_best is "{Config.select_best}"')
    for idx in NCV.index:
        b = NCV.loc[idx].estimator.cv_results_[f'rank_by_{Config.select_best}_test_{Config.refit_scorer}'].argmin()  # get index of candidate which won this round
        for s in Config.scorers.keys():
            NCV.loc[idx, f'inner_{Config.select_best}_test_{s}'] = NCV.loc[idx].estimator.cv_results_[f'{Config.select_best}_test_{s}'][b]


def get_test_sets(splitter, X, y, prefix=''):
    """
    Get the names and indices of samples in test sets.
    """
    return {
        f'{prefix}_test_set_indices': [test_index for train_index, test_index in splitter.split(X, y)],
        f'{prefix}_test_set_samples': [X.index.values[test_index] for train_index, test_index in splitter.split(X, y)],    
    }


def get_best_params(NCV, params=None):
    '''
    Extracts the params of the wining models of each fold. There are
    two similar but different variants implemented. Which one is used
    depends on whether a param grid is supplied or not.
    
    params: if supplied with the corresponding gridsearch param grid,
            the output params will only report the chosen regressor
            params
            if params=None, chosen params for every pipeline step
            are reported
    '''
    
    if params is None:
        ## Get best model params for each of the outer cv folds:
        best_params_df = pd.DataFrame()
        for i, model in enumerate(NCV['estimator']):
            best_params = model.best_params_
            bp_df = pd.DataFrame()
            for key, value in best_params.items():
                bp_df[key] = [value]
                bp_df.index = [i]
            if 'regressor' in bp_df.columns:
                bp_df.regressor = bp_df.regressor.apply(lambda x: x.__class__.__name__)
            best_params_df = pd.concat([best_params_df, bp_df])
            
        return best_params_df
    
    else:
        ## Extract specific regressor params chosen for each best model
        param_choices = pd.DataFrame()
        for i in NCV.reset_index(drop=True).index:
            regr = NCV.reset_index(drop=True).loc[i, 'estimator'].best_estimator_.named_steps['regressor']
            name_reg = regr.__class__.__name__
            res_params = regr.get_params()
            paramgrids = [grid for grid in params if grid['regressor'][0].__class__.__name__ == name_reg]
            if len(paramgrids) > 1 and name_reg == 'XGBRegressor':
                correct_grid = [(res_params['booster'] in paramgrid['regressor__booster']) for paramgrid in paramgrids]
                paramgrids = list(compress(paramgrids, correct_grid))
            if len(paramgrids) > 1:
                raise ValueError('There seems to be some duplication in the regressor param grids!')
            for param, value in res_params.items():
                if f'regressor__{param}' in paramgrids[0].keys():
                    param_choices.loc[i, f'{name_reg}__{param}'] = value
        return param_choices

    
# def refit_on_all(NCV, model_X, model_y, est, i):
#     return clone(est.best_estimator_.named_steps['regressor']).fit(model_X[NCV.features.loc[i]], model_y)
    

def process_results(NCV, model_X, model_y, params=None, allSamples_Scores=False, refitOnAll=False):
    
    best_params_df = get_best_params(NCV, params)
    NCV = pd.concat([NCV.reset_index(drop=True), best_params_df], axis=1).set_index(NCV.index)
    
    ## Get names of features used by the models
    if 'preprocessor__selector__kw_args' in NCV.columns:
        NCV.rename(columns={'preprocessor__selector__kw_args': 'features'}, inplace=True)
        s = NCV.features.apply(lambda x: [x['feature_set'], x['feature_sets'][x['feature_set']]])
        d = pd.DataFrame.from_dict(dict(zip(s.index, s.values))).T
        NCV.insert(NCV.columns.get_loc('features'), 'feature_combi_ID', d[0])
        NCV.features = d[1]
        NCV.drop('preprocessor__selector', axis=1, inplace=True)
        
    if allSamples_Scores:
        ## Calculate scores of the best model for each outer cv fold against all data
        for key, val in Config.scorers.items():
            NCV[f'allSamples_{key}'] = [
                val[0](model_y, est.predict(model_X))
                for _, est in NCV['estimator'].items()
            ]
    if refitOnAll:
        ## Now refit all models in outerCV on all data
        # tried with Parallel but didn't work due to "could not pickle" error
        # NCV['estimator_refit_on_all'] = joblib.Parallel(n_jobs=-1)(joblib.delayed(refit_on_all)(NCV, model_X, model_y, est, i)
        #     for i, est in NCV['estimator'].items())
        NCV['estimator_refit_on_all'] = [
            clone(est.best_estimator_.named_steps['regressor']).fit(
            model_X[NCV.features.loc[i]], model_y)
            for i, est in NCV['estimator'].items()
        ]
        if allSamples_Scores:
            ## Calculate scores against all data again after refitting
            for key, val in Config.scorers.items():
                NCV[f'allSamples_{key}_refit'] = [
                    val[0](model_y, est.predict(model_X[NCV.features.loc[i]]))
                    for i, est in NCV['estimator_refit_on_all'].items()
                ]
    ## Drop regressor objects from summary
    # NCV.drop('regressor', axis=1, inplace=True)
    return NCV
    
def transform_score_df(scored_multi):
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
    df = df.reset_index(drop=True).set_index(['run_with', 'NCV_repetitions'])
    
    return df


def aggregation(NCV, setup, r=None):
    scored_comp = pd.DataFrame()
    for model_type, group in NCV.groupby('run_with'):
        scored = aggregate_scores(group, setup, r=r)
        scored = scored.assign(run_with = model_type)
        scored.set_index(['run_with', scored.index], inplace=True)
        scored_comp = pd.concat([scored_comp, scored])
    return scored_comp
    
    
def aggregate_scores(NCV, setup, r=None):
    ## Calculate score aggregations and their variance
    if r is None:
        r = setup['repeats'][0]
    f = setup['folds'][0]
    rep_groups = NCV.groupby('NCV_repetition')
    scored = pd.DataFrame({
        key: [
        np.median(rep_groups[f'test_{key}'].median()),  # Median of repetitions' outer test median scores
        iqm(rep_groups[f'test_{key}'].median()),  # IQM of repetitions' outer test median scores
        np.mean(rep_groups[f'test_{key}'].median()),  # Mean of repetitions' outer test median scores
        np.std(rep_groups[f'test_{key}'].median()),  # Standard deviation of repetitions' outer test median scores
        np.median(NCV[f'test_{key}']),  # Median of all outer test scores
        np.median(rep_groups[f'test_{key}'].agg(iqm)),  # Median of repetitions' outer test iqm scores
        iqm(rep_groups[f'test_{key}'].agg(iqm)),  # IQM of repetitions' outer test iqm scores
        np.mean(rep_groups[f'test_{key}'].agg(iqm)),  # Mean of repetitions' outer test iqm scores
        np.std(rep_groups[f'test_{key}'].agg(iqm)),  # Standard deviation of repetitions' outer test iqm scores
        iqm(NCV[f'test_{key}']),  # IQM of all outer test scores
        np.median(rep_groups[f'test_{key}'].mean()),  # Median of repetitions' outer test mean scores
        iqm(rep_groups[f'test_{key}'].mean()),  # IQM of repetitions' outer test mean scores
        np.mean(rep_groups[f'test_{key}'].mean()),  # Mean of repetitions' outer test mean scores
        np.std(rep_groups[f'test_{key}'].mean()),  # Standard deviation of repetitions' outer test mean scores
        np.mean(NCV[f'test_{key}']),  # Mean of all outer test scores
            ] for key in Config.scorers
    }, index=[
        f'Median_of_medians_of_{r}_repetitions',
        f'IQM_of_medians_of_{r}_repetitions',
        f'Mean_of_medians_of_{r}_repetitions',
        f'Stdev_of_medians_of_{r}_repetitions',
        f'Median_of_all_{r * f}_folds_together',
        f'Median_of_IQMs_of_{r}_repetitions',
        f'IQM_of_IQMs_of_{r}_repetitions',
        f'Mean_of_IQMs_of_{r}_repetitions',
        f'Stdev_of_IQMs_of_{r}_repetitions',
        f'IQM_of_all_{r * f}_folds_together',
        f'Median_of_means_of_{r}_repetitions',
        f'IQM_of_means_of_{r}_repetitions',
        f'Mean_of_means_of_{r}_repetitions',
        f'Stdev_of_means_of_{r}_repetitions',
        f'Mean_of_all_{r * f}_folds_together',
    ])
    return scored


def rensembling(NCV):
    '''
    For repNCV results from running in comparative mode,
    the corresponding results that would have come out
    of an equivalent run in competitive mode, can be generated
    and get appended below the NCV DF.
    '''
    if Config.ncv_mode == 'comparative':
        rensembled = NCV.rename(  # Turning index level for model class into combined string of all classes like in competitive mode
        index={
            r: '# Competitive mode:, ' + ', '.join(list(NCV.index.levels[0]))
            for r in NCV.index.get_level_values(0)
        })
        ensemble_idx = rensembled.groupby(['NCV_repetition', 'OuterCV_fold'])[f'inner_{Config.select_best}_test_{Config.refit_scorer}'].agg(pd.Series.idxmax).to_list()
        NCV = pd.concat([NCV, rensembled.loc[ensemble_idx]])
    return NCV


def make_header(NCV, setup, starttime, time_needed, droplist, model_X, num_feat, feature_candidates_list, scaler, regressor_params, dup_testsets):
    ## Create a dataframe of aggregated scores and standard deviations
    scored = aggregate_scores(NCV, setup)    
    scored = scored.round(4).to_csv(sep=';') if Config.ncv_mode=='competitive' else 'COMPARATIVE RUN: no common scores available. Look at the individual scores of each model class!'
    lin_combis = Config.lin_combis if hasattr(Config, 'lin_combis') else 'no restrictions'
    tree_combis = Config.tree_combis if hasattr(Config, 'tree_combis') else 'no restrictions'
    ## Prepare meta-data header of NCV run for export to file
    header = f'''

    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    Model started:;{starttime}
    Duration:;{time_needed} on {joblib.cpu_count()} cpu cores
    Response (target variable):;{target}
    Outliers excluded {len(droplist)}:;{droplist}
    Samples:;{model_X.index.to_list()}
    Vertical merge:;{Config.vertical_merge}

    Number of features per candidate (min, max):;{num_feat};Special restrictions for certain model types:;lin_combis:;{lin_combis};tree_combis:;{tree_combis}
    Total number of feature combinations tested:;{len(feature_candidates_list)}
    Available features: {len(featurelist)};{featurelist}
    Restricted combinations (mutal exclusive features, exclusive keywords):;{Config.mutual_exclusive};{Config.exclusive_keywords}

    Scaler:;{scaler}
    Regressors:;{[[g['regressor'][0].__class__.__name__, [{k: v} for k, v in g.items() if k.startswith('regressor__')]] for g in regressor_params]}
    Winning candidate of gridsearch (inner CV loop) determined by best;{Config.select_best} {Config.refit_scorer} score; Note: results within each repetition are sorted by this!

    CV schemes (set mode: '{setup['stratification_mode'] if 'stratification_mode' in setup.keys() else None}'):
        outer:;{setup['cv_scheme'][0].__class__.__name__};mode (last split):{setup['cv_scheme'][0].mode}; Number of duplicated test sets:;{len(dup_testsets)}
        inner:;{setup['cv_scheme'][1].__class__.__name__};mode (last split):{setup['cv_scheme'][1].mode};
    Strata (set strata: '{setup['strata'] if 'strata' in setup.keys() else None}'):
        outer strata (last split):;{setup['cv_scheme'][0].strata}
        inner strata (last split):;{setup['cv_scheme'][1].strata}
    Folds:
        outer:;{setup['folds'][0]}
        inner:;{setup['folds'][1]}
    Repetitions:
        outer:;{setup['repeats'][0]}
        inner:;{setup['repeats'][1]}
    
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Aggregated scores:
    {scored}
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    return header


def augment_predictions(predi_series, samples_data_df, target=None, kind=''):
    samples_data_df.rename(columns={target: f'{target}_observed'}, inplace=True)
    cols = ['LON', 'LAT', 'SedDryBulkDensity'] + featurelist
    if target:
        cols = [f'{target}_observed'] + cols
    df = predi_series.to_frame().join(samples_data_df[cols])
    df['Type'] = kind
    return df


def aggregate_folds_only(NCV):
    ## Calculates the aggregated scores from NCV over the folds of every repetition, without aggregating over the repetitions.
    
    gb = NCV.groupby(['run_with', 'NCV_repetition'])
    
    content = [[an, s, g[f'test_{s}'].agg(a)]
               for an, a in Config.aggregators.items()
               for s in Config.scorers.keys()
               for gn, g in gb]
    
    index = pd.MultiIndex.from_tuples(
        [gn
         for a in Config.aggregators.values()
         for s in Config.scorers.keys()
         for gn, g in gb],
        names=['run_with', 'NCV_repetitions'])
    
    columns=['fold_aggregator', 'Scorer', 'Score_value']
    
    return pd.DataFrame(
        content,
        index=index,
        columns=columns,
    )
