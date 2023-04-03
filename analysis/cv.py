import numpy as np
import pandas as pd
from itertools import compress
import joblib
from datetime import datetime
import time
import threading

from statsmodels.sandbox.tools.cross_val import LeaveOneOut
from sklearn.model_selection import GridSearchCV, cross_validate, KFold, RepeatedKFold
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from sklearn import clone, set_config
# set_config(transform_output='pandas')  # only works for sklearn >= 1.2
from sklearn.utils import parallel_backend

import glm
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
    for train_index, test_index in LeaveOneOut(df.shape[0]):
        train = df.loc[train_index, :]
        test = df.loc[test_index, :]
        glm_res = glm.glm(train)
        predi = pd.concat([test.Sample, glm_res.predict(test).rename('pred')], axis=1)
        pred = pd.concat([pred, predi])

    target = df.loc[:, Config.glm_formula.split(' ~')[0]]  # isolate target variable (observed values)
    target_name = Config.glm_formula.split(' ~')[0]
    pred.loc[:, target_name] = target

    maxe = max_error(target, pred.pred)
    medae = median_absolute_error(target, pred.pred)
    medape = median_absolute_percentage_error(target, pred.pred)
    mae = mean_absolute_error(target, pred.pred)
    mape = mean_absolute_percentage_error(target, pred.pred)
    rmse = np.sqrt(mean_squared_error(target, pred.pred))
    r2 = r2_score(target, pred.pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)

    metrics = pd.DataFrame(columns=['Metric', 'Value', 'Info'])
    metrics.loc[0] = ['Max Error', maxe, f"Where: {pred.loc[np.abs(target - pred.pred ).idxmax(), 'Sample']}"]
    metrics.loc[1] = ['Median Absolute Error', medae, 'In units of target variable']
    metrics.loc[2] = ['Median Absolute Percentage Error', medape, 'Relative error: 0 (no error), 1 (100% misestimated), >1 (arbitrarily wrong)']
    metrics.loc[3] = ['Mean Absolute Error', mae, 'In units of target variable']
    metrics.loc[4] = ['Mean Absolute Percentage Error', mape, 'Relative error: 0 (no error), 1 (100% misestimated), >1 (arbitrarily wrong)']
    metrics.loc[5] = ['Root Mean Square Error', rmse, 'In units of target variable']
    metrics.loc[6] = ['R²', r2, 'From 1 (perfect prediction) and 0 (just as good as predicting the mean) to neg. infinity (arbitrarily wrong)']
    metrics.loc[7] = ['Adjusted R²', adj_r2, 'Like R² but takes into account sample and feature number. An additional feature is justified if it increases this metric.']
    return pred, metrics


def make_setup_dict(**kwargs):
    setup = kwargs  # dict of lists of NCV parameter settings: first element for outer, second for inner CV    
    if 'repeats' in setup.keys():
        if isinstance(setup['repeats'], int):
            setup['repeats'] = (setup['repeats'], setup['repeats'])
        else:
            setup['repeats'] = (setup['repeats'][0], max(int(setup['repeats'][1]), 1))
            
    if 'folds' in setup.keys():
        if isinstance(setup['folds'], int):
            setup['folds'] = (setup['folds'], setup['folds'])
    
    setup['cv_scheme'] = [
        KFold(  # using single KFold in outer which will be repeated manually in loop to extract the test set indices at each iteration
            n_splits = setup['folds'][0],
        ),
        RepeatedKFold(  # using RepeatedKFold for inner
            n_splits = setup['folds'][1],
            n_repeats = setup['repeats'][1],
        ),
    ]
    return setup


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

            setup['cv_scheme'][0].shuffle = True  # activate shuffling to yiels varying outer fold train / test splits
            setup['cv_scheme'][0].random_state = setup['cv_scheme'][1].random_state = i  # set different random states for each repetition

            outerCV = ncv(pipe, params, model_X, model_y, scorers, setup)

            get_median_cv_scores(outerCV)
            get_iqm_cv_scores(outerCV)

            rep = {'NCV_repetition': np.repeat(i, len(list(outerCV.values())[0]))}

            testsets = get_test_sets(setup['cv_scheme'][0], model_X)
            outerCV = {**rep, **testsets, **outerCV}

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
    if Config.select_best == 'median':
        avg_inner_test_scores = np.median(inner_test_scores, axis=0)
    elif Config.select_best == 'mean':
        avg_inner_test_scores = np.mean(inner_test_scores, axis=0)
    elif Config.select_best == 'iqm':
        avg_inner_test_scores = np.array([iqm(s) for s in inner_test_scores.T])
    else:
        raise ValueError(f'Can only select for best mean, median or iqm score. You supplied "{Config.select_best}".')
    return avg_inner_test_scores.argmax()
    

def get_median_cv_scores(outerCV):
    """
    Add median cross-validation scores to nested CV results.
    :param outerCV: result object of nested cross-validation
    """

    for outer_fold in range(len(outerCV['estimator'])):
        res = outerCV['estimator'][outer_fold].cv_results_
        res_df = pd.DataFrame(res)
        for k, v in Config.scorers.items():
            if f'rank_test_{k}' in res:
                res[f'rank_by_mean_test_{k}'] = res.pop(f'rank_test_{k}')
            res_df[f'median_test_{k}'] = res_df.filter(regex=f'^split._test_{k}').median(axis=1)
            if all(res_df[f'median_test_{k}'].isna()):
                res_df[f'rank_by_median_test_{k}'] = np.nan
            else:
                res_df[f'rank_by_median_test_{k}'] = res_df[f'median_test_{k}'].rank(ascending=False).astype(int)

            res[f'median_test_{k}'] = res_df[f'median_test_{k}'].to_numpy()
            res[f'rank_by_median_test_{k}'] = res_df[f'rank_by_median_test_{k}'].to_numpy()


def get_iqm_cv_scores(outerCV):
    """
    Add iqm cross-validation scores to nested CV results.
    :param outerCV: result object of nested cross-validation
    """

    for outer_fold in range(len(outerCV['estimator'])):
        res = outerCV['estimator'][outer_fold].cv_results_
        res_df = pd.DataFrame(res)
        for k, v in Config.scorers.items():
            if f'rank_test_{k}' in res:
                res[f'rank_by_mean_test_{k}'] = res.pop(f'rank_test_{k}')
            res_df[f'iqm_test_{k}'] = [iqm(s) for s in res_df.filter(regex=f'^split._test_{k}').values]
            if all(res_df[f'iqm_test_{k}'].isna()):
                res_df[f'rank_by_iqm_test_{k}'] = np.nan
            else:
                res_df[f'rank_by_iqm_test_{k}'] = res_df[f'iqm_test_{k}'].rank(ascending=False).astype(int)

            res[f'iqm_test_{k}'] = res_df[f'iqm_test_{k}'].to_numpy()
            res[f'rank_by_iqm_test_{k}'] = res_df[f'rank_by_iqm_test_{k}'].to_numpy()


def get_inner_test_scores(NCV):
    for idx in NCV.index:
        b = NCV.loc[idx].estimator.cv_results_[f'rank_by_{Config.select_best}_test_{Config.refit_scorer}'].argmin()  # get index of candidate which won this round
        for s in Config.scorers.keys():
            NCV.loc[idx, f'inner_test_{s}'] = NCV.loc[idx].estimator.cv_results_[f'{Config.select_best}_test_{s}'][b]


def get_test_sets(splitter, model_X):
    """
    Get the names and indices of samples in test sets.
    """
    return {
        'test_set_indices': [test_index for train_index, test_index in splitter.split(model_X)],
        'test_set_samples': [model_X.index.values[test_index] for train_index, test_index in splitter.split(model_X)],
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
        np.mean(rep_groups[f'test_{key}'].median()),  # Mean of repetitions' outer test median scores
        np.std(rep_groups[f'test_{key}'].median()),  # Standard deviation of repetitions' outer test median scores
        np.median(NCV[f'test_{key}']),  # Median of all outer test scores
        np.mean(rep_groups[f'test_{key}'].agg(iqm)),  # Mean of repetitions' outer test iqm scores
        np.std(rep_groups[f'test_{key}'].agg(iqm)),  # Standard deviation of repetitions' outer test iqm scores
        iqm(NCV[f'test_{key}']),  # IQM of all outer test scores
        np.mean(rep_groups[f'test_{key}'].mean()),  # Mean of repetitions' outer test mean scores
        np.std(rep_groups[f'test_{key}'].mean()),  # Standard deviation of repetitions' outer test mean scores
        np.mean(NCV[f'test_{key}']),  # Mean of all outer test scores
            ] for key in Config.scorers
    }, index=[
        f'Mean_of_medians_of_{r}_repetitions',
        f'Stdev_of_medians_of_{r}_repetitions',
        f'Median_of_all_{r * f}_folds_together',
        f'Mean_of_IQMs_of_{r}_repetitions',
        f'Stdev_of_IQMs_of_{r}_repetitions',
        f'IQM_of_all_{r * f}_folds_together',
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
        ensemble_idx = rensembled.groupby(['NCV_repetition', 'OuterCV_fold'])[f'inner_test_{Config.refit_scorer}'].agg(pd.Series.idxmax).to_list()
        NCV = pd.concat([NCV, rensembled.loc[ensemble_idx]])
    return NCV


def make_header(NCV, setup, starttime, time_needed, droplist, model_X, num_feat, feature_candidates_list, scaler, regressor_params):
    ## Create a dataframe of aggregated scores and standard deviations
    scored = aggregate_scores(NCV, setup)    
    scored = scored.round(4).to_csv(sep=';') if Config.ncv_mode=='competitive' else 'COMPARATIVE RUN: no common scores available. Look at the individual scores of each model class!'
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

    Number of features per candidate (min, max):;{num_feat}
    Total number of feature combinations tested:;{len(feature_candidates_list)}
    Available features: {len(featurelist)};{featurelist}
    Restricted combinations (mutal exclusive features, exclusive keywords):;{Config.mutual_exclusive};{Config.exclusive_keywords}

    Scaler:;{scaler}
    Regressors:;{[[g['regressor'][0].__class__.__name__, [{k: v} for k, v in g.items() if k.startswith('regressor__')]] for g in regressor_params]}
    Winning candidate of gridsearch (inner CV loop) determined by best;{Config.select_best} {Config.refit_scorer} score; Note: results within each repetition are sorted by this!

    CV schemes:
        inner:;{setup['cv_scheme'][1]}
        outer:;{setup['cv_scheme'][0]}
    Repetitions:
        inner:;{setup['repeats'][1]}
        outer:;{setup['repeats'][0]}
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Aggregated scores:
    {scored}
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    return header