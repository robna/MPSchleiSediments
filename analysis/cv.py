import numpy as np
import pandas as pd
import json
from hashlib import md5
import pickle
from pathlib import Path
from itertools import combinations
from joblib import Parallel, delayed, cpu_count

from statsmodels.sandbox.tools.cross_val import LeaveOneOut
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error

import glm
from settings import Config


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


def SelectFeatures(model_X, feature_set, feature_sets):
    return model_X.loc[:, feature_sets[feature_set]]


def generate_feature_sets(featurelist, mutual_exclusive, exclusive_keywords, num_feat='all', n_jobs=1, save=False):
    """
    Generate all possible feature combinations of a given size.
    :param featurelist: list of all available features
    :param mutual_exclusive: list of lists of features that are mutually exclusive
    :param exclusive_keywords: list of keywords, for each of which maximum one feature containing it may be present in any given combination
    :param num_feat: number of features in each set, can be an integer, or a tuple (min, max) or 'all' (default)
    :return: list of feature sets
    """

    if isinstance(featurelist, pd.DataFrame):
        featurelist = featurelist.columns.to_list()

    if num_feat == 'all':
        min_num , max_num = (1, len(featurelist))
    elif isinstance(num_feat, int):
        min_num = max_num = num_feat
    elif isinstance(num_feat, (list, tuple)):
        if len(num_feat) == 2:
            min_num, max_num = num_feat
        elif len(num_feat) == 1:
            min_num = num_feat[0]
            max_num = len(featurelist)
        else:
            raise ValueError(f'num_feat as list or tuple may only have two elements (min_num, max_num) OR one element (min_num,) where max_num = len(featurelist). You supplied {type(num_feat)} of length {len(num_feat)}.')
    else:
        raise ValueError('num_feat must be an integer, a tuple or "all".')
        
    md5_tail = md5(json.dumps(featurelist, sort_keys=True).encode('utf-8')).hexdigest()[-5:]  # get the hash of featurelist
    flp = Path(f'../data/exports/cache/feature_candidates_list_min{min_num}_max{max_num}_HASH{md5_tail}.pkl')  # feature list path
    if flp.exists():
        with open(flp, 'rb') as f:
            fl =  pickle.load(f)
            print(f'Loaded feature candidates list from file: {f.name}')
            print(f'Number of feature sets: {len(fl)}')
            return fl

    if n_jobs == 1:
        nfl = [list(combinations(featurelist, l)) for l in range(min_num, max_num+1)]
        # flatten the list of lists and remove combinations containing mutual exclusive features
        fl = [
            list(item)
                for sublist in nfl
                for item in sublist
            if not any(all(pd.Series(ex_feats).isin(item))
                for ex_feats in mutual_exclusive)
            and sum(keyword in feat 
                for keyword in exclusive_keywords
                for feat in item)
            <= len(exclusive_keywords)
            ]

    else:      
        nfl = Parallel(n_jobs=n_jobs, verbose=1)(delayed(lambda x: list(combinations(featurelist, x)))(x) for x in range(min_num, max_num+1))
        # flatten and return, using joblib Parallel:
        fl = Parallel(n_jobs=n_jobs, verbose=1)(delayed(list)(item)
                for sublist in nfl
                for item in sublist
            if not any(all(pd.Series(ex_feats).isin(item))
                for ex_feats in mutual_exclusive)
            and sum(keyword in feat
                for keyword in exclusive_keywords
                for feat in item)
            <= len(exclusive_keywords))
    
    print(f'Combination generation finished: {len(fl)} combinations generated.')
    
    if save:
        with open(flp, 'wb') as f:
            pickle.dump(fl, f)
    return fl


def iqm(n):
    """
    Returns the interquartile mean input array n
    Found here: https://codegolf.stackexchange.com/a/86469
    """
    return sum(sorted(list(n)*4)[len(n):-len(n)])/len(n)/2


def best_scored(cv_results):
    """
    Find the best median / mean / interquartile mean score from a cross-validation result dictionary.
    :param cv_results: dictionary of cross-validation results
    :return: index of best median score
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
        raise ValueError(f'Can only select for best mean, median or iqm score. You supplied {Config.select_best}')
    return avg_inner_test_scores.argmax()
    

def get_median_cv_scores(outerCV):
    """
    Add median cross-validation scores to nested CV results.
    :param outerCV: result object of nested cross-validation
    """

    for outer_fold in range(len(outerCV['estimator'])):
        res = outerCV['estimator'][outer_fold].cv_results_
        res_df = pd.DataFrame(res)
        for k, v in Config.scoring.items():
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
        for k, v in Config.scoring.items():
            if f'rank_test_{k}' in res:
                res[f'rank_by_mean_test_{k}'] = res.pop(f'rank_test_{k}')
            res_df[f'iqm_test_{k}'] = [iqm(s) for s in res_df.filter(regex=f'^split._test_{k}').values]
            if all(res_df[f'iqm_test_{k}'].isna()):
                res_df[f'rank_by_iqm_test_{k}'] = np.nan
            else:
                res_df[f'rank_by_iqm_test_{k}'] = res_df[f'iqm_test_{k}'].rank(ascending=False).astype(int)

            res[f'iqm_test_{k}'] = res_df[f'iqm_test_{k}'].to_numpy()
            res[f'rank_by_iqm_test_{k}'] = res_df[f'rank_by_iqm_test_{k}'].to_numpy()


def median_absolute_percentage_error(y_true, y_pred, epsilon=np.finfo(np.float64).eps):
    """
    Median Absolute Percentage Error
    :param y_true: true values
    :param y_pred: predicted values
    :param epsilon: a small positive value to avoid division by zero
    :return: Median Absolute Percentage Error
    """
    return np.median(np.abs(y_true - y_pred) / np.maximum(epsilon, np.abs(y_true)))
