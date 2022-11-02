import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed, cpu_count

from statsmodels.sandbox.tools.cross_val import LeaveOneOut
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, r2_score

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
    mae = mean_absolute_error(target, pred.pred)
    rmse = np.sqrt(mean_squared_error(target, pred.pred))
    r2 = r2_score(target, pred.pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)

    metrics = pd.DataFrame(columns=['Metric', 'Value', ''])
    metrics.loc[0] = ['Max Error', maxe, pred.loc[np.abs(target - pred.pred ).idxmax(), 'Sample']]
    metrics.loc[1] = ['Mean Absolute Error', mae, '']
    metrics.loc[2] = ['Root Mean Square Error', rmse, '']
    metrics.loc[3] = ['R²', r2, '']
    metrics.loc[4] = ['Adjusted R²', adj_r2, '']

    return pred, metrics


def generate_feature_sets(featurelist, mutual_exclusive, exclusive_keywords, num_feat='all', n_jobs=1):
    """
    Generate all possible feature combinations of a given size.
    :param featurelist: list of all available features
    :param mutual_exclusive: list of lists of features that are mutually exclusive
    :param exclusive_keywords: list of keywords, for each of which maximum one feature containing it may be present in any given combination
    :param num_feat: number of features in each set, can be an integer, or a tuple (min, max) or 'all' (default)
    :return: list of feature sets as pandas index objects
    """

    if isinstance(featurelist, pd.DataFrame):
        featurelist = featurelist.columns

    if num_feat == 'all':
        min_num , max_num = (1, len(featurelist))
    elif isinstance(num_feat, int):
        min_num = max_num = num_feat
    elif isinstance(num_feat, (list, tuple)):
        min_num, max_num = num_feat
    else:
        raise ValueError('num_feat must be an integer, a tuple or "all".')
    
    if n_jobs == 1:
        fl = [list(combinations(featurelist, l)) for l in range(min_num, max_num+1)]
    else:
        fl = Parallel(n_jobs=n_jobs, verbose=1)(delayed(lambda x: list(combinations(featurelist, x)))(x) for x in range(min_num, max_num+1))

    # flatten the list of lists and remove combinations containing mutual exclusive features
    return [
        item
            for sublist in fl
            for item in sublist
        if not any(all(pd.Series(ex_feats).isin(item))
            for ex_feats in mutual_exclusive)
        and sum(keyword in feat 
            for keyword in exclusive_keywords
            for feat in item)
        <= len(exclusive_keywords)
        ]
