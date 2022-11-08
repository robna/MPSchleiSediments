import numpy as np
import pandas as pd
import pickle
from pathlib import Path
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
        featurelist = featurelist.columns

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
    
    if Path(f'../data/feature_candidates_list_min{min_num}_max{max_num}.pkl').exists():
        with open(f'../data/feature_candidates_list_min{min_num}_max{max_num}.pkl', 'rb') as f:
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
        # with open(f'../data/feature_candidates_list_min{min_num}_max{max_num}.txt', 'w') as f:
        #     for line in fl:
        #         f.write(f"{line}\n")
        with open(f'../data/feature_candidates_list_min{min_num}_max{max_num}.pkl', 'wb') as f:
            pickle.dump(fl, f)
    return fl


def best_median_score(cv_results):
    """
    Find the best median score from a cross-validation result dictionary.
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
    median_inner_test_scores = np.median(inner_test_scores, axis=0)
    return median_inner_test_scores.argmax()
    

    def get_median_cv_scores(outerCV):
        """
        Add median cross-validation scores to nested CV results.
        :param outerCV: result object of nested cross-validation
        """

        for outer_fold in range(len(outerCV['estimator'])):
            res = outerCV['estimator'][outer_fold].cv_results_
            res_df = pd.DataFrame(res)
            for k, v in Config.scoring.items():
                res[f'rank_by_mean_test_{k}'] = res.pop(f'rank_test_{k}')
                res_df[f'median_test_{k}'] = res_df.filter(regex=f'^split._test_{k}').median(axis=1)
                res_df[f'rank_by_median_test_{k}'] = res_df[f'median_test_{k}'].rank(ascending=False).astype(int)

                res[f'median_test_{k}'] = res_df[f'median_test_{k}'].to_numpy()
                res[f'rank_by_median_test_{k}'] = res_df[f'rank_by_median_test_{k}'].to_numpy()
                