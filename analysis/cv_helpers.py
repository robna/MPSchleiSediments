import numpy as np
import pandas as pd
import json
from hashlib import md5
import pickle
from pathlib import Path
from itertools import combinations
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

from helpers import tqdm_joblib


def SelectFeatures(model_X, feature_set, feature_sets):
    return model_X.loc[:, feature_sets[feature_set]]
    ### Maybe better alternative instead of using custom function with FunctionTransformer, use a class... something like:
    # from sklearn.base import BaseEstimator, TransformerMixin
    # class FeatureSubsetSelector(BaseEstimator, TransformerMixin):
    #     def __init__(self, feature_subsets):
    #         self.feature_subsets = feature_subsets
            
    #     def fit(self, X, y=None):
    #         return self
        
    #     def transform(self, X):
    #         selected_features = []
    #         for subset in self.feature_subsets:
    #             selected_features.extend(subset)
    #         return X[selected_features]


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
        
    md5_tail = md5(json.dumps([featurelist, mutual_exclusive, exclusive_keywords], sort_keys=True).encode('utf-8')).hexdigest()[-5:]  # get the hash of featurelist, mutual_exclusive and exclusive_keywords
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


def median_absolute_percentage_error(y_true, y_pred, epsilon=np.finfo(np.float64).eps):
    """
    Median Absolute Percentage Error
    :param y_true: true values
    :param y_pred: predicted values
    :param epsilon: a small positive value to avoid division by zero
    :return: Median Absolute Percentage Error
    """
    return np.median(np.abs(y_true - y_pred) / np.maximum(epsilon, np.abs(y_true)))


def inter_rank(NCV, refit_scorer):
    if 'Inter-Rep-rank' not in NCV.index.names:
        ## Make ranking among all folds (across reps)
        NCV['Inter-Rep_rank'] = NCV[f'test_{refit_scorer}'].rank(ascending=False).astype(int)
        NCV.set_index('Inter-Rep_rank', append=True, inplace=True)
    return NCV


def unnegate(NCV, scorers):
    # saving a list of scores which have turned up with negated values in the results
    negated = [ 
        key for key, val in scorers.items()
        if any(sub in str(val)
        for sub in ['neg_', 'greater_is_better=False'])
    ]
    ## Convert negated scores back to normal
    NCV[NCV.filter(regex='|'.join(negated)).columns] *= -1
    return NCV


def fix_feature_combis(params, feature_candidates_list, lin_combis=[2,3], tree_combis=[5]):
    '''
    Small helper function to use different model classes,
    i.e. linear (TweedieRegressor and XGB with linear booster)
         and tree-based (XGB with tree booster and RF)
    with different lengths of allowed feature combinations.
    In default it will prepare the gridsearch param-grid such
    that the linear models will run on all possible combinations
    of 2 or 3 features, while tree models will run on all
    combinations of 5 features.
    '''
    lin_fsets = [fset for fset in feature_candidates_list if len(fset) in lin_combis]
    lin_flst = [{'feature_set': i} for i in range(len(lin_fsets))]
    [d.update({'feature_sets': lin_fsets}) for d in lin_flst]

    tree_fsets = [fset for fset in feature_candidates_list if len(fset) in tree_combis]
    tree_flst = [{'feature_set': i} for i in range(len(tree_fsets))]
    [d.update({'feature_sets': tree_fsets}) for d in tree_flst]

    for i, pg in enumerate(params):
        rn = pg['regressor'][0].__class__.__name__
        if rn == 'TweedieRegressor':
            l = len(params[i]['preprocessor__selector__kw_args'])
            params[i]['preprocessor__selector__kw_args'] = lin_flst
            l2 = len(params[i]['preprocessor__selector__kw_args'])
            print(f'{rn}: Number of feature sets changed from {l} to {l2}')
        elif rn == 'XGBRegressor':
            if pg['regressor__booster'][0] == 'gblinear':
                l = len(params[i]['preprocessor__selector__kw_args'])
                params[i]['preprocessor__selector__kw_args'] = lin_flst
                l2 = len(params[i]['preprocessor__selector__kw_args'])
                print(f'{rn} {pg["regressor__booster"][0]}: Number of feature sets changed from {l} to {l2}')
            else:
                l = len(params[i]['preprocessor__selector__kw_args'])
                params[i]['preprocessor__selector__kw_args'] = tree_flst
                l2 = len(params[i]['preprocessor__selector__kw_args'])
                print(f'{rn} {pg["regressor__booster"][0]}: Number of feature sets changed from {l} to {l2}')
        else:
            l = len(params[i]['preprocessor__selector__kw_args'])
            params[i]['preprocessor__selector__kw_args'] = tree_flst
            l2 = len(params[i]['preprocessor__selector__kw_args'])
            print(f'{rn}: Number of feature sets changed from {l} to {l2}')
    return params
    

def ensemble_predict(esti, X, n_jobs=-1, verbose=False):
    '''
    esti: dataframe with columns 'estimator_refit_on_all' (trained models) and 'features' (lists of feature names used by the respective model)
    X: dataframe with columns of features
    returns: df of predictions
    '''
    # p = [
    #     est['estimator_refit_on_all'].predict(X.loc[[P], est.features])
    #     for _, est in esti.iterrows()
    #     for P in X.index
    #     ]
    with tqdm_joblib(tqdm(desc=f'Predicting {len(X)} samples with {len(esti)} members', total=len(X)*len(esti))) as progress_bar:
        p = Parallel(n_jobs=-1, verbose=verbose, backend='threading')(
            delayed(est.estimator_refit_on_all.predict)(X.loc[[s], est.features])
            for _, est in esti.iterrows()
            for s in X.index
            )
    p = np.array(p).reshape(len(esti), len(X))
    return pd.DataFrame(p, index=esti.index, columns=X.reset_index().Sample)
    
    
def aggregate_predictions(pred_df, ensemble_aggregator, target):
    '''
    pred_df: dataframe of ensemble predictions (row = members, columns = samples)
    ensemble_aggregator: tuple: either (rep_aggregator,) if ensemble was previously already reduced, or (rep_aggregator, folds_aggregator)
    '''
    if len(ensemble_aggregator) > 1:
        pred_df = pd.DataFrame([g.agg(ensemble_aggregator[1]) for _, g in pred_df.groupby('NCV_repetition')]) #.agg(Config.aggregators[rep_aggregator])
    return pd.Series([ensemble_aggregator[0](s) for _, s in pred_df.items()], index=pred_df.columns, name=f'{target}_predicted')


def check_testset_duplicates(NCV):
    ## takes the NCV dataframe and checks for duplicates in the test sets
    # if there are multiple runs in NCV (i.e. in comparative mode),
    # the test sets would be identical between runs, so we only need to check one
    unique_runs = NCV.index.get_level_values('run_with').unique()
    # if there is only one run, slicing for unique_runs[0] would return the same dataframe
    df = NCV.loc[NCV.index.get_level_values('run_with') == unique_runs[0]]
    testset_length_freq = df.outer_test_set_samples.apply(lambda x: len(x)).value_counts()

    ## Check for duplicates in test sets
    all_testsets = pd.DataFrame.from_dict([r for r in df.outer_test_set_samples])
    dup_testsets = all_testsets[all_testsets.duplicated()]
    
    return dup_testsets, testset_length_freq
