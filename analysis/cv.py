import numpy as np
import pandas as pd

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
