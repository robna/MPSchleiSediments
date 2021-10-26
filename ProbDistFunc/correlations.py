import numpy as np
import pandas as pd
from scipy import stats

from KDE_settings import Config


def predictorcorr(df_range_conc, predictors, col_name):
    df_r = pd.DataFrame(columns=['lower_size', 'upper_size', 'r', 'p'])
    
    for index, range_conc in df_range_conc.iterrows():
        lower, upper = index.split('_')
        if range_conc.max() > 1e-32:  # don't proceed with correlation if all probabilities in current size slice are practically zero
            r = stats.pearsonr(range_conc, predictors[col_name])
        else: r = [np.nan, np.nan]    
        df_r.loc[len(df_r)] = [lower, upper, r[0], r[1]]
    
        print(f'Correlating {col_name} with size range            [{lower},        {upper}]                ', end="\r", flush=True)
    
    print(df_r.loc[df_r.r == df_r.r.max()], end="\r", flush=True)
    bestLower, bestUpper = df_r.loc[df_r.r == df_r.r.max()].iloc[0, 0:2]
    return bestLower, bestUpper, df_r




def crosscorr(datax, datay):
    """
    inspired from:
    https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
    datax, datay : pandas.Series objects of equal length
    
    Returns
    -------
    best: highest correlation coefficients and corresponding shift
    df_r: Dataframe containing shift steps and resulting pearson correlation coeffs between datax and datay
    """
    
    lags = range(-len(datax), len(datax)+1)
    df_r = pd.DataFrame(lags, columns = ['shifted'])
    
    r = [datax.corr(datay.shift(lag)) for lag in lags]
    df_r['pearson'] = r
    
    best = df_r.loc[df_r['pearson'] == df_r['pearson'].max()]

    return best, df_r


def rangecorr(mp, sed):
    """Calculate a correlation matrix containing Pearson correlation
    coefficients for all combinations of any original or summed bins of MP and sediments.
    """

    corrMat = np.corrcoef(mp, sed)
    corrMat = corrMat[:len(mp), len(sed):]  # only take upper right quadrant of correlation matrix

    corrMatDF = pd.DataFrame(corrMat, index=mp.index, columns=sed.index)  # turn np array into df
    
    corrMatDF.index.name = 'MP'
    corrMatDF.columns.name = 'SED'
    
    return corrMatDF
