import numpy as np
import pandas as pd
from scipy import stats

from KDE_settings import Config


def range_conc_correlation(size_pdfs, sdd_MP_sed):  # TODO: adapt correlation loop from ParticleAlign project
    step = (Config.upper_size_limit - Config.lower_size_limit) / Config.kde_steps
    df_r = pd.DataFrame(columns=['lower_size', 'upper_size', 'r', 'p'])

    for i in size_pdfs.x_d:
        for j in size_pdfs.x_d[size_pdfs.x_d > i]:
            size_sum = size_pdfs.loc[(size_pdfs.x_d >= i) & (size_pdfs.x_d < j)].sum()
            size_sum.drop('x_d', inplace=True)
            range_prob = size_sum * step
            range_conc = range_prob * sdd_MP_sed.set_index('Sample').Concentration

            r = stats.pearsonr(range_conc, sdd_MP_sed.set_index('Sample').TOC)
            df_r.loc[len(df_r)] = [i, j, r[0], r[1]]
            print(f'Correlating TOC with size range            [{i},        {j}]                ', end="\r", flush=True)

    print(df_r.loc[df_r.r == df_r.r.max()])
    bestLower, bestUpper = df_r.loc[df_r.r == df_r.r.max()].iloc[0, 0:2]
    return bestLower, bestUpper, df_r


