import numpy as np
import pandas as pd
from scipy import stats

from KDE_settings import Config


def correlate_size_ranges(df):  # TODO: adapt correlation loop from ParticleAlign project
    df_r = pd.DataFrame(columns=['alpha', 'beta',
                                 'r_BDI', 'p_BDI',
                                 'r_IQI', 'p_IQI'])

    p = np.arange(Config.minParameterForBDI, Config.maxParameterForBDI, Config.stepParameterForBDI)
    for alpha in p:
        for beta in p:
            if alpha == 0 and beta ==0:
                pass
            else:
                df = make_BDI(df, alpha, beta)
                r_BDI = stats.pearsonr(df.particle_loss, df.BDI)
                r_IQI = stats.pearsonr(df.particle_loss, df.IQI)
                df_r.loc[len(df_r)] = [alpha, beta,
                                       r_BDI[0], r_BDI[1],
                                       r_IQI[0], r_IQI[1]]
                print(f'Running BDI optimisation with alpha =     {round(alpha, 2)}                        ', end="\r", flush=True)

    # print(df_r.loc[df_r.r_IQI == df_r.r_IQI.max()])
    print(df_r.loc[df_r.r_BDI == df_r.r_BDI.max()])
    bestAlpha, bestBeta = df_r.loc[df_r.r_BDI == df_r.r_BDI.max()].iloc[0, 0:2]

    return bestAlpha, bestBeta
