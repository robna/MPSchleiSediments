import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def sed_pca(df):
    """
    Takes a df of sediment frequencies in size bins (rows = samples, columns = size bins)
    and returns a df of principle components. The number of PCs is chosen automatically,
    so that >=.95 of information is retained.
    """
    
    x = StandardScaler().fit_transform(df.values)
    
    pca = PCA(.95)
    principalComponents = pca.fit_transform(x)
    
    pc_names = ['PC' + str(z + 1) for z in range(pca.n_components_)]  # create as many labels for columns 'PC_#' as there are principle components  
    
    pc_df = pd.DataFrame(data = principalComponents, index = df.index, columns = pc_names)
    pc_exp = pca.explained_variance_ratio_
    pc_loadings = pca.components_.T  # this gives the eigenvalues which say how much each variable contributed to a PC. Add  "* np.sqrt(pca.explained_variance_)" to yield the correlations between each variable and PC instead. From: https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html

    return pc_df, pc_exp, pc_loadings
