import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa, pcoa_biplot


def PCA(df, num=0.95):  # TODO: rephrase docstring as it is not specific to sediments (now also used for composition)
    """
    Takes a df of sediment frequencies in size bins (rows = samples, columns = size bins)
    and returns a df of principle components. If the number of PCs is not provided chosen,
    it will be automatically determined so that >=.95 of information is retained.
    """
    
    x = StandardScaler().fit_transform(df.values)
    
    pca = PCA(n_components=num)
    principalComponents = pca.fit_transform(x)
    
    pc_names = ['PC' + str(z + 1) for z in range(pca.n_components_)]  # create as many labels for columns 'PC_#' as there are principle components  
    
    pc_df = pd.DataFrame(data = principalComponents, index = df.index, columns = pc_names)
    
    pc_loadings = pd.DataFrame(pca.components_.T, columns=pc_names, index=df.columns)  # this gives the eigenvalues which say how much each variable contributed to a PC. Add  "* np.sqrt(pca.explained_variance_)" to yield the correlations between each variable and PC instead. From: https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html

    print('PCA explained variance:', pca.explained_variance_ratio_)
    print('PCA total explained', sum(pca.explained_variance_ratio_))

    return pc_df, pc_loadings, pca.explained_variance_ratio_


def PCOA(df, num = 3):  # TODO: rephrase docstring as it is not specific to sediments (now also used for composition)
    """
    | Takes a dataframe of sediment grain sizes (compositional data), with rows = samples, columns = size bins.
    | num is the number of principal coordinates to return (default = 3).
    | Calculates a Bray-Curtis dissimilarity matrix for the PCoA.
    | Calculates a Principal Coordinate Analysis and returns its scores.
    | Returns the requested number of PCos as a pandas df and prints the explained proportions.
    """

    df = df.div(df.sum(axis=1), axis=0)  # ensure closed compositional data
    data = df.values
    ids = df.index.to_list()
    bc_dm = beta_diversity("braycurtis", data, ids)  # create Bray-Curtis dissimilarity matrix
    bc_pc = pcoa(bc_dm)  # Principal Coordinate Analysis, generates a skbio.stats.ordination.Ordination object
    bc_pc = pcoa_biplot(bc_pc, df)  # calculates the `features` of the pcoa ordination object
    
    expl = bc_pc.proportion_explained.iloc[0:num]
    print('PCoA: Proportion explained:', '\r\n', expl, '   PCoA Total:', expl.sum())
    
    scor = bc_pc.samples.iloc[:, 0:num] # returns the scores of first num_coords principal coordinates
    load = bc_pc.features.iloc[:, 0:num] # returns the loadings of the first num_coords principal coordinates
    load = load.loc[np.sqrt((load**2).sum(axis=1)).sort_values(ascending=False).index]  # sorts loadings by magnitude
    
    return scor, load, expl
