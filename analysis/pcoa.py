from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa


def sed_pcoa(df, num_coords = 3):
    """
    | Takes a datframe of sediment grain sizes (compositional data), with rows = samples, columns = size bins.
    | num_coords is the number of principal coordinates to return (default = 3).
    | Calculates a distance matrix using the distance measure defined in Config.
    | Calculates a Principal COordinate Analysis and returns its scores.
    | Returns the requested number of PCs as apandas df and prints their explained proportion.
    """
    
    data = df.values
    ids = df.index.to_list()
    bc_dm = beta_diversity("braycurtis", data, ids)
    bc_pc = pcoa(bc_dm)
    
    exp = bc_pc.proportion_explained.iloc[0:num_coords]
    print('Proportion explained:', '\r\n', exp, '   Total: exp.sum', exp.sum())
    
    return bc_pc.samples.iloc[:, 0:num_coords]
    
    