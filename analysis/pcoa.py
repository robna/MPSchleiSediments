from skbio.diversity.beta import pw_distances
from skbio.stats.ordination import PCoA


def sed_pcoa(data: pd.Dataframe):
    """
    | Takes a datframe of sediment grain sizes (compositional data), with rows = samples, columns = size bins.
    | Calculates a distance matrix using the distance measure defined in Config.
    | Calculates a Principal COordinate Analysis and returns its scores.
    """
    bc_dm = pw_distances(data, ids, "braycurtis")
    bc_pc = PCoA(bc_dm)
    
    return bc_dm.scores()
    
    