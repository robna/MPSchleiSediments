from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa, pcoa_biplot
import matplotlib.pyplot as plt
import numpy as np

def sed_pcoa(df, num_coords = 3):  # TODO: rename function and rephrase docstring as it is not specific to sediments (now also used for composition)
    """
    | Takes a dataframe of sediment grain sizes (compositional data), with rows = samples, columns = size bins.
    | num_coords is the number of principal coordinates to return (default = 3).
    | Calculates a Bray-Curtis dissimilarity matrix for the PCoA.
    | Calculates a Principal Coordinate Analysis and returns its scores.
    | Returns the requested number of PCos as a pandas df and prints the explained proportions.
    """
    
    data = df.values
    ids = df.index.to_list()
    bc_dm = beta_diversity("braycurtis", data, ids)  # create Bray-Curtis dissimilarity matrix
    bc_pc = pcoa(bc_dm)  # Principal Coordinate Analysis, generates a skbio.stats.ordination.Ordination object
    bc_pc = pcoa_biplot(bc_pc, df)  # calculates the `features` of the pcoa ordination object
    
    exp = bc_pc.proportion_explained.iloc[0:num_coords]
    print('Proportion explained:', '\r\n', exp, '   Total: exp.sum', exp.sum())
    
    return bc_pc.samples.iloc[:, 0:num_coords], bc_pc.features.iloc[:, 0:num_coords]


def biplot(ordination_results, c=None, y=None, n_feat=None, figsize=(10,6.18)):  # function adapted from here: https://github.com/biocore/scikit-bio/issues/1710#issuecomment-647791743
    """
    Adapted from `erdogant/pca` on GitHub:
    https://github.com/erdogant/pca/blob/7b1249d9e215e93de5ba606efaa9dd8c0fd5928c/pca/pca.py#L334
    to use skbio OrdinationResults object

    Create the Biplot based on model.
    Parameters
    ----------
    figsize : (float, float), optional, default: None
        (width, height) in inches. If not provided, defaults to rcParams["figure.figsize"] = (10,8)
    Returns
    -------
    tuple containing (fig, ax)
    """
    # Pre-processing
    # y, topfeat, n_feat = self._fig_preprocessing(y, n_feat)
    # Figure
    with plt.style.context("seaborn-white"):

        fig, ax = plt.subplots(figsize=figsize)
        # Plot projection
        ax.scatter(ordination_results.samples.iloc[:,0], ordination_results.samples.iloc[:,1], c=c, edgecolor="black", linewidth=0.618)
        ax.set_xlabel("PCoA.1 (%0.3f)"%(ordination_results.proportion_explained["PC1"]), fontsize=15)
        ax.set_ylabel("PCoA.2 (%0.3f)"%(ordination_results.proportion_explained["PC2"]), fontsize=15)

        # Gather loadings from the top features from topfeat
        loadings = ordination_results.features.T
        if n_feat == None:
            n_feat = loadings.shape[1]

        # xvector = self.results['loadings'][topfeat.index.values].iloc[0,:]
        # yvector = self.results['loadings'][topfeat.index.values].iloc[1,:]
        xvector = loadings.iloc[0,:]
        yvector = loadings.iloc[1,:]

        # Use the PCs only for scaling purposes
        # xs = self.results['PC'].iloc[:,0].values
        # ys = self.results['PC'].iloc[:,1].values
        xs = ordination_results.samples.iloc[:,0]
        ys = ordination_results.samples.iloc[:,1]

        # Boundaries figures
        maxR = np.max(xs)*0.8
        maxL = np.min(xs)*0.8
        maxT = np.max(ys)*0.8
        maxB = np.min(ys)*0.8

        np.where(np.logical_and(np.sign(xvector)>0, (np.sign(yvector)>0)))

        # Plot and scale values for arrows and text
        # scalex = 1.0 / (self.results['loadings'][topfeat.index.values].iloc[0,:].max() - self.results['loadings'][topfeat.index.values].iloc[0,:].min())
        # scaley = 1.0 / (self.results['loadings'][topfeat.index.values].iloc[1,:].max() - self.results['loadings'][topfeat.index.values].iloc[1,:].min())
        scalex = 1.0 / (loadings.iloc[0,:].max() - loadings.iloc[0,:].min())
        scaley = 1.0 / (loadings.iloc[1,:].max() - loadings.iloc[1,:].min())
        # Plot the arrows
        for i in range(0, n_feat):
            # arrows project features (ie columns from csv) as vectors onto PC axes
            newx = xvector[i] * scalex
            newy = yvector[i] * scaley
            # figscaling = np.abs([np.abs(xs).max() / newx, np.abs(ys).max() / newy])
            # figscaling = figscaling.max()
            # newx = newx * figscaling * 0.1
            # newy = newy * figscaling * 0.1
            newx = newx * 500
            newy = newy * 500

            # Max boundary right x-axis
            if np.sign(newx)>0:
                newx = np.minimum(newx, maxR)
            # Max boundary left x-axis
            if np.sign(newx)<0:
                newx = np.maximum(newx, maxL)
            # Max boundary Top
            if np.sign(newy)>0:
                newy = np.minimum(newy, maxT)
            # Max boundary Bottom
            if np.sign(newy)<0:
                newy = np.maximum(newy, maxB)

            ax.arrow(0, 0, newx, newy, color='r', width=0.001, head_width=0.01, alpha=0.382)
            ax.text(newx * 1.25, newy * 1.25, xvector.index.values[i], color='red', ha='center', va='center')

        # plt.show()
        return(fig, ax)    
    