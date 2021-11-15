import rpy2.robjects as robjects  # import R functions
import rpy2.robjects.packages as rpackages  # import rpy2's package module
from rpy2.robjects.packages import importr  # import R's package importer

# import R's "base" package
base = importr('base')
# import R's "utils" package
utils = importr('utils')
# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('pcaCoDa')
# R vector of strings
from rpy2.robjects.vectors import StrVector
# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

robjects.r('''  # TODO: add R code here (see at https://rdrr.io/cran/robCompositions/man/pcaCoDa.html) ''')
# The result of the function is returned to the Python Environment