from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# one-time execution to build & install the ffsimulator R package
# utils= importr('utils')
# utils.chooseCRANmirror(ind=1)
# utils.install_packages(StrVector(['devtools']))
# devtools = importr('devtools')
# devtools.install_github('dynastyprocess/ffpros')
# print(rpackages.isinstalled("ffpros"))

# # if success you can then import the package
ffsimulator = importr("ffsimulator")
ffscrapr = importr("ffscrapr")
ffpros = importr("ffpros")

def convert_r_to_py(df):
    with (ro.default_converter + pandas2ri.converter).context():
        new_df = ro.conversion.get_conversion().rpy2py(df)
        
    return new_df