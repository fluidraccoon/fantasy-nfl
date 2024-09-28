# Import the necessary rpy2 functions and packages
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r
import pandas as pd

# # Activate the pandas2ri converter to handle Pandas DataFrames <-> R DataFrames
# pandas2ri.activate()

# # Import R packages from the ffpros package
ffpros = importr('ffpros')

# # Define the page, sport, and metadata parameters to pass to fp_rankings
# page = "dynasty-overall"  # Example page, you can change it
# year = 2016
# sport = "nfl"
# include_metadata = False

# # Call the fp_rankings function from the ffpros R package using rpy2
# fp_rankings_result = ro.r['fp_rankings'](page, year=year, sport=sport, include_metadata=include_metadata)

# # If the result is a data frame (likely a tibble in R), convert it back to pandas DataFrame
# fp_rankings_df = pandas2ri.rpy2py(fp_rankings_result)

# # Display the pandas DataFrame in Python
# print(pandas2ri.rpy2py(fp_rankings_result))

def fetch_fp_rankings(page, season):
    try:
        # Call the fp_rankings function
        result = r['fp_rankings'](page, year=season)
        # Convert the R result (tibble) to pandas DataFrame
        return pandas2ri.rpy2py(result)
    except Exception as e:
        print(f"Error fetching rankings for {page}, {season}: {e}")
        return pd.DataFrame()

df = fetch_fp_rankings("qb-cheatsheets", 2023)
print(df)