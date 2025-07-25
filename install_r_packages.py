import os

# Set R environment variables before importing rpy2 (required for Windows)
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.3.1'
os.environ['R_USER'] = os.environ.get('USERNAME', 'default')
os.environ['R_LIBS_USER'] = os.path.join(os.environ.get('USERPROFILE', ''), 'Documents', 'R', 'win-library', '4.3')

# Add R to PATH
r_bin_path = r'C:\Program Files\R\R-4.3.1\bin\x64'
if r_bin_path not in os.environ['PATH']:
    os.environ['PATH'] = r_bin_path + ';' + os.environ['PATH']

# Set additional environment variables for Windows compatibility
os.environ['SHELL'] = 'cmd'
os.environ['COMSPEC'] = 'cmd.exe'
os.environ['R_ARCH'] = '/x64'

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Ensure R library directory exists
r_libs_path = os.environ['R_LIBS_USER']
os.makedirs(r_libs_path, exist_ok=True)

# Set library path in R (convert backslashes to forward slashes for R)
r_libs_path_r = r_libs_path.replace('\\', '/')
ro.r(f'.libPaths("{r_libs_path_r}")')

print("Installing ffscrapr R package...")
utils = importr('utils')
utils.chooseCRANmirror(ind=1)  # Choose CRAN mirror
utils.install_packages('ffscrapr', repos='https://cloud.r-project.org')
print("ffscrapr installation complete!")

# Test import
try:
    ffscrapr = importr('ffscrapr')
    print("✅ ffscrapr imported successfully!")
except Exception as e:
    print(f"❌ Error importing ffscrapr: {e}")
