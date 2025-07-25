import os
import sys

# Set R environment variables before importing rpy2
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.3.1'
os.environ['R_USER'] = os.environ.get('USERNAME', 'default')
os.environ['R_LIBS_USER'] = os.path.join(os.environ.get('USERPROFILE', ''), 'Documents', 'R', 'win-library', '4.3')

# Add R to PATH
r_bin_path = r'C:\Program Files\R\R-4.3.1\bin\x64'
if r_bin_path not in os.environ['PATH']:
    os.environ['PATH'] = r_bin_path + ';' + os.environ['PATH']

# Try to set shell to cmd instead of sh
os.environ['SHELL'] = 'cmd'
os.environ['COMSPEC'] = 'cmd.exe'

# Set additional environment variables that might help
os.environ['R_ARCH'] = '/x64'

print("R environment variables set:")
print(f"R_HOME: {os.environ.get('R_HOME')}")
print(f"R_USER: {os.environ.get('R_USER')}")
print(f"R_LIBS_USER: {os.environ.get('R_LIBS_USER')}")
print(f"SHELL: {os.environ.get('SHELL')}")
print(f"R_ARCH: {os.environ.get('R_ARCH')}")

try:
    print("\nTesting rpy2 import...")
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    print("✅ rpy2 imported successfully!")
    
    # Test basic R functionality
    print("\nTesting basic R functionality...")
    r_version = ro.r('R.version.string')
    print(f"R version: {r_version[0]}")
    
    print("✅ rpy2 is working correctly!")
    
except Exception as e:
    print(f"❌ Error importing rpy2: {e}")
    print("\nThis might require installing Rtools for Windows.")
    print("You can download it from: https://cran.r-project.org/bin/windows/Rtools/")
    sys.exit(1)
