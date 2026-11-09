"""Machine-specific paths: template for local_paths.py.

Copy this file to ``local_paths.py`` in the same folder (src/common/) and
set every path for your machine. ``local_paths.py`` is ignored by git, so
your paths are never committed. All scripts read their paths from it.

Only two kinds of paths go here: the project folder (BASE_PATH) and the
large, multi-file external datasets that usually live elsewhere (for example
a shared copy on a cluster). Everything else is found at a fixed place inside
BASE_PATH: inputs and derived files in BASE_PATH/data, model runs and analysis
outputs in BASE_PATH/results.
"""
from pathlib import Path

# Project folder that holds data/ and results/
BASE_PATH = Path('/path/to/project')

# CHELSA V2.1 daily GeoTIFFs: one sub-folder per variable (pr, tas, rsds, tasmin, tasmax)
CHELSA_PATH = Path('/path/to/chelsa')

# Raw ERA5-Land NetCDF files: sd/sd_ERA5-Land_<year>0101-<year>1231.nc
ERA5LAND_PATH = Path('/path/to/ERA5-Land/raw')

# Folders with the daily gridded SWE maps compared in the analyses
SOURCE_PATHS = {
    # ERA5-Land regridded to 30" files: sd_ERA5-Land_<DD>_<MM>_<YYYY>.tif
    'era5_land': Path('/path/to/era5_land_30arcsec'),
    # HySWE files: SWE_<DD>_<MM>_<YYYY>.tif
    'hybrid': Path('/path/to/hyswe'),
}
