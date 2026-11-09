import numpy as np
from pandas import Timestamp

# GENERAL
# If True, run the whole workflow as a quick test on few stations
TEST_MODE: bool = False

# TARGET AND PREDICTORS
PREDICT_dSWE: bool = False  # If True, ML predicts dSWE and adds it to prev_swe
HYBRID_MODE: bool = True  # If True, include ERA5 SWE as a predictor
DYN_VARS: list = [
    # Target variable
    'swe',
    # Meteorological predictors
    'pr', 'tas', 'rsds', #'tasmin', 'tasmax',
]
STATIC_VARS: list = [
    # Topographic predictors
    'elevation_mn', 'slope_mn', 'eastness_mn', 'northness_mn', 'tpi_mn', #'elevation_sd',
    # # Land cover predictors
    # 'bare_lc', 'cropland_lc', 'deciduous_lc', 'evergreen_lc', 'grassland_lc', 'shrubland_lc', 'urban_lc', 'water_lc', 'wetland_lc',
]

# DATES
START_WARMUP: Timestamp = Timestamp('1979-10-01')
START_DATE: Timestamp = Timestamp('1980-10-01')
END_DATE: Timestamp = Timestamp('2020-09-30')

# STRATIFICATION BINS
ELEVATION_BINS: list = [-np.inf, 500, 1500, np.inf]
SNOW_CLASSES: list = ['Prairie', 'Montane Forest', 'Ephemeral', 'Maritime', 'Tundra', 'Boreal Forest']

# ANALYSIS
MOUNTAIN_NAMES: tuple = ("Sierra Nevada", "Coast Mountains", "Canadian Rockies", "European Alps",
             "Pyrenees", "Southern Scandes", "Caucasus Mountains", "Pamir Mountains", "Eastern Sayan", "Himalaya")
