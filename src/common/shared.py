"""Values shared by the scripts: derived from config.py or fixed by the code; not meant to be edited.

Experiment settings go in config.py, machine-specific paths in local_paths.py.
"""
import numpy as np
import config

# Number of warmup days before the first simulation day (spin-up of the hidden state and SWE)
WARMUP_DAYS: int = (config.START_DATE - config.START_WARMUP).days

# Stations of the quick test (config.TEST_MODE), default is Pyrenees stations
TEST_STATION_IDS: list = [11427, 11425, 11431, 11430, 11423, 11432, 11428, 11429, 11424, 11426, 11433]

# Prefix of every file written in test mode, so test outputs never overwrite real files
TEST_PREFIX: str = 'test_' if config.TEST_MODE else ''

# Labels of the right-closed elevation bands of config.ELEVATION_BINS,
# e.g., '500-', '500-1500', '1500+'
ELEVATION_LABELS: list = [f"{hi:g}-" if lo == -np.inf else f"{lo:g}+" if hi == np.inf else f"{lo:g}-{hi:g}"
                          for lo, hi in zip(config.ELEVATION_BINS[:-1], config.ELEVATION_BINS[1:])]

# Factor that converts each gridded source's SWE to mm (ERA5-Land stores metres of water equivalent)
SOURCE_TO_MM: dict = {
    "era5_land": 1000.0,
    "hybrid": 1.0,
}

# Years of the aggregated land-cover maps
LC_YEARS: tuple = tuple(range(1985, 2021, 5))

# Land-cover group of each class code, in the band order of the aggregated land-cover maps
LC_CLASS_GROUPS: dict = {
    10: 'cropland', 11: 'cropland', 12: 'cropland', 20: 'cropland',
    51: 'evergreen', 52: 'evergreen', 61: 'deciduous', 62: 'deciduous',
    71: 'evergreen', 72: 'evergreen', 81: 'deciduous', 82: 'deciduous',
    91: 'evergreen', 92: 'evergreen',
    120: 'shrubland', 121: 'shrubland', 122: 'shrubland',
    130: 'grassland', 140: 'grassland',
    181: 'wetland', 182: 'wetland', 183: 'wetland', 184: 'wetland', 185: 'wetland', 186: 'wetland', 187: 'wetland',
    190: 'urban',
    150: 'bare', 152: 'bare', 153: 'bare', 200: 'bare', 201: 'bare', 202: 'bare',
    210: 'water', 220: 'bare',
    255: 'no_data',
}

# Station sets a model run can train and evaluate on (main.py --station-set)
STATION_SETS: tuple = ('full', 'tune', 'subset')

# Default hyperparameters of a model run; patience defaults to max(1, int(0.1 * n_epochs))
HPARAM_DEFAULTS: dict = {
    'batch_size': 128,
    'hidden_size': 128,
    'learning_rate': 0.001,
    'optimizer_name': 'Adam',
    'n_epochs': 100,
    'chunk_size': 999999,
    'min_delta': 0.0,
    'dropout': 0.0,
    'max_training_hours': 199.0,
}

# Smaller defaults for test-mode runs and the 'subset' station set
SMALL_RUN_HPARAM_DEFAULTS: dict = {
    'batch_size': 32,
    'hidden_size': 64,
    'n_epochs': 30,
}

# Optimizers a model run can use
OPTIMIZERS: tuple = ('Adam', 'RMSprop', 'SGD')
