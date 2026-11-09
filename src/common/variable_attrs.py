from dataclasses import dataclass, field
from typing import List
import config
import local_paths
import rioxarray

@dataclass
class Variable:
    """A class to hold information about a single variable."""
    short_name: str
    long_name: str
    units: str
    dynamic: bool
    offset: float = 0.0
    scale_factor: float = 1.0
    predictor: bool = False
    train_norm: bool = True
    log_scaling: bool = False

class VariableCollection:
    """A container to hold and manage a collection of Variable objects."""

    def __init__(self):
        self._variables = {}

    def add_variable(self, var: Variable):
        """Adds a new variable to the collection."""
        self._variables[var.short_name] = var

    def get(self, short_name: str) -> Variable:
        """Retrieves a single variable by its short name (fast)."""
        return self._variables.get(short_name)

    def find(self, **kwargs) -> List[Variable]:
        """
        Generic search method to find all variables matching attribute criteria.
        Example: find(units='K', predictor=True)
        """
        results = []
        for var in self._variables.values():
            # Check if the variable matches all specified criteria
            if all(getattr(var, key) == value for key, value in kwargs.items()):
                results.append(var)
        return results

    # def get_predictors(self) -> List[Variable]:
    #     """Convenience method to retrieve all predictor variables."""
    #     # This is just a specialized use of our generic find() method
    #     return self.find(predictor=True)
        
    def __iter__(self):
        """Allows looping directly over the collection, e.g., `for var in collection:`"""
        return iter(self._variables.values())

VARIABLE_ATTRS = VariableCollection()

# Add SWE to the collection
VARIABLE_ATTRS.add_variable(Variable(
    short_name='swe',
    long_name='Snow Water Equivalent',
    units='mm',
    dynamic=True,
    predictor=True,
    train_norm=True,
    log_scaling=False
))

# Add ERA5-Land SWE to the collection
VARIABLE_ATTRS.add_variable(Variable(
    short_name='era5_swe',
    long_name='ERA5-Land Snow Water Equivalent',
    units='mm',
    dynamic=True,
    predictor=True,
    train_norm=True,
    log_scaling=False
))

# Add the CHELSA variables to the collection
for var_name in ['pr', 'tas', 'rsds', 'tasmin', 'tasmax']:

    # Open the first available file for the variable to retrieve its attributes
    path = local_paths.CHELSA_PATH / var_name
    file = next(path.glob(f'CHELSA_{var_name}*.tif'))
    ds = rioxarray.open_rasterio(file, engine='rasterio')

    # Store the attributes to a Variable class
    variable = Variable(
        short_name=var_name,
        long_name=ds.attrs.get('variable_long_name', var_name),
        units=ds.attrs.get('variable_unit', ''),
        offset=ds.attrs.get('add_offset', 0.0),
        scale_factor=ds.attrs.get('scale_factor', 1.0),
        dynamic=True,
        predictor=True if var_name in config.DYN_VARS else False,
        train_norm=True,
        log_scaling=True if var_name=='pr' else False
    )
    VARIABLE_ATTRS.add_variable(variable)

# Add the topographic variables to the collection from the GMTED-based layers
topo_path = local_paths.BASE_PATH / 'data' / 'topo_data' / 'Amatulli_2016'
for path in topo_path.glob('*_GMTED*.tif'):

    # Open the file and retrieve its variable, aggregation method and description
    ds = rioxarray.open_rasterio(path, engine='rasterio')
    var = path.stem.split('_')[0]
    agg = path.stem.split('_')[1][-2:]
    short_name = var + '_' + agg
    file_desc = ds.attrs.get('TIFFTAG_DOCUMENTNAME', short_name)

    # Define the units
    if var == 'slope':
        units = 'degrees'
    elif var in ['eastness', 'northness']:
        units = '-'
    else:
        units = 'm'

    # Create the Variable object
    variable = Variable(
        short_name=short_name,
        long_name=file_desc.replace('30 arc-seconds ', ''),
        units=units,
        offset=ds.attrs.get('add_offset', 0.0),
        scale_factor=ds.attrs.get('scale_factor', 1.0),
        dynamic=False,
        predictor=True if short_name in config.STATIC_VARS else False,
        train_norm=True,
        log_scaling=False if var in ['eastness', 'northness', 'tpi'] else True
    )
    VARIABLE_ATTRS.add_variable(variable)

# deciduous,evergreen,grassland,no_data,shrubland,urban,water,wetland
# Add the land cover variable to the collection
for var_name in ['bare_lc', 'cropland_lc', 'deciduous_lc', 'evergreen_lc',
                 'grassland_lc', 'shrubland_lc', 'urban_lc', 'water_lc', 'wetland_lc']:
    VARIABLE_ATTRS.add_variable(Variable(
        short_name=var_name,
        long_name=f'{var_name.split("_")[0].capitalize()} Land Cover Fraction',
        units='-',
        dynamic=False,
        predictor=True if var_name in config.STATIC_VARS else False,
        train_norm=True,
        log_scaling=False
    ))