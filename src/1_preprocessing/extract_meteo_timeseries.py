"""Extract the daily CHELSA meteorology at every NH-SWE station (nearest pixel).

Inputs: BASE_PATH/data/STATION_METADATA.csv and CHELSA_PATH/<var>/CHELSA_<var>_DD_MM_YYYY_V.2.1.tif
for pr, tas, rsds, tasmin and tasmax, from config.START_WARMUP to config.END_DATE.

Outputs: BASE_PATH/data/meteo_data/NH_METEO_ID_<id>.csv (one column per variable), and the matched
pixel coordinates (METEO_LAT, METEO_LON) written into STATION_METADATA.csv. In test mode: the test
stations only, 2010-2012, test_ files and no metadata update.
Don't run it together with extract_era5_swe_timeseries.py: both rewrite STATION_METADATA.csv.

Usage: python src/1_preprocessing/extract_meteo_timeseries.py
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union
import rioxarray
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared
import local_paths

# Select the variables for which to extract the time series
variables = ['pr', 'tas', 'rsds', 'tasmin', 'tasmax']

def precompute_indices(
    reference_raster_path: Union[str, Path],
    locations: Dict[int, Tuple[float, float]]
) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, Tuple[float, float]]]:
    """
    Pre-compute exact indices and corresponding lon/lat for each location in xarray.
    
    Args:
        reference_raster_path: Path to reference raster file
        locations: Dictionary mapping station IDs to (longitude, latitude) tuples

    Returns:
        Tuple of two dicts:
            - Dictionary mapping station IDs to (y_index, x_index) tuples
            - Dictionary mapping station IDs to (lon, lat) as found in the raster
    """
    ds = rioxarray.open_rasterio(reference_raster_path)
    location_indices = {}
    location_coords = {}

    for loc_id, (lon, lat) in locations.items():
        # First get the nearest point
        selected = ds.sel(x=lon, y=lat, method='nearest')
        
        # Then get the index positions of those coordinates
        x_idx = ds.indexes['x'].get_loc(selected.x.item())
        y_idx = ds.indexes['y'].get_loc(selected.y.item())
        
        # Get the actual lon/lat from the raster grid
        raster_lon = float(selected.x.item())
        raster_lat = float(selected.y.item())
        
        # Store the indices and raster lon/lat in separate dicts
        location_indices[loc_id] = (x_idx, y_idx)
        location_coords[loc_id] = (raster_lon, raster_lat)
    
    ds.close()
    return location_indices, location_coords

def extract_values(
    raster_paths: pd.Series, 
    location_indices: Dict[int, Tuple[int, int]]
) -> Dict[Union[str, Path], Dict[int, float]]:
    """
    Extract values using xarray with pre-computed indices.
    
    Args:
        raster_paths: List of paths to raster files
        location_indices: Dictionary mapping station IDs to (y_index, x_index) tuples
    
    Returns:
        Nested dictionary mapping {raster_path: {station_id: value}}
    """
    results = {}
    
    # Precompute arrays for vectorized extraction
    station_ids = list(location_indices.keys())
    x_indices = [location_indices[loc_id][0] for loc_id in station_ids]
    y_indices = [location_indices[loc_id][1] for loc_id in station_ids]
    
    # Iterate over each raster file
    pbar = tqdm(list(raster_paths.items()), unit="files", mininterval=1)
    for date, path in pbar:

        # Update the tqdm description for the current file
        pbar.set_description(f"Processing file for date: {date.strftime('%Y-%m-%d')}")
        
        # Extract all values at once using vectorized indexing
        try:
            with rioxarray.open_rasterio(path) as ds:
                values = ds.isel(band=0, x=xr.DataArray(x_indices, dims='z'),
                                 y=xr.DataArray(y_indices, dims='z')).values
        
        # Handle any exceptions that occur during file reading or processing
        except Exception as e:
            print(f"Warning: Skipping file {path} due to error: {e}")
            continue

        # Convert values to float and set fill values to NaN
        values = values.astype(float)
        values[values == ds.attrs['_FillValue']] = np.nan

        # Scale the values according to the raster attributes
        scale_factor = ds.attrs['scale_factor']
        offset = ds.attrs['add_offset']
        values = values * scale_factor + offset

        # Store the extracted values in the results dictionary
        loc_values = dict(zip(station_ids, values))
        results[date] = loc_values

    return results

###############################################################################
# MAIN
###############################################################################

# Read the metadata
metadata = pd.read_csv(local_paths.BASE_PATH / 'data' / 'STATION_METADATA.csv', index_col=0)
print("Metadata loaded successfully.")

# Filter the metadata to only include the subset of stations in test mode
if config.TEST_MODE:
    metadata = metadata[metadata.index.isin(shared.TEST_STATION_IDS)]

# Set the base path to the meteo data directory and get the list of files
file_names = list(local_paths.CHELSA_PATH.glob('**/CHELSA_*.tif'))

# Store the meteo paths in a dataframe and add the variable name
df_meteo_paths = pd.DataFrame({"path": file_names})
df_meteo_paths["variable"] = df_meteo_paths["path"].apply(lambda x: x.parts[-2])

# Check if the variables are present in the dataframe
df_vars = df_meteo_paths.variable.unique()
if not all(var in df_vars for var in variables):
    missing_vars = [var for var in variables if var not in df_vars]
    raise ValueError(f"Variables {missing_vars} are not present in the CHELSA data path: {local_paths.CHELSA_PATH}\nAvailable variables: {df_vars}")
print(f"Processing variables {variables} out of {df_meteo_paths.variable.unique().tolist()}")

# Extract the date from the filename and sort the dataframe
process_date = lambda x: '-'.join(x.parts[-1].split('_')[-4:-1])
df_meteo_paths["date"] = df_meteo_paths["path"].apply(lambda x: process_date(x))
df_meteo_paths["date"] = pd.to_datetime(df_meteo_paths["date"], format='%d-%m-%Y')
df_meteo_paths.sort_values(by="date", inplace=True)

if config.TEST_MODE:
    # Select only from 2010 to 2012
    df_meteo_paths = df_meteo_paths[(df_meteo_paths["date"].dt.year >= 2010) &
                                        (df_meteo_paths["date"].dt.year <= 2012)]
else:
    # Keep only values between the start of the warmup and end date
    df_meteo_paths = df_meteo_paths[(df_meteo_paths["date"] >= config.START_WARMUP) &
                                    (df_meteo_paths["date"] <= config.END_DATE)]

# Select the stations to extract values for
station_coords = {station_id: (row.LON, row.LAT)
                   for station_id, row in metadata.iterrows()}

# Precompute the indices for the first raster
reference_raster_path = df_meteo_paths.path.iloc[0]
station_indices, meteo_coords = precompute_indices(reference_raster_path, station_coords)
print("Pre-computed station location indices for the first CHELSA file.")
print("It is assumed that the same indices are valid for the remaining files.")

# Remove any existing CSV files in the output directory
output_dir = local_paths.BASE_PATH / 'data' / 'meteo_data'
output_dir.mkdir(parents=True, exist_ok=True)

# For each variable, process and write one variable at a time for each station, extracting values on the fly
for variable in variables:
    print(f"Extracting and writing values for variable: {variable}")
    df_variable_paths = df_meteo_paths[df_meteo_paths.variable == variable]
    variable_paths = df_variable_paths.set_index("date").path
    var_values = extract_values(variable_paths, station_indices)

    # Gather the variable values for each station
    for station_id in tqdm(metadata.index, unit="stations", mininterval=1, desc=f"Saving extracted data"):
        var_df = pd.DataFrame.from_dict({date: values[station_id] for date, values in var_values.items()},
                           orient='index', columns=[variable])
        csv_path = output_dir / f'{shared.TEST_PREFIX}NH_METEO_ID_{station_id}.csv'

        # If the CSV already exists, read it and concatenate the new series
        if csv_path.exists():
            df_existing = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
            df_new = df_existing.join(var_df, how='outer', rsuffix=f'_new')
            
            # If the variable already exists, drop the old column
            if variable in df_new.columns and f"{variable}_new" in df_new.columns:
                df_new.drop(columns=[variable], inplace=True)

            # Rename the new column to the variable name
            df_new.rename(columns={f"{variable}_new": variable}, inplace=True)

        # If the CSV does not exist, create a new DataFrame with the series
        else:
            df_new = var_df
        df_new.to_csv(csv_path)
    print(f"Saved extracted values for variable {variable} to {output_dir}")

if not config.TEST_MODE:
    # Update the metadata by adding the meteo coordinates
    for station_id, (lon, lat) in meteo_coords.items():
        metadata.at[station_id, 'METEO_LAT'] = lat
        metadata.at[station_id, 'METEO_LON'] = lon

    # Store the updated metadata with the meteo coordinates
    metadata.to_csv(
        local_paths.BASE_PATH / 'data' / 'STATION_METADATA.csv',
        header=True
    )