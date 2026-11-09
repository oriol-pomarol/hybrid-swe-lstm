"""Extract the daily ERA5-Land SWE at every NH-SWE station (nearest land pixel with data), in mm.

Inputs: BASE_PATH/data/STATION_METADATA.csv and ERA5LAND_PATH/sd/sd_ERA5-Land_<year>0101-<year>1231.nc
for the years from config.START_WARMUP to config.END_DATE.

Outputs: BASE_PATH/data/swe_data/ERA5_LAND/NH_ERA5_SWE_ID_<id>.csv (column 'SWE (mm)'), and the matched
pixel coordinates (ERA5_SWE_LAT, ERA5_SWE_LON) written into STATION_METADATA.csv. In test mode: the
test stations only, 2010-2012, test_ files and no metadata update.
Don't run it together with extract_meteo_timeseries.py: both rewrite STATION_METADATA.csv.

Usage: python src/1_preprocessing/extract_era5_swe_timeseries.py
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared
import local_paths

def find_nearest_valid_point(
    ds: xr.DataArray,
    lon: float,
    lat: float,
    mask: xr.DataArray = None,
    idx_radius: int = 3
) -> xr.DataArray:
    """
    Find the nearest valid point in the xarray dataset to the given lon/lat.
    
    Args:
        ds: xarray DataArray with 'sd' variable
        lon: Longitude of the target location
        lat: Latitude of the target location
        idx_radius: Search radius in index units
    Returns:
        xarray DataArray of the nearest valid point
    """

    def is_valid(point: xr.DataArray) -> bool:
        if mask is not None:
            lon_idx = ds.indexes['longitude'].get_loc(point.longitude.item())
            lat_idx = ds.indexes['latitude'].get_loc(point.latitude.item())
            mask_value = mask.isel(longitude=lon_idx, latitude=lat_idx).item()
            return mask_value == 1
        else:
            return not point.isnull().all()

    # Select the nearest point
    nearest = ds.sel(longitude=lon, latitude=lat, method='nearest')
    if is_valid(nearest):
        return nearest
    else:
        print(f"Nearest point at ({nearest.longitude.item()}, {nearest.latitude.item()}) is NaN. Searching surrounding points.")
    
    # If the nearest point is NaN, retrieve the indexes of the surrounding points
    lon_idx = ds.indexes['longitude'].get_loc(nearest.longitude.item())
    lat_idx = ds.indexes['latitude'].get_loc(nearest.latitude.item())
    lon_start = max(lon_idx - idx_radius, 0)
    lon_end = min(lon_idx + idx_radius + 1, len(ds.indexes['longitude']))
    lat_start = max(lat_idx - idx_radius, 0)
    lat_end = min(lat_idx + idx_radius + 1, len(ds.indexes['latitude']))

    # Order the surrounding points by distance to the target lon/lat
    surrounding_points = []
    for lon_idx in range(lon_start, lon_end):
        for lat_idx in range(lat_start, lat_end):
            point_lon = ds.indexes['longitude'][lon_idx]
            point_lat = ds.indexes['latitude'][lat_idx]
            distance = np.sqrt((point_lon - lon)**2 + (point_lat - lat)**2)
            surrounding_points.append((distance, lon_idx, lat_idx))
        
    # Check the surrounding points in order of distance
    surrounding_points.sort(key=lambda x: x[0])
    for _, lon_idx, lat_idx in surrounding_points:
        candidate = ds.isel(longitude=lon_idx, latitude=lat_idx)
        if is_valid(candidate):
            print(f"Found valid point at ({candidate.longitude.item()}, {candidate.latitude.item()}).")
            return candidate
    
    print(f"Warning: No valid points found within radius {idx_radius}. Returning NaN.")
    return nearest

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
    ds = xr.open_dataset(reference_raster_path, engine='netcdf4')
    ds = ds.sel(latitude=slice(90, 0))
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')
    location_indices = {}
    location_coords = {}

    # Land-sea mask: pixels with data on the first day of the reference file
    time_dim = next(dim for dim in ds['sd'].dims if dim not in ('latitude', 'longitude'))
    land_mask = ds['sd'].isel({time_dim: 0}).notnull().load()

    for loc_id, (lon, lat) in locations.items():

        # Select the nearest point in the raster to the given lon/lat
        nearest_valid = find_nearest_valid_point(ds['sd'], lon, lat, land_mask)

        # Then get the index positions of those coordinates
        x_idx = ds.indexes['longitude'].get_loc(nearest_valid.longitude.item())
        y_idx = ds.indexes['latitude'].get_loc(nearest_valid.latitude.item())
        
        # Get the actual lon/lat from the raster grid
        raster_lon = float(nearest_valid.longitude.item())
        raster_lat = float(nearest_valid.latitude.item())
        
        # Store the indices and raster lon/lat in separate dicts
        location_indices[loc_id] = (x_idx, y_idx)
        location_coords[loc_id] = (raster_lon, raster_lat)
    
    ds.close()
    return location_indices, location_coords

def extract_values(
    raster_paths: Dict[int, Path],
    location_indices: Dict[int, Tuple[int, int]]
) -> Dict[Union[str, Path], Dict[int, float]]:
    """
    Extract values using xarray with pre-computed indices.
    
    Args:
        raster_paths: Dictionary mapping year to path of raster files
        location_indices: Dictionary mapping station IDs to (y_index, x_index) tuples
    
    Returns:
        Nested dictionary mapping {raster_path: {station_id: value}}
    """    
    # Precompute arrays for vectorized extraction
    station_ids = list(location_indices.keys())
    x_indices = [location_indices[loc_id][0] for loc_id in station_ids]
    y_indices = [location_indices[loc_id][1] for loc_id in station_ids]

    # Initialize results dictionary
    results = {station_id: [] for station_id in station_ids}
    dates = []
    
    # Iterate over each raster file
    for year, path in tqdm(raster_paths.items(), unit="files", mininterval=1):
        
        # Add the processed year to the tqdm description
        tqdm.write(f"Processing file for year: {year}")
        
        # Extract all values at once using vectorized indexing
        try:
            with xr.open_dataset(path, engine='netcdf4') as ds:
                if ds.longitude.max() > 180:
                    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')
                ds = ds.sel(latitude=slice(90, 0))
                values = ds['sd'].isel(longitude=xr.DataArray(x_indices, dims='z'),
                                         latitude=xr.DataArray(y_indices, dims='z')).values
        
        # Handle any exceptions that occur during file reading or processing
        except Exception as e:
            print(f"Warning: Skipping file {path} due to error: {e}")
            continue

        # Convert values to float
        values = values.astype(float)

        # Set to NaN any values with GRIB_missingValue
        missing_value = ds['sd'].attrs.get('GRIB_missingValue', None)
        if missing_value is not None:
            values[values == missing_value] = np.nan
        else:
            print(f"Warning: 'GRIB_missingValue' attribute not found in {path}. No missing values set.")

        # Convert from meters to millimeters
        values = values * 1000

        # Store the extracted values in the results dictionary
        dates.append(ds['sd'].valid_time.values)
        for i in range(len(station_ids)):
            results[station_ids[i]].append(values[:, i])

    return dates, results

###############################################################################
# MAIN
###############################################################################

# Read the metadata
metadata = pd.read_csv(local_paths.BASE_PATH / 'data' / 'STATION_METADATA.csv', index_col=0)

print("Metadata loaded successfully.")

# Set the data directory 
swe_path = local_paths.ERA5LAND_PATH / 'sd'
file_names = {}
for year in range(config.START_WARMUP.year, config.END_DATE.year + 1):
    # In test mode, only process a subset of years
    if config.TEST_MODE:
        if year not in [2010, 2011, 2012]:
            continue
    year_path = swe_path / f'sd_ERA5-Land_{year}0101-{year}1231.nc'
    if not year_path.exists():
        raise FileNotFoundError(f"Directory not found: {year_path}")
    else:
        file_names[year] = year_path

# Filter the metadata to only include the subset of stations in test mode
if config.TEST_MODE:
    metadata = metadata[metadata.index.isin(shared.TEST_STATION_IDS)]

# Select the stations to extract values for
station_coords = {station_id: (row.LON, row.LAT)
                   for station_id, row in metadata.iterrows()}

# Precompute the indices for one raster
reference_raster_path = file_names[next(iter(file_names))]
station_indices, era5_coords = precompute_indices(reference_raster_path, station_coords)

if not config.TEST_MODE:
    # Update the metadata by adding the era5 coordinates
    for station_id, (lon, lat) in era5_coords.items():
        metadata.at[station_id, 'ERA5_SWE_LAT'] = lat
        metadata.at[station_id, 'ERA5_SWE_LON'] = lon

    # Store the updated metadata with the era5 coordinates
    metadata.to_csv(
        local_paths.BASE_PATH / 'data' / 'STATION_METADATA.csv',
        header=True
    )

# Check that the output directory exists
output_dir = local_paths.BASE_PATH / 'data' / 'swe_data' / 'ERA5_LAND'
output_dir.mkdir(parents=True, exist_ok=True)

# Process and write one variable at a time for each station, extracting values on the fly

print(f"Extracting and writing time series for ERA5-Land SWE data")
dates, values = extract_values(file_names, station_indices)

# Concatenate the dates
dates = np.concatenate(dates)

# Gather the variable values for each station
for station_id in metadata.index:
    # Concatenate the values
    conc_values = np.concatenate(values[station_id])
    df_values = pd.DataFrame(conc_values, index=dates, columns=['SWE (mm)'])
    csv_path = output_dir / f'{shared.TEST_PREFIX}NH_ERA5_SWE_ID_{station_id}.csv'
    df_values.to_csv(csv_path)