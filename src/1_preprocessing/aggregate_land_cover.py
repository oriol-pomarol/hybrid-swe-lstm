"""Aggregate the 30 m GLC_FCS30D land-cover maps to pixel counts per class on the 30" CHELSA grid.

Only needed for the land-cover predictors (Appendix A of the paper).

Inputs: the land-cover tiles in BASE_PATH/data/land_cover_data/decompressed/<folder>/*.tif and one
CHELSA map (CHELSA_PATH/pr/) as the target grid.

Output (BASE_PATH/data/land_cover_data/aggregated/): aggregated_land_cover_<year>.nc for the years in
shared.LC_YEARS, one band per class code in the order of shared.LC_CLASS_GROUPS (overwritten).

Usage: python src/1_preprocessing/aggregate_land_cover.py
"""
import numpy as np
from pathlib import Path
import rioxarray
import xarray as xr
import netCDF4 as nc
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import shared
import local_paths

def initialize_lowres_datasets(output_dir: Path, years:list) -> tuple:
    """Initialize a low-resolution land cover dataset per year with zero values."""

    # Load a meteo dataset as a base for the aggregated land cover dataset
    meteo_path = local_paths.CHELSA_PATH / 'pr' / 'CHELSA_pr_11_07_2020_V.2.1.tif'
    with rioxarray.open_rasterio(meteo_path, cache=False) as ds_meteo:

        # Take only the northern hemisphere part of the dataset
        ds_meteo = ds_meteo.sel(y=ds_meteo.y >= 0)

        # Store the dimensions of the meteo dataset
        lowres_resolution = ds_meteo.rio.resolution()

        # Create a low-resolution land cover dataset with meteo dataset dimensions
        ds_zeroes = xr.Dataset(
            data_vars={
                'lc_freq': (
                    ['y', 'x', 'lc_class'], 
                    np.zeros((ds_meteo.y.size, ds_meteo.x.size, len(shared.LC_CLASS_GROUPS)), dtype=np.int16)
                )
            },
            coords={
                'y': ds_meteo.y,
                'x': ds_meteo.x,
            }
        )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to file the 0-filled aggregated land cover dataset for all years
    for year in years:
        output_file = output_dir / f'aggregated_land_cover_{year}.nc'
        if output_file.exists():
            output_file.unlink(missing_ok=True)
            print(f"Deleted existing file: {output_file}")
        ds_zeroes.to_netcdf(output_file, mode='w')
    print(f"Created zero-filled files in {output_dir}")

    return lowres_resolution


def add_time_dimension(ds: xr.Dataset) -> xr.Dataset:
    """Add a time dimension to the dataset based on the filename."""
    # Set the years depending on the filename containing '5years' or 'Annual'
    if '5years' in ds.encoding['source']:
        years = [1985, 1990, 1995]
    elif 'Annual' in ds.encoding['source']:
        years = [y for y in range(2000, 2023)]
    else:
        raise ValueError("Filename does not contain '5years' or 'Annual'")
    
    # Assign the years as coordinates to the 'band' dimension
    ds = ds.assign_coords(time=("band", years))
    ds = ds.swap_dims({"band": "time"})
    ds = ds.drop_vars('band')
    ds.time.attrs.update({
        'standard_name': 'time',
        'long_name': 'Time',
        'units': 'year'
    })
    return ds

def create_coord_mapping_cache(nc_lowres, ds_highres, lowres_resolution):
    """
    Pre-compute and cache coordinate mappings.
    """
    # Calculate resolution
    res_x, res_y = lowres_resolution

    # Pre-compute edges
    lowres_x_edges = np.concatenate([nc_lowres.variables['x'][:] - res_x/2, 
                                   [nc_lowres.variables['x'][-1] + res_x/2]])
    lowres_y_edges = np.concatenate([nc_lowres.variables['y'][:] - res_y/2, 
                                   [nc_lowres.variables['y'][-1] + res_y/2]])

    # Compute mapping indices
    highres_to_lowres_x = np.digitize(ds_highres.x.values, lowres_x_edges) - 1
    highres_to_lowres_y = np.digitize(ds_highres.y.values, lowres_y_edges) - 1
    
    # Clip indices to valid range
    highres_to_lowres_x = np.clip(highres_to_lowres_x, 0, nc_lowres.dimensions['x'].size - 1)
    highres_to_lowres_y = np.clip(highres_to_lowres_y, 0, nc_lowres.dimensions['y'].size - 1)

    return {
        'highres_to_lowres_x': highres_to_lowres_x,
        'highres_to_lowres_y': highres_to_lowres_y,
    }

def create_class_lookup_table() -> np.ndarray:
    """Map land cover class values to band indices; every other value goes to the last band (code 255)."""
    class_indices = np.array(list(shared.LC_CLASS_GROUPS)[:-1])

    # Create lookup table (large enough to handle value 250)
    lookup = np.full(256, len(class_indices), dtype=np.int64)
    lookup[class_indices] = np.arange(len(class_indices))
    return lookup

def process_1_year_of_data(year_highres_values, nc_lowres_lc, coord_cache, class_lookup):

    # Extract the dimensions of the high-resolution data
    n_y, n_x = year_highres_values.shape

    # Check if there are any valid points
    if year_highres_values.size == 0:
        print(f"Warning: No valid points found for this file/year. Skipping processing.")
        return

    # Correct the order of indices in the values, flatten them and map classes
    year_highres_values = year_highres_values.flatten()
    year_highres_values = class_lookup[year_highres_values.astype(np.int64)]

    # Calculate the low-res ones using the pre-computed cache
    highres_y_indices, highres_x_indices = np.indices((n_y, n_x))
    highres_x_indices = highres_x_indices.flatten()
    highres_y_indices = highres_y_indices.flatten()
    lowres_x_indices = coord_cache['highres_to_lowres_x'][highres_x_indices]
    lowres_y_indices = coord_cache['highres_to_lowres_y'][highres_y_indices]

    # Find the range of x and y indices
    min_x_idx, max_x_idx = lowres_x_indices.min(), lowres_x_indices.max()
    min_y_idx, max_y_idx = lowres_y_indices.min(), lowres_y_indices.max()

    # Filter valid data
    subset_x_idxs = lowres_x_indices - min_x_idx
    subset_y_idxs = lowres_y_indices - min_y_idx

    # Calculate the number of bins in the subset
    n_x_bins = max_x_idx - min_x_idx + 1
    n_y_bins = max_y_idx - min_y_idx + 1
    n_classes = nc_lowres_lc.dimensions['lc_class'].size

    # Create 3D flat indices for the subset
    flat_indices = (subset_x_idxs * (n_y_bins * n_classes) + 
                    subset_y_idxs * n_classes + 
                    year_highres_values)
    
    # Use bincount to count occurrences
    class_counts = np.bincount(flat_indices, minlength=n_x_bins * n_y_bins * n_classes)

    # Reshape to 3D and convert to int16
    class_counts = class_counts.reshape(n_x_bins, n_y_bins, n_classes).astype(np.int16)

    # Add the results to the low-res dataset
    nc_lowres_lc['lc_freq'][min_y_idx:max_y_idx+1, min_x_idx:max_x_idx+1, :] += np.transpose(class_counts, (1, 0, 2))
    print("Updated region: "
            f"x[{nc_lowres_lc.variables['x'][min_x_idx]:.2f}:"
            f"{nc_lowres_lc.variables['x'][max_x_idx]:.2f}],"
            f"y[{nc_lowres_lc.variables['y'][min_y_idx]:.2f}:"
            f"{nc_lowres_lc.variables['y'][max_y_idx]:.2f}],"
            f" shape: {class_counts.shape}")
    return

###############################################################################
# MAIN SCRIPT
###############################################################################

# Gather the high-resolution land cover datasets file paths
lc_path = local_paths.BASE_PATH/ 'data' / 'land_cover_data' / 'decompressed'
dirs = sorted([d for d in lc_path.iterdir() if d.is_dir()])
files = [f for d in dirs for f in sorted(d.glob('*.tif'))]

# Initialize the low-resolution land cover datasets
output_dir = local_paths.BASE_PATH/ 'data' / 'land_cover_data' / 'aggregated'
years = list(shared.LC_YEARS)
lowres_resolution = initialize_lowres_datasets(output_dir, years)

# Create a lookup table for land cover classes
class_lookup = create_class_lookup_table()

for file_path in tqdm(files, desc="Processing land cover files"):

    # Load the high-resolution land cover dataset
    ds_highres_lc = rioxarray.open_rasterio(file_path, cache=False)
    ds_highres_lc = add_time_dimension(ds_highres_lc)

    for i, year in enumerate(ds_highres_lc.time.values):

        if year not in years:
            print(f"Skipping year {year} as it is not in the target years list.")
            continue

        # Load the low-resolution land cover dataset for this year
        print(f"Processing year {year} for file {file_path.name}")
        output_file = output_dir / f'aggregated_land_cover_{year}.nc'

        with nc.Dataset(output_file, 'r+') as nc_lowres_lc:

            # Create coordinate mapping cache for this lowres dataset
            if i == 0:
                coord_cache = create_coord_mapping_cache(
                    nc_lowres_lc,ds_highres_lc, lowres_resolution
                    )

            # Process 1 year of data
            process_1_year_of_data(
                ds_highres_lc.sel(time=year).values, nc_lowres_lc, coord_cache, class_lookup
            )

    # Close the loaded highres dataset
    ds_highres_lc.close()