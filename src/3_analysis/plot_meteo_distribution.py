"""Histograms of the meteorological predictors at the stations vs. over the HySWE domain.

Inputs: the station sequences in BASE_PATH/data/temp, the CHELSA maps and BASE_PATH/data/valid_swe_regression_flags.nc.
Outputs (results/predictor_distribution/): meteo_distribution.json (counts; reused if present) and meteo_distribution.png

Usage: python src/3_analysis/plot_meteo_distribution.py
"""
import rioxarray
import json
import xarray as xr
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared
import local_paths
from variable_attrs import VARIABLE_ATTRS

# Define the paths to the meteorological data
meteo_path = local_paths.BASE_PATH / 'data/meteo_data'
temp_path = local_paths.BASE_PATH / 'data/temp'
results_path = local_paths.BASE_PATH / 'results' / 'predictor_distribution'
results_path.mkdir(parents=True, exist_ok=True)

def initialize_bin_edges():
    """Initialize the bin edges for each variable based on predefined ranges."""
    
    # Define the number of bins and minimum and maximum values for each variable
    n_bins = 1024
    var_ranges = {
        'pr': (0, 900),
        'tas': (170, 360),
        'rsds': (0, 120)
    }

    # Create bin edges for each variable
    bin_edges = {}
    for var, (min_val, max_val) in var_ranges.items():
        edges = np.linspace(min_val, max_val, n_bins + 1)
        bin_edges[var] = edges.tolist()

    return bin_edges

def load_inference_mask():
    """Load the valid HySWE domain (is_valid) as a boolean DataArray."""
    mask_path = local_paths.BASE_PATH / 'data' / 'valid_swe_regression_flags.nc'
    with xr.open_dataset(mask_path, engine='netcdf4') as ds:

        # Ensure the dimensions are correctly named
        if set(ds.dims) == {'lat', 'lon'}:
            ds = ds.rename({'lat': 'y', 'lon': 'x'})

        # Ensure the y dimension is in descending order
        if ds['y'].values[0] < ds['y'].values[-1]:
            ds = ds.isel(y=slice(None, None, -1))

        return ds['is_valid'] == 1

def load_station_data():
    """Load all station CSVs into a dictionary."""

    # Find all csv files in the data path and extract the station IDs
    station_paths = temp_path.glob(f'{shared.TEST_PREFIX}dynamic_predictors_ID_*.csv')
    station_dict = {path.stem.split('_')[-1]: path for path in station_paths}

    # Function to load a single CSV file
    def load_csv(station_id, path):
        meteo_data = pd.read_csv(path, index_col=0)
        meteo_data.index = pd.to_datetime(meteo_data.index)
        meteo_data.index.name = 'Date'
        if 'SWE (mm)' in meteo_data.columns:
            meteo_data.drop(columns=['SWE (mm)'], inplace=True)
        return station_id, meteo_data

    # Create a dictionary to hold the DataFrames for each station
    meteo_dfs = {}

    # Use ThreadPoolExecutor for threading
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_csv, station_id, path): station_id for station_id, path in station_dict.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading meteorological data"):
            station_id, meteo_data = future.result()
            meteo_dfs[station_id] = meteo_data

    return meteo_dfs

def process_station_data(meteo_dfs, bin_edges):
    """Process and aggregate all station data."""

    # Merge all DataFrames and add NHSWEID as a column
    merged_meteo_df = pd.concat(
        [df.assign(NHSWEID=station_id) for station_id, df in meteo_dfs.items()]
    )

    # Set the index to both the date and station ID
    merged_meteo_df.set_index(['NHSWEID', merged_meteo_df.index], inplace=True)

    # Retrieve the histograms for each variable given the edge bins from the stats
    station_counts = {}
    for variable, edges in bin_edges.items():

        # Compute the histogram counts for the station data
        counts, _ = np.histogram(merged_meteo_df[variable].values, bins=edges)

        # Store the histogram counts in the dictionary
        station_counts[variable] = counts.tolist()

    return station_counts

def process_inference_data(mask_da, bin_edges):
    """Process inference data and compute histograms."""

    inference_counts = {}
    for variable, edges in bin_edges.items():

        print(f"Processing inference data for variable: {variable}")

        # Find the file paths for the inference data
        file_paths = list(local_paths.CHELSA_PATH.glob(f'{variable}/*.tif'))

        # Select only files between 1980 and 2020
        file_paths = [fp for fp in file_paths if 1980 <= int(fp.stem.split('_')[-2]) <= 2020]
        if config.TEST_MODE:
            file_paths = np.random.choice(file_paths, size=2, replace=False)

        global_hist = np.zeros(len(edges) - 1, dtype=np.int64)
        for fp in tqdm(file_paths, desc="Processing files", unit="file"):

            # Open the raster file and constrain it to the NH
            da = rioxarray.open_rasterio(fp)
            da = da.where(da.y >= 0, drop=True)

            # Select the data
            data = da.data.squeeze()[mask_da].ravel()
            if data.size == 0:
                print(f"Warning: No data found in file {fp}. Skipping.")
                continue

            # Set the fill value to NaN
            data = data.astype(float)
            data[data == da.attrs.get('_FillValue', np.nan)] = np.nan

            # Find the scale factor and adjust it for the rsds data
            scale_factor = da.attrs['scale_factor']
            if variable == 'rsds' and scale_factor == 1.0:
                scale_factor = 0.001

            # Find the offset and apply the transformation
            offset = da.attrs['add_offset']
            data = data * scale_factor + offset

            # Check for out-of-range values
            if data.max() > edges[-1]:
                print(f"Warning: Maximum value {data.max()} exceeds the last bin edge {edges[-1]} for file {fp}.")
            if data.min() < edges[0]:
                print(f"Warning: Minimum value {data.min()} is below the first bin edge {edges[0]} for file {fp}.")

            # Calculate the file histogram and update the global histogram
            file_hist, _ = np.histogram(data, bins=edges)
            global_hist += file_hist

        # Store the histogram counts for the variable
        inference_counts[variable] = global_hist.tolist()

    return inference_counts

def save_counts(station_counts, inference_counts, bin_edges):
    """Save the counts to a JSON file."""

    # Store the results in a dictionary
    meteo_distribution = {}
    for variable in bin_edges.keys():
        meteo_distribution[variable] = {
            'bin_edges': bin_edges[variable],
            'station_counts': station_counts[variable],
            'inference_counts': inference_counts[variable]
        }

    # Save the results to a JSON file
    with open(results_path / f'{shared.TEST_PREFIX}meteo_distribution.json', 'w') as f:
        json.dump(meteo_distribution, f)

    return meteo_distribution

def plot_meteo_distribution(meteo_distribution, bin_join_factor=20):
    """Plot the histograms of the meteorological data."""

    # Get all variables from the histogram data
    variables = list(meteo_distribution.keys())

    # Calculate grid dimensions for subplots
    n_vars = len(variables)
    n_cols = min(3, n_vars)  # Max 3 columns
    n_rows = (n_vars + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), sharey=True)
    axes = np.array(axes).reshape(-1)  # Flatten axes array for easy indexing

    # Loop through all variables
    for i, (var, var_distribution) in enumerate(meteo_distribution.items()):
        ax = axes[i]
        
        # Get data for the current variable
        station_counts = np.array(var_distribution['station_counts'])
        edges = np.array(var_distribution['bin_edges'])
        inference_counts = np.array(var_distribution['inference_counts'])
        
        # Join bins if bin_join_factor > 1
        if bin_join_factor > 1:
            # Calculate how many groups we'll have after joining
            n_bins = len(station_counts)
            n_groups = n_bins // bin_join_factor

            # If not evenly divisible, adjust to use complete groups only
            if n_bins % bin_join_factor != 0:
                usable_bins = n_groups * bin_join_factor
                station_counts = station_counts[:usable_bins]
                inference_counts = inference_counts[:usable_bins]
                edges = edges[:usable_bins+1]
            
            # Group and sum bins
            station_counts = np.sum(station_counts.reshape(n_groups, bin_join_factor), axis=1)
            inference_counts = np.sum(inference_counts.reshape(n_groups, bin_join_factor), axis=1)

            # Keep only the edges at group boundaries
            edges = np.array([edges[i*bin_join_factor] for i in range(n_groups+1)])

        # Calculate bar positions and widths
        bar_width = np.diff(edges)
        bar_centers = edges[:-1] + bar_width/2
        
        # Plot the data as a bar plot
        ax.bar(bar_centers, inference_counts, width=bar_width, 
                alpha=0.4, color='coral', edgecolor='orangered',
                zorder=1)
        ax.bar(bar_centers, station_counts, width=bar_width, 
                      alpha=0.6, color='skyblue', edgecolor='steelblue',
                      zorder=3)

        # Set titles and labels
        ax.set_title(VARIABLE_ATTRS.get(var).long_name)
        ax.set_xlabel(VARIABLE_ATTRS.get(var).units)
        ax.set_ylabel('Frequency')

        # Apply log scale
        ax.set_yscale('log')

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Add a common legend for the figure
    handles = [
        plt.Rectangle((0,0),1,1, facecolor='skyblue', alpha=0.6, edgecolor='steelblue'),
        plt.Rectangle((0,0),1,1, facecolor='coral', alpha=0.4, edgecolor='orangered')
    ]
    labels = ['Station Data', 'Inference Data']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.09), ncol=2)

    plt.savefig(results_path / f'{shared.TEST_PREFIX}meteo_distribution.png', dpi=300)

    return

###############################################################################
# MAIN SCRIPT
###############################################################################

# Load precomputed results if available
meteo_dist_file = results_path / 'meteo_distribution.json'
if meteo_dist_file.exists() and not config.TEST_MODE:
    print(f"Loading precomputed results from {meteo_dist_file}")
    with open(meteo_dist_file, 'r') as f:
        meteo_distribution = json.load(f)
else:
    if config.TEST_MODE:
        print("Calculating results for test mode...")
    else:
        print("Precomputed results not found, calculating from scratch...")

    # Initialize bin edges for each variable
    bin_edges = initialize_bin_edges()

    # Load and process the station data
    station_dfs = load_station_data()
    station_counts = process_station_data(station_dfs, bin_edges=bin_edges)

    # Load and process the inference data
    mask_da = load_inference_mask()
    inference_counts = process_inference_data(mask_da, bin_edges=bin_edges)

    # Save results
    meteo_distribution = save_counts(station_counts, inference_counts, bin_edges)

# Plot the histograms of the meteorological data
plot_meteo_distribution(meteo_distribution, bin_join_factor=20)