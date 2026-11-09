"""Histograms of the topographic predictors at the stations vs. over the HySWE domain.

Inputs: BASE_PATH/data/temp/static_predictors.csv, the GMTED layers in BASE_PATH/data/topo_data/Amatulli_2016/
and BASE_PATH/data/valid_swe_regression_flags.nc.
Outputs (results/predictor_distribution/): topo_distribution.json (counts; reused if present) and topo_distribution.png

Usage: python src/3_analysis/plot_topo_distribution.py
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
import local_paths
from variable_attrs import VARIABLE_ATTRS

# Define the paths to the topographic data
topo_path = local_paths.BASE_PATH / 'data' / 'topo_data'
temp_path = local_paths.BASE_PATH / 'data' / 'temp'
download_path = topo_path / 'Amatulli_2016'
results_path = local_paths.BASE_PATH / 'results' / 'predictor_distribution'
results_path.mkdir(parents=True, exist_ok=True)

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

def process_inference_data(mask_da):
    """Process inference data and compute histograms."""

    inference_counts = {}
    inference_edges = {}
    downloaded_files = download_path.glob('*_GMTED*.tif')
    for file in tqdm(downloaded_files, desc="Processing inference data"):

        var = file.stem.split('_')[0] + '_' + file.stem.split('_')[1][-2:]

        # Skip if the variable is not a predictor
        if VARIABLE_ATTRS.get(var).predictor != True:
            continue

        # Open the raster file and constrain it to the NH
        da = rioxarray.open_rasterio(file)
        da = da.where(da.y >= 0, drop=True)

        # Select the data
        data = da.data.squeeze()[mask_da].ravel()
        if data.size == 0:
            print(f"Warning: No data found for variable {var} in file {file}. Skipping.")
            continue

        # Set the fill value (-9999.0) to NaN
        data[data == -9999.0] = np.nan

        # Calculate the file histogram and update the global histogram
        hist, bin_edges = np.histogram(data, bins=1024)

        # Store the histogram counts and bin edges for the variable
        inference_counts[var] = hist.tolist()
        inference_edges[var] = bin_edges.tolist()

    return inference_counts, inference_edges

def process_station_data(bin_edges):
    """Process and aggregate all station data."""

    # Load the static data dataframe
    topo_df = pd.read_csv(temp_path / 'static_predictors.csv', index_col=0)

    # Retrieve the histograms for each variable given the edge bins from the stats
    station_counts = {}
    for variable, edges in bin_edges.items():

        # Compute the histogram counts for the station data
        counts, _ = np.histogram(topo_df[variable].values, bins=edges)

        # Store the histogram counts in the dictionary
        station_counts[variable] = counts.tolist()

    return station_counts

def save_counts(station_counts, inference_counts, bin_edges):
    """Save the counts to a JSON file."""

    # Store the results in a dictionary
    topo_distribution = {}
    for variable in bin_edges.keys():
        topo_distribution[variable] = {
            'bin_edges': bin_edges[variable],
            'station_counts': station_counts[variable],
            'inference_counts': inference_counts[variable]
        }

    # Save the results to a JSON file
    with open(results_path / f'topo_distribution.json', 'w') as f:
        json.dump(topo_distribution, f)

    return topo_distribution

def plot_topo_distribution(topo_distribution, bin_join_factor=20):
    """Plot the histograms of the topographic data."""

    # Get all variables from the histogram data
    variables = list(topo_distribution.keys())

    # Calculate grid dimensions for subplots
    n_vars = len(variables)
    n_cols = min(3, n_vars)  # Max 3 columns
    n_rows = (n_vars + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), sharey=True)
    axes = np.array(axes).reshape(-1)  # Flatten axes array for easy indexing

    # Loop through all variables
    for i, (var, var_distribution) in enumerate(topo_distribution.items()):
        ax = axes[i]
        
        # Get data for the current variable
        station_counts = np.array(var_distribution['station_counts'])
        edges = np.array(var_distribution['bin_edges'])
        inference_counts = np.array(var_distribution['inference_counts'])

        # Scale the edges according to the variable attributes
        scale_factor = VARIABLE_ATTRS.get(var).scale_factor
        offset = VARIABLE_ATTRS.get(var).offset
        edges = np.array(edges) * scale_factor + offset
        
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

        # Set title and labels
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

    plt.savefig(results_path / 'topo_distribution.png', dpi=300)

    return

###############################################################################
# MAIN SCRIPT
###############################################################################

# Load precomputed results if available
topo_dist_file = results_path / 'topo_distribution.json'
if topo_dist_file.exists():
    print(f"Loading precomputed results from {topo_dist_file}")
    with open(topo_dist_file, 'r') as f:
        topo_distribution = json.load(f)

else:
    print("Precomputed results not found, calculating from scratch...")

    # Load and process the inference data
    mask_da = load_inference_mask()
    inference_counts, bin_edges = process_inference_data(mask_da)

    # Load and process the station data
    station_counts = process_station_data(bin_edges)

    # Save results
    topo_distribution = save_counts(station_counts, inference_counts, bin_edges)

# Plot the histograms of the topographic data
plot_topo_distribution(topo_distribution, bin_join_factor=20)