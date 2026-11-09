"""Land-cover fractions at the stations vs. over the HySWE domain, per land-cover year.

Inputs: BASE_PATH/data/land_cover_data/NH_LC_YEARLY.csv, the aggregated maps in land_cover_data/aggregated/
and BASE_PATH/data/valid_swe_regression_flags.nc.
Outputs (results/predictor_distribution/): lc_freq_distribution.json (reused if present) and
lc_freq_yearly_distribution.png

Usage: python src/3_analysis/plot_lc_distribution.py
"""
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import shared
import local_paths

results_path = local_paths.BASE_PATH / 'results' / 'predictor_distribution'
results_path.mkdir(parents=True, exist_ok=True)

def load_station_data(land_cover_path):
    """Load the yearly land-cover counts of all stations (from extract_station_attributes.py)."""
    lc_yearly_df = pd.read_csv(land_cover_path / 'NH_LC_YEARLY.csv', index_col=['NHSWEID', 'year'])
    lc_yearly_df.columns = lc_yearly_df.columns.astype(int)
    return lc_yearly_df

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

def process_station_data(lc_yearly_df):
    """Process and aggregate all station data."""

    # Normalize the frequencies by pixel and year
    rel = lc_yearly_df.div(lc_yearly_df.sum(axis=1), axis=0)

    # Aggregate the class codes into the land cover class groups
    rel = rel.T.groupby(shared.LC_CLASS_GROUPS).sum().T

    # Calculate yearly mean, lower, and upper bounds for each class
    yearly = rel.groupby(level='year').mean()
    mean = yearly.mean()
    lower = mean - yearly.min()
    upper = yearly.max() - mean

    return mean, lower, upper

def process_inference_data(mask_da, years, land_cover_path):
    """Process inference data year by year, return mean, lower, upper."""

    # Load the land cover data for each year sequentially
    yearly_means = []
    for year in tqdm(years, desc="Processing inference LC data by year"):
        file = land_cover_path / 'aggregated' / f'aggregated_land_cover_{year}.nc'
        with xr.open_dataset(file, engine='netcdf4') as ds:

            # Ensure the y dimension matches with the mask
            if not (np.abs(ds['y'].values - mask_da.y.values) < 1e-5).all():
                print(f"Warning: y coordinates do not match for year {year}")

            # Mask the data using the inference mask and convert to DataFrame
            data = ds['lc_freq'].values[mask_da.values]
            df = pd.DataFrame(data, columns=list(shared.LC_CLASS_GROUPS))

        # Normalize the frequencies by pixel and year
        rel = df.div(df.sum(axis=1), axis=0)

        # Aggregate the class codes into the land cover class groups
        rel = rel.T.groupby(shared.LC_CLASS_GROUPS).sum().T

        # Append the yearly mean to the list
        yearly_means.append(rel.mean())

    # Create a DataFrame with the yearly means
    yearly_df = pd.DataFrame(yearly_means, index=years)

    # Calculate the mean, lower, and upper bounds
    mean = yearly_df.mean()
    lower = mean - yearly_df.min()
    upper = yearly_df.max() - mean

    return mean, lower, upper

def save_stats(station_stats, inference_stats):
    """Combine and save results to a JSON file."""

    # Store the results in a dictionary
    lc_distribution = {
        'classes': list(station_stats[0].index),
        'station': {
            'yearly_mean': station_stats[0].to_list(),
            'lower_err': station_stats[1].to_list(),
            'upper_err': station_stats[2].to_list(),
        },
        'inference': {
            'yearly_mean': inference_stats[0].to_list(),
            'lower_err': inference_stats[1].to_list(),
            'upper_err': inference_stats[2].to_list(),
        }
    }

    # Save the results to a JSON file
    with open(results_path / 'lc_freq_distribution.json', 'w') as f:
        json.dump(lc_distribution, f)
    print(f"Results saved to {results_path / 'lc_freq_distribution.json'}")

    return lc_distribution

def plot_lc_distribution(lc_distribution):
    """Plot the land cover distribution from the processed data."""

    # Prepare the data for plotting
    station_mean = lc_distribution['station']['yearly_mean']
    station_err = [lc_distribution['station']['lower_err'],
                   lc_distribution['station']['upper_err']]
    inference_mean = lc_distribution['inference']['yearly_mean']
    inference_err = [lc_distribution['inference']['lower_err'],
                     lc_distribution['inference']['upper_err']]
    classes = lc_distribution['classes']

    # Create a bar plot for the land cover class distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    indices = np.array(range(len(classes)))
    inference_bars = ax.bar(
        indices - bar_width/2, inference_mean, 
        yerr=inference_err, capsize=5,
        color='skyblue', edgecolor='steelblue', alpha=0.7, label='Inference Data', zorder=2,
        width=bar_width
    )
    station_bars = ax.bar(
        indices + bar_width/2, station_mean, 
        yerr=station_err, capsize=5,
        color='coral', edgecolor='orangered', alpha=0.7, label='Station Data', zorder=1,
        width=bar_width
    )

    # Customize the plot
    ax.set_xlabel("Land Cover Class")
    ax.set_ylabel("Frequency", color='black')
    ax.set_xticks(indices)
    ax.set_xticklabels([lc_class.upper() for lc_class in classes], rotation=45)
    plt.title("Average LC Class Distribution (min/max range, grouped by year)")
    ax.legend(handles=[inference_bars, station_bars], 
              labels=['Inference Data', 'Station Data'])
    plt.tight_layout()

    # Save the plot
    plt.savefig(results_path / 'lc_freq_yearly_distribution.png', dpi=300)

    return

################################################################################
# MAIN SCRIPT
################################################################################

lc_dist_file = results_path / 'lc_freq_distribution.json'
years = list(shared.LC_YEARS)
land_cover_path = local_paths.BASE_PATH / 'data' / 'land_cover_data'

# Load precomputed results if available
if lc_dist_file.exists():
    print(f"Loading precomputed results from {lc_dist_file}")
    with open(lc_dist_file, 'r') as f:
        lc_distribution = json.load(f)
else:
    print("Precomputed results not found, calculating from scratch...")

    # Load and process station data
    lc_yearly_df = load_station_data(land_cover_path)
    station_stats = process_station_data(lc_yearly_df)

    # Load and process inference data
    mask_da = load_inference_mask()
    inference_stats = process_inference_data(mask_da, years, land_cover_path)

    # Save results
    lc_distribution = save_stats(station_stats, inference_stats)

# Plot the land cover distribution
print("Plotting land cover distribution...")
plot_lc_distribution(lc_distribution)