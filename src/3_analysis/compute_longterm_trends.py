"""Yearly peak SWE and core snow season length, averaged over the Northern Hemisphere and each mountain range.

Inputs: the daily maps in SOURCE_PATHS[<source>] (snow years 1981-2020), BASE_PATH/data/analysis_mask_no10s.tif
and the GMBA ranges of config.MOUNTAIN_NAMES (BASE_PATH/data/GMBA_Inventory_v2.0_standard_300/).

Output: results/temp/<source>_<region>_yearly_<variable>.csv

Usage: python src/3_analysis/compute_longterm_trends.py --source {era5_land,hybrid} --variables peak_swe css_length
"""
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import rasterio
from rasterio.features import rasterize
from dataclasses import dataclass, field
from typing import Callable, Dict
from copy import deepcopy
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared
import local_paths

###############################################################################
# 0. Define the metrics
###############################################################################

@dataclass
class YearlyMetric:
    name: str                      # short name, for saving/plotting
    update_fn: Callable            # (state, pixel_values) -> new_state   (state is a tuple)
    finalize_fn: Callable = lambda state: state[0]   # state -> final DataArray
    init_fn: Callable = None       # (n_pixels) -> initial state tuple
    n_states: int = 1               # used only if init_fn is None (defaults to zeros)
    plot_kwargs: dict = field(default_factory=dict)

    def init_year(self, n_pixels: int):
        if self.init_fn is not None:
            self.state = self.init_fn(n_pixels)
        else:
            self.state = list(np.zeros(n_pixels) for _ in range(self.n_states))

    def update(self, pixel_values: np.ndarray):
        self.state = self.update_fn(self.state, pixel_values)

    def finalize(self):
        return self.finalize_fn(self.state)

def update_peak_swe(state, pixel_values):
    """Update the running maximum of the SWE values for each pixel."""
    state[0] = np.maximum(state[0], pixel_values)
    return state

def update_css_length(state, pixel_values):
    """Update the current and longest consecutive snow-covered length for each pixel."""
    THRESHOLD = 5 # mm
    snow_covered = pixel_values > THRESHOLD
    state[1] = np.where(snow_covered, state[1] + 1, 0)
    state[0] = np.maximum(state[0], state[1])
    return state

VARIABLE_METRICS = {
    "peak_swe": YearlyMetric(name="peak_swe", update_fn=update_peak_swe, n_states=1),
    "css_length": YearlyMetric(name="css_length", update_fn=update_css_length, n_states=2),
}

###############################################################################
# 1. Compute the yearly peak SWE averaged over all pixels
###############################################################################
def compute_yearly_metrics(source: str, metrics: Dict[str, YearlyMetric]):
    """Compute the yearly metrics for the given source and metric names."""

    # Read the mask for valid pixels
    with rasterio.open(local_paths.BASE_PATH / 'data' / 'analysis_mask_no10s.tif') as src:
        valid_mask = src.read(1).astype(bool)
        
        # Read the mountain mask shapefile and create a mask for each mountain range
        mountain_mask_gdf = gpd.read_file(local_paths.BASE_PATH / 'data' / 'GMBA_Inventory_v2.0_standard_300' / 'GMBA_Inventory_v2.0_standard_300.shp')
        regions = {"Northern Hemisphere": valid_mask}
        for mountain in config.MOUNTAIN_NAMES:
            geometry = mountain_mask_gdf[mountain_mask_gdf["MapName"] == mountain].geometry
            mountain_mask = rasterize(geometry, out_shape=src.shape, transform=src.transform, all_touched=True)
            regions[mountain] = np.logical_and(mountain_mask, valid_mask)

    for region, mask in regions.items():
        print(f"Number of valid pixels in the {region}: {np.sum(mask)}")

    stats_columns = ['avg', 'std', '0th', '1th', '5th', '10th', '25th', '50th', '75th', '90th', '95th', '99th', '100th']
    yearly_metric_dfs = {
        region_name: {
            metric_name: pd.DataFrame(
                index=pd.date_range(start='1980-01-01', end='2020-12-31', freq='YE'),
                columns=stats_columns,
            )
            for metric_name in metrics
        }
        for region_name in regions
    }

    for snow_year in range(1981, 2021):

        print(f"Computing yearly metrics for snow year {snow_year}...")

        region_metrics = {}
        for region, mask in regions.items():
            region_metrics[region] = {
                name: deepcopy(metric) for name, metric in metrics.items()
            }
            for metric in region_metrics[region].values():
                metric.init_year(np.sum(mask))

        # Daily maps of this source: the folder from local_paths, the file name per date
        swe_dir = Path(local_paths.SOURCE_PATHS[source])
        swe_file = {"era5_land": "sd_ERA5-Land_%d_%m_%Y.tif", "hybrid": "SWE_%d_%m_%Y.tif"}[source]
        for date in tqdm(pd.date_range(start=f"{snow_year-1}-10-01", end=f"{snow_year}-09-30"), desc=f"Processing dates for snow year {snow_year}"):
            swe_path = swe_dir / date.strftime(swe_file)

            with rasterio.open(swe_path) as src:
                swe_data = np.ma.filled(src.read(1, masked=True), np.nan)
                swe_data *= shared.SOURCE_TO_MM[source]
                for region, mask in regions.items():
                    valid_swe_values = swe_data[mask]
                    for metric in region_metrics[region].values():
                        metric.update(valid_swe_values)

        # Finalize the metrics for this snow year
        for region, region_metric_dict in region_metrics.items():
            for var_name, metric in region_metric_dict.items():
                yearly_values = metric.finalize()

                # Compute the average, std and several percentiles for this snow year.
                avg, std = np.mean(yearly_values), np.std(yearly_values)
                percentiles = np.percentile(yearly_values, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])


                yearly_metric_dfs[region][var_name].loc[f"{snow_year}"] = [avg, std] + list(percentiles)
                # print(f"Yearly {var_name} for {region}, snow year {snow_year}: avg={avg}, std={std}, percentiles={percentiles}")

    # Save the yearly metrics to a CSV file
    for region, region_metric_dfs in yearly_metric_dfs.items():
        region_name = region.replace(' ', '_').lower()
        for var_name, yearly_metric_df in region_metric_dfs.items():
            path = local_paths.BASE_PATH / "results" / "temp" / f"{source}_{region_name}_yearly_{var_name}.csv"
            yearly_metric_df.to_csv(path, index_label='year')
            print(f"Saved yearly {var_name} values for {region} to CSV file: {path}")

###############################################################################
# 2. Main function to parse arguments and call the computation
###############################################################################

if __name__ == "__main__":

    # Parse command-line arguments
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=local_paths.SOURCE_PATHS.keys(), default="era5_land")
    p.add_argument("--variables", nargs="+", choices=VARIABLE_METRICS.keys(), required=True)
    args = p.parse_args()

    # Ensure that the sources and variables are valid
    if args.source not in local_paths.SOURCE_PATHS:
        raise ValueError(f"Invalid source: {args.source}. Valid sources are: {list(local_paths.SOURCE_PATHS.keys())}")
    for var in args.variables:
        if var not in VARIABLE_METRICS:
            raise ValueError(f"Invalid variable: {var}. Valid variables are: {list(VARIABLE_METRICS.keys())}")

    # Compute the yearly metrics for the specified source and variables
    metrics = {name: VARIABLE_METRICS[name] for name in args.variables}
    results = compute_yearly_metrics(args.source, metrics)