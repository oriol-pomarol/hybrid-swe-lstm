"""Mean SWE per day of year and year over the Northern Hemisphere and each mountain range.

Inputs: the daily maps in SOURCE_PATHS[<source>] (1980-10-01 to 2020-09-30),
BASE_PATH/data/analysis_mask_no10s.tif and the GMBA ranges of config.MOUNTAIN_NAMES.

Output: results/temp/<source>_<region>_doy_avg_swe.csv

Usage: python src/3_analysis/compute_doy_swe.py --source {era5_land,hybrid}
"""
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import rasterio
from rasterio.features import rasterize
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared
import local_paths

###############################################################################
# 1. Compute the DOY SWE averaged over all pixels
###############################################################################
def get_noleap_doy(date: pd.Timestamp) -> int:
    """Get the day of year (DOY) for a given date, ignoring 29th February (leap day)."""
    doy = date.dayofyear
    if date.is_leap_year and date.month > 2:
        doy -= 1
    return doy

def compute_doy_swe(source: str):
    """Compute the DOY SWE for the given source."""

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

    # Start a dict to store the average DOY SWE values
    doy_avg_swe_dict = {region: {doy: {} for doy in range(1, 366)} for region in regions.keys()}

    # Daily maps of this source: the folder from local_paths, the file name per date
    swe_dir = Path(local_paths.SOURCE_PATHS[source])
    swe_file = {"era5_land": "sd_ERA5-Land_%d_%m_%Y.tif", "hybrid": "SWE_%d_%m_%Y.tif"}[source]
    for date in tqdm(pd.date_range(start="1980-10-01", end="2020-09-30")):

        # Skip leap days for simplicity
        if date.month == 2 and date.day == 29:
            continue

        swe_path = swe_dir / date.strftime(swe_file)

        with rasterio.open(swe_path) as src:
            swe_data = np.ma.filled(src.read(1, masked=True), np.nan)
            swe_data *= shared.SOURCE_TO_MM[source]
            for region, mask in regions.items():
                valid_swe_values = swe_data[mask]
                doy_avg_swe_dict[region][get_noleap_doy(date)][date.year] = np.mean(valid_swe_values)

    # Save the DOY SWE values to a CSV file
    for region in regions.keys():
        doy_swe_df = pd.DataFrame(doy_avg_swe_dict[region]).sort_index()
        path = local_paths.BASE_PATH / "results" / "temp" / f"{source}_{region.replace(' ', '_').lower()}_doy_avg_swe.csv"
        doy_swe_df.to_csv(path, index_label='year')
        print(f"Saved DOY SWE values for {region} to CSV file: {path}")


###############################################################################
# 2. Main function to parse arguments and call the computation
###############################################################################

if __name__ == "__main__":

    # Parse command-line arguments
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=local_paths.SOURCE_PATHS.keys(), default="era5_land")
    args = p.parse_args()

    # Ensure that the source is valid
    if args.source not in local_paths.SOURCE_PATHS:
        raise ValueError(f"Invalid source: {args.source}. Valid sources are: {list(local_paths.SOURCE_PATHS.keys())}")

    # Compute the DOY SWE for the specified source
    results = compute_doy_swe(args.source)