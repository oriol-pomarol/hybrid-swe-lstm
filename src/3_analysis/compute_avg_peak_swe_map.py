"""Map of the average yearly peak SWE (snow years 1981-2020) of a gridded source.

Input: the daily maps in SOURCE_PATHS[<source>].
Output: results/temp/<source>_avg_peak_swe.tif

Usage: python src/3_analysis/compute_avg_peak_swe_map.py --source {era5_land,hybrid}
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import shared
import local_paths

###############################################################################
# 1. Compute the average peak SWE map
###############################################################################

def compute_avg_peak_swe(source: str):
    """Compute and save the average yearly peak SWE for every pixel."""
    reference_date = pd.Timestamp("1980-10-01")

    # Daily maps of this source: the folder from local_paths, the file name per date
    swe_dir = Path(local_paths.SOURCE_PATHS[source])
    swe_file = {"era5_land": "sd_ERA5-Land_%d_%m_%Y.tif", "hybrid": "SWE_%d_%m_%Y.tif"}[source]
    with rasterio.open(swe_dir / reference_date.strftime(swe_file)) as reference:
        profile = reference.profile.copy()
        shape = reference.shape

    peak_sum = np.zeros(shape, dtype=np.float64)
    peak_count = np.zeros(shape, dtype=np.uint16)

    for snow_year in range(1981, 2021):
        yearly_peak = np.full(shape, np.nan, dtype=np.float32)
        dates = pd.date_range(
            start=f"{snow_year - 1}-10-01",
            end=f"{snow_year}-09-30",
        )

        for date in tqdm(
            dates,
            desc=f"Processing dates for snow year {snow_year}",
            leave=False,
        ):
            with rasterio.open(swe_dir / date.strftime(swe_file)) as dataset:
                swe = np.ma.filled(dataset.read(1, masked=True), np.nan)
                swe *= shared.SOURCE_TO_MM[source]

            yearly_peak = np.fmax(yearly_peak, swe)

        valid_peak = np.isfinite(yearly_peak)
        peak_sum[valid_peak] += yearly_peak[valid_peak]
        peak_count[valid_peak] += 1

    average_peak = np.full(shape, np.nan, dtype=np.float32)
    valid_pixels = peak_count > 0
    average_peak[valid_pixels] = (
        peak_sum[valid_pixels] / peak_count[valid_pixels]
    ).astype(np.float32)

    output_path = local_paths.BASE_PATH / "results" / "temp" / f"{source}_avg_peak_swe.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(average_peak, 1)

    print(f"Saved average peak SWE map to {output_path}")

###############################################################################
# 2. Main function to parse arguments and call the computation
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=local_paths.SOURCE_PATHS.keys(), default="era5_land")
    args = parser.parse_args()
    compute_avg_peak_swe(args.source)
