"""Compute the share of HySWE domain pixels in each (snow class, elevation band) stratum.

These weights are used by the weighted station metrics. Elevation bands: 500- (<= 500 m),
500-1500 and 1500+.

Inputs (under BASE_PATH/data): swe_data/SnowClass_GL_01km_30.0arcsec_2021_v01.0.nc,
topo_data/Amatulli_2016/elevation_1KMmn_GMTEDmn.tif and valid_swe_regression_flags.nc.

Output (in --out-dir, default BASE_PATH/data): CLASS_ELEV_WEIGHTS.csv

Usage: python src/1_preprocessing/compute_stratum_weights.py [--out-dir DIR]
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared
import local_paths

# Snow class codes of the SnowClass dataset
CLASS_DICT = {
    1: "Tundra",
    2: "Boreal Forest",
    3: "Maritime",
    4: "Ephemeral",
    5: "Prairie",
    6: "Montane Forest",
    7: "Ice",
    8: "Ocean",
}


def compute_weights() -> pd.DataFrame:
    data_dir = local_paths.BASE_PATH / 'data'

    # Snow classes on the 0-84N grid, latitude descending
    swe_classes_ds = xr.open_dataset(data_dir / 'swe_data' / 'SnowClass_GL_01km_30.0arcsec_2021_v01.0.nc',
                                     engine='netcdf4', decode_coords='all')
    swe_classes_ds = swe_classes_ds.sel(lat=slice(0, 84))
    swe_classes_ds = swe_classes_ds.sortby('lat', ascending=False)

    # Mean elevation, Northern Hemisphere
    mean_elevation_ds = xr.open_dataset(data_dir / 'topo_data' / 'Amatulli_2016' / 'elevation_1KMmn_GMTEDmn.tif')
    mean_elevation_ds = mean_elevation_ds.sel(y=slice(90, 0)).squeeze()
    mean_elevation_ds = mean_elevation_ds.rename({'y': 'lat', 'x': 'lon'})

    # HySWE domain
    flags_ds = xr.open_dataset(data_dir / 'valid_swe_regression_flags.nc')

    # The three grids must match the SnowClass grid
    tolerance = 1e-9
    for ds_name, ds in zip(['Elevation', 'Validity flags'], [mean_elevation_ds, flags_ds]):
        if len(ds['lat']) != len(swe_classes_ds['lat']) or len(ds['lon']) != len(swe_classes_ds['lon']):
            raise ValueError(f"Coordinate length mismatch found in {ds_name} dataset.")
        lat_diff = np.abs(ds['lat'].values - swe_classes_ds['lat'].values)
        lon_diff = np.abs(ds['lon'].values - swe_classes_ds['lon'].values)
        if not (np.all(lat_diff < tolerance) and np.all(lon_diff < tolerance)):
            raise ValueError(f"Coordinate mismatch found in {ds_name} dataset.")
        print(f"{ds_name} dataset coordinates match with Snow Class dataset within tolerance.")

    # Pairs of snow class and elevation over the domain
    swe_classes = swe_classes_ds['SnowClass'].values
    elevations = mean_elevation_ds['band_data'].values
    is_valid = flags_ds['is_valid'].values
    class_elevation_df = pd.DataFrame({
        'swe_class': swe_classes[is_valid == 1],
        'elevation': elevations[is_valid == 1],
    })

    # Elevation bands and their labels from the config (the same bands that stratify the station
    # split and the metrics): '500-', '500-1500', '1500+' (right-closed)
    class_elevation_df['elevation_bin'] = pd.cut(class_elevation_df['elevation'],
                                                 bins=config.ELEVATION_BINS,
                                                 labels=shared.ELEVATION_LABELS)

    # Pixel counts per stratum, as fractions of the total
    class_elevation_counts = class_elevation_df.groupby(['swe_class', 'elevation_bin'], observed=True).size().unstack(fill_value=0)
    class_elevation_counts.index = class_elevation_counts.index.map(CLASS_DICT)
    print(class_elevation_counts)
    return class_elevation_counts.div(class_elevation_counts.sum().sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='Folder for the output (default: BASE_PATH/data). Use another folder to compare with the existing file.')
    args = parser.parse_args()
    out_dir = args.out_dir or local_paths.BASE_PATH / 'data'

    weights = compute_weights()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'CLASS_ELEV_WEIGHTS.csv'
    weights.to_csv(out_path)
    print(f"Saved the stratum weights to {out_path}")
