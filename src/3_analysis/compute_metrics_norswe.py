"""Validate a gridded SWE source against the independent NorSWE stations (paper Figs. 3 and 6, Table 6).

Uses the NorSWE stations that are not SNOTEL and are more than 1' from every NH-SWE station.

Inputs: BASE_PATH/data/swe_data/NorSWE/NorSWE-NorEEN_1979-2021_v3.nc, STATION_METADATA.csv and the daily
maps in SOURCE_PATHS[<source>] (or, with --source_from_presaved, data/temp/norswe_<source>_swe_values.csv).

Outputs: data/temp/norswe_<source>_swe_values.csv (unless presaved), results/temp/norswe_valid_stations.csv,
results/temp/norswe_<source>_values_mt_<code>.csv and results/norswe_<source>_metrics.csv
(MAE, RMSE, bias and NSE per measurement type).

Usage: python src/3_analysis/compute_metrics_norswe.py --source {era5_land,hybrid} [--source_from_presaved]
"""
import argparse
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
import rasterio
from rasterio.transform import rowcol
from netCDF4 import Dataset
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import shared
import local_paths

norswe_path = local_paths.BASE_PATH / "data/swe_data/NorSWE/NorSWE-NorEEN_1979-2021_v3.nc"

def compute_norswe_metrics(source: str, source_from_presaved: bool = True):

    METRIC_FUNCTIONS = {
        'MAE': lambda x, y: np.nanmean(np.abs(np.subtract(x, y))),
        'RMSE': lambda x, y: np.sqrt(np.nanmean((np.subtract(x, y)) ** 2)),
        'Bias': lambda x, y: np.nanmean(np.subtract(y, x)),
        'NSE': lambda x, y: 1 - (np.nansum((np.subtract(x, y)) ** 2) / np.nansum((x - np.nanmean(x)) ** 2)),
    }

    ds = Dataset(norswe_path, 'r')
    vars = list(ds.variables.keys())
    ds.close()

    ###############################################################################
    # 1. Load and filter the NorSWE dataset
    ###############################################################################

    print("Loading NorSWE dataset...")
    # Load the dataset with xarray, dropping unnecessary variables
    load_vars = ['snw', 'time', 'lat', 'lon', 'type_mes', 'elevation', 'station_id', 'station_name', 'source']
    drop_vars = [v for v in vars if v not in load_vars]
    print("Dropping variables:", drop_vars)

    norswe_ds = xr.open_dataset(norswe_path,
                         engine='netcdf4',
                         chunks=None,
                         decode_times=True,
                         mask_and_scale=False,
                         drop_variables=drop_vars)


    # Select only the stations that are not from the Snotel source
    norswe_sources = norswe_ds['source'].values.astype('U')
    snotel_mask = np.char.startswith(norswe_sources, 'Snotel')
    keep_idx = np.nonzero(~snotel_mask)[0]
    norswe_ds_no_snotel = norswe_ds.isel({'station_id': keep_idx})
    print(f"Removed Snotel stations: {np.sum(snotel_mask)}")

    # Load the NH SWE metadata
    nh_swe_df = pd.read_csv(local_paths.BASE_PATH / 'data/STATION_METADATA.csv', index_col=0)
    nh_swe_df['LON'] = nh_swe_df['LON'].apply(lambda x: x - 360 if x > 180 else x)
    nh_swe_df = nh_swe_df[nh_swe_df['LAT'] >= 0]

    # Check which stations are present both in the NorSWE dataset and in the metadata
    min_distance_threshold = 1/60 # 1 minute
    is_in_nhswe = []
    for station_id in norswe_ds_no_snotel['station_id'].values:
        station_lon = norswe_ds_no_snotel['lon'].sel(station_id=station_id).values.item()
        station_lat = norswe_ds_no_snotel['lat'].sel(station_id=station_id).values.item()
        distances = np.sqrt((nh_swe_df['LON'] - station_lon)**2 + (nh_swe_df['LAT'] - station_lat)**2)
        min_distance = distances.min()
        if min_distance < min_distance_threshold:
            is_in_nhswe.append(True)
        else:
            is_in_nhswe.append(False)
    norswe_ds_no_snotel = norswe_ds_no_snotel.assign(is_in_nhswe=is_in_nhswe)

    # Select only the stations that are not overlapping
    only_norswe_mask = ~norswe_ds_no_snotel['is_in_nhswe'].values
    keep_idx = np.nonzero(only_norswe_mask)[0]
    norswe_ds_valid = norswe_ds_no_snotel.isel({'station_id': keep_idx})
    norswe_ds_valid = norswe_ds_valid.drop_vars('is_in_nhswe')
    valid_station_ids_all = norswe_ds_valid['station_id'].values
    valid_station_lats_all = norswe_ds_valid['lat'].values
    valid_station_lons_all = norswe_ds_valid['lon'].values
    time_index = pd.Index(norswe_ds_valid['time'].values)

    print(f"Removed overlapping stations with NH-SWE: {np.sum(~only_norswe_mask)}")
    print(f"Number of valid stations after filtering: {len(valid_station_ids_all)}")

    # Save the filtered stations, written through a temporary file per source
    stations_dir = local_paths.BASE_PATH / "results" / "temp"
    stations_dir.mkdir(parents=True, exist_ok=True)
    stations_df = pd.DataFrame({
        'station_id': np.asarray(valid_station_ids_all).astype('U'),
        'lat': valid_station_lats_all,
        'lon': valid_station_lons_all,
        'type_mes': norswe_ds_valid['type_mes'].values,
    })
    stations_tmp_path = stations_dir / f"norswe_valid_stations_{source}.tmp"
    stations_df.to_csv(stations_tmp_path, index=False)
    stations_tmp_path.replace(stations_dir / "norswe_valid_stations.csv")
    print(f"Saved the valid NorSWE stations to: {stations_dir / 'norswe_valid_stations.csv'}")


    ###############################################################################
    # 2. Retrieve source SWE values for each station and date
    ###############################################################################
    simulation_range = pd.date_range(start='1980-10-01', end='2020-09-30', freq='D')
    simulation_time_positions = time_index.get_indexer(simulation_range)
    if np.any(simulation_time_positions < 0):
        missing_dates = simulation_range[simulation_time_positions < 0]
        raise ValueError(f"NorSWE is missing dates required for the simulation range, starting with {missing_dates[0]}")

    if source_from_presaved:
        print(f"Loading previously extracted {source} SWE values from CSV file...")
        # Load the source SWE values from the CSV file
        source_swe_df = pd.read_csv(local_paths.BASE_PATH / f"data/temp/norswe_{source}_swe_values.csv", index_col='station_id')
        source_swe = source_swe_df.to_dict(orient='index')

    else:
        print(f"Extracting {source} SWE values for each station and date...")
        source_swe = {}
        # Daily maps of this source: the folder from local_paths, the file name per date
        swe_dir = Path(local_paths.SOURCE_PATHS[source])
        swe_file = {"era5_land": "sd_ERA5-Land_%d_%m_%Y.tif", "hybrid": "SWE_%d_%m_%Y.tif"}[source]
        # This reference raster is only used to obtain the grid transform and map
        # NorSWE station coordinates to source SWE pixel indices.
        reference_source_swe_path = swe_dir / simulation_range[0].strftime(swe_file)
        with rasterio.open(reference_source_swe_path) as src:
            valid_station_rows_all, valid_station_cols_all = rowcol(
                src.transform,
                valid_station_lons_all,
                valid_station_lats_all,
                op=np.round,
            )
        valid_station_rows_all = np.asarray(valid_station_rows_all, dtype=int)
        valid_station_cols_all = np.asarray(valid_station_cols_all, dtype=int)
        for date_idx, date in enumerate(tqdm(simulation_range, total=len(simulation_range), desc=f"Loading {source} SWE data")):
            date_str = date.strftime('%d_%m_%Y')

            # Select only the stations that have valid SWE values for this date.
            date_pos = simulation_time_positions[date_idx]
            valid_mask = ~np.isnan(norswe_ds_valid['snw'].isel(time=date_pos).values)
            if not valid_mask.any():
                print(f"No valid SWE values for {date_str}. Skipping...")
                continue
            valid_station_ids = valid_station_ids_all[valid_mask]
            valid_station_rows = valid_station_rows_all[valid_mask]
            valid_station_cols = valid_station_cols_all[valid_mask]

            # Load the source SWE dataset for this date
            source_swe_path = swe_dir / date.strftime(swe_file)
            with rasterio.open(source_swe_path) as src:
                source_swe_data = np.ma.filled(src.read(1, masked=True), np.nan)
                source_swe_data *= shared.SOURCE_TO_MM[source]

            # Retrieve the SWE values from the source dataset at the locations of the valid stations
            source_swe_values = source_swe_data[valid_station_rows, valid_station_cols]
            for station_id, source_swe_value in zip(valid_station_ids, source_swe_values):
                if station_id not in source_swe:
                    source_swe[station_id] = {}
                source_swe[station_id][date_str] = source_swe_value

        # Save the source SWE values to a CSV file
        source_swe_df = pd.DataFrame.from_dict(source_swe, orient='index')
        vals_path = local_paths.BASE_PATH / "data/temp" / f"norswe_{source}_swe_values.csv"
        vals_path.parent.mkdir(parents=True, exist_ok=True)
        source_swe_df.to_csv(vals_path, index_label='station_id')
        print(f"Saved {source} SWE values to CSV file: {vals_path}")

    ###############################################################################
    # 3. Compute the metrics for each measurement type
    ###############################################################################
    print("Computing metrics for each measurement type...")
    code_to_mt = {
        0: "Multi-point manual snow survey",
        1: "Single-point manual SWE measurement",
        2: "Snow pillow or snow scale",
        3: "Passive gamma",
        4: "GNSS/GPS methods",
        5: "Cosmic ray attenuation",
        6: "Time domain reflectometry",
        63: "Missing value",
        64: "Airborne gamma SWE",
        # 7–62 are reserved
    }

    def compute_metrics(norswe_values, source_values, metrics_fncs):
        metrics = []
        if len(norswe_values) != len(source_values):
            raise ValueError(f"Length mismatch: NorSWE={len(norswe_values)}, {source.upper()} SWE={len(source_values)}")
        finite_mask = np.isfinite(norswe_values) & np.isfinite(source_values)
        norswe_values = norswe_values[finite_mask]
        source_values = source_values[finite_mask]
        for metric_name, metric_fn in metrics_fncs.items():
            try:
                metric_value = metric_fn(norswe_values, source_values)
            except Exception as e:
                print(f"Error computing metric {metric_name}: {e}")
                metric_value = np.nan
            metrics.append((metric_value))
        return metrics

    mt_metrics = {}
    for mt_code in code_to_mt.keys():
        mt_name = code_to_mt[mt_code]
        print(f"Computing metrics for measurement type {mt_code} ({mt_name})...")

        # Select only the stations that have this measurement type
        mt_mask = norswe_ds_valid['type_mes'].values == mt_code
        if np.sum(mt_mask) == 0:
            print(f"No stations with measurement type {mt_code}. Skipping...")
            continue
        mt_station_ids = norswe_ds_valid['station_id'].values[mt_mask]

        # Compute the metrics for each station and date
        mt_values_source, mt_values_norswe = [], []
        mt_station_positions = np.nonzero(mt_mask)[0]
        for station_pos, station_id in tqdm(zip(mt_station_positions, mt_station_ids), total=len(mt_station_ids), desc=f"Computing metrics for measurement type {mt_code}"):
            norswe_station_values = norswe_ds_valid['snw'].isel(station_id=station_pos).values
            norswe_values = norswe_station_values[simulation_time_positions]
            source_values = np.asarray(list(source_swe[station_id].values()))
            if len(norswe_values) != len(source_values):
                raise ValueError(f"Length mismatch for station {station_id}: NorSWE={len(norswe_values)}, {source.upper()} SWE={len(source_values)}")
            mt_values_source.extend(source_values)
            mt_values_norswe.extend(norswe_values)

        # Clean the lists from non-finite values before computing the metrics
        THRESHOLD = 1e36
        finite_mask_norswe = np.isfinite(mt_values_norswe) & (np.abs(mt_values_norswe) < THRESHOLD)
        print(f" Number of NorSWE values for measurement type {mt_code}: {np.sum(finite_mask_norswe)}/{len(finite_mask_norswe)}")
        mt_values_norswe = np.array(mt_values_norswe)[finite_mask_norswe]
        mt_values_source = np.array(mt_values_source)[finite_mask_norswe]
        finite_mask_source = np.isfinite(mt_values_source) & (np.abs(mt_values_source) < THRESHOLD)
        mt_values_source[~finite_mask_source] = np.nan
        print(f" Number of SWE values used for metric computation of measurement type {mt_code}: {np.sum(finite_mask_source)}/{len(finite_mask_source)}")

        # Save the valid values to a CSV file for this measurement type
        mt_values_df = pd.DataFrame({
            'norswe': mt_values_norswe,
            f'{source}': mt_values_source,
        })
        mt_values_df.to_csv(local_paths.BASE_PATH / f"results/temp/norswe_{source}_values_mt_{mt_code}.csv", index=False)

        aggregated_metrics = compute_metrics(mt_values_norswe, mt_values_source, METRIC_FUNCTIONS)
        mt_metrics[mt_code] = aggregated_metrics

    # Save the metrics to a CSV file
    metrics_df = pd.DataFrame.from_dict(mt_metrics, orient='index', columns=list(METRIC_FUNCTIONS.keys()))
    metrics_df.to_csv(local_paths.BASE_PATH / f"results/norswe_{source}_metrics.csv", index_label='measurement_type')

###############################################################################
# 4. Main function to parse arguments and call the computation
###############################################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=local_paths.SOURCE_PATHS.keys(), default="era5_land")
    p.add_argument("--source_from_presaved", action="store_true", help="If set, load source SWE values from a previously saved CSV file instead of extracting them from the source dataset.")
    args = p.parse_args()

    # Ensure that the source is valid
    if args.source not in local_paths.SOURCE_PATHS:
        raise ValueError(f"Invalid source: {args.source}. Valid sources are: {list(local_paths.SOURCE_PATHS.keys())}")

    results = compute_norswe_metrics(args.source, args.source_from_presaved)