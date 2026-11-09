"""Build the model inputs per station: the configured predictors, ERA5-Land SWE and observed SWE.

Inputs (under BASE_PATH/data): STATION_METADATA.csv, topo_data/NH_TOPO.csv,
land_cover_data/NH_LAND_COVER.csv (only with *_lc predictors), and per station
meteo_data/NH_METEO_ID_<id>.csv, swe_data/ERA5_LAND/NH_ERA5_SWE_ID_<id>.csv and
swe_data/NH_SWE/files/NH_SWE_ID_<id>.csv (gap-filled values dropped).

Filters: stations more than 500 m outside their pixel's elevation range; snow years with a gap of
more than 120 days in Nov-Apr; stations with less than 365 days of valid range, fewer than 30
observations or more than 5 missing predictor days in a row. Other predictor gaps are interpolated.

Outputs (BASE_PATH/data/temp/, replacing the old files): static_predictors.csv and
dynamic_predictors_ID_<id>.csv, starting shared.WARMUP_DAYS before the first observation.
Re-run after changing config.DYN_VARS or config.STATIC_VARS.

Usage: python src/1_preprocessing/save_training_sequences.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared
import local_paths
from variable_attrs import VARIABLE_ATTRS

# Define the min_range and min_targets for station dropping criteria
min_range = 365
min_targets = 30
max_gap_size = 120

# Retrieve the predictor columns from the variable attributes
dynamic_cols = VARIABLE_ATTRS.find(predictor=True, dynamic=True)
static_cols = VARIABLE_ATTRS.find(predictor=True, dynamic=False)

# Define the paths to load the meteo and SWE data
meteo_path = local_paths.BASE_PATH / 'data' / 'meteo_data'
era5_path = local_paths.BASE_PATH / 'data' / 'swe_data' / 'ERA5_LAND'
swe_path = local_paths.BASE_PATH / 'data' / 'swe_data' / 'NH_SWE' / 'files'

# Define a function to load the meteo and SWE data for a given station ID
def load_station_dynamic(station_id):

    # Read the meteo data for the station
    meteo_file = meteo_path / f'NH_METEO_ID_{station_id}.csv'
    meteo_df = pd.read_csv(meteo_file, index_col=0, parse_dates=[0])

    # Read the ERA5-Land SWE data for the station
    era5_file = era5_path / f'NH_ERA5_SWE_ID_{station_id}.csv'
    era5_df = pd.read_csv(era5_file, index_col=0, parse_dates=[0])
    era5_df.rename(columns={'SWE (mm)': 'era5_swe'}, inplace=True)

    # Read the SWE data for the station
    swe_file = swe_path / f'NH_SWE_ID_{station_id}.csv'
    swe_df = pd.read_csv(swe_file, index_col=0, parse_dates=[0])
    swe_df.rename(columns={'SWE (mm)': 'swe'}, inplace=True)
    swe_df = swe_df[swe_df['gapfill_flag'].isna()]

    # Join the dynamic data into a single DataFrame
    dynamic_df = meteo_df.join(swe_df, how='outer').join(era5_df, how='outer')

    return dynamic_df, station_id

# Define a function to check if the station contains large missing clusters
def has_large_missing_clusters(df, threshold=5):

    # Check for NaN rows
    nan_rows = df.isna().any(axis=1)

    # Skip if there are not sufficient missing values
    if nan_rows.sum() < threshold:
        return False

    # Check for large missing clusters
    nan_clusters = nan_rows.astype(int).groupby((nan_rows != nan_rows.shift()).cumsum()).sum()
    if (nan_clusters > threshold).any():
            return True
    return False

def get_snow_year(date):
    """Get the snow year for a given date, starting in October."""
    return date.year + 1 if date.month >= 10 else date.year

def find_largest_consecutive_gap(dates):
    """Find the largest consecutive gap in a series of dates."""
    arr = dates.values.astype('datetime64[D]').astype(int)
    if arr.size == 0:
        largest_consecutive_gap = 0
    elif arr.size == 1:
        largest_consecutive_gap = 1
    else:
        diffs = np.diff(arr)
        breaks = np.where(diffs != 1)[0]
        if breaks.size == 0:
            largest_consecutive_gap = arr.size
        else:
            splits = np.split(arr, breaks + 1)
            largest_consecutive_gap = max(s.size for s in splits)

    return largest_consecutive_gap

def remove_years_with_missing_months(series, months=[11, 12, 1, 2, 3, 4], max_gap=120):
    """Remove snow years that contain many missing values in specified months"""

    # Store the dropped snow years
    dropped_years = []
    n_obs_dropped = 0

    # Check for missing values in each snow year of the filtered DataFrame
    clean_swe = series.dropna()
    for snow_year, group in clean_swe.groupby(clean_swe.index.map(get_snow_year)):

        # Add missing values at the missing dates of that snow year
        full_date_range = pd.date_range(start=f"{snow_year-1}-10-01", end=f"{snow_year}-09-30", freq='D')
        assert group.index.isin(full_date_range).all(), f"Unexpected dates in group index for snow year {snow_year}"
        full_group = group.reindex(full_date_range, fill_value=np.nan)

        # Retrieve the dates of the selected months and find the largest consecutive gap
        missing_dates = full_group.index[full_group.isna()]
        missing_dates_in_months = missing_dates[missing_dates.month.isin(months)]
        largest_gap = find_largest_consecutive_gap(missing_dates_in_months)

        if largest_gap > max_gap:
            dropped_years.append(snow_year)
            n_obs_dropped += len(group)
    series.loc[series.index.map(get_snow_year).isin(dropped_years)] = np.nan

    return series, dropped_years, n_obs_dropped

def process_dynamic_station(args):
        
    # Unpack the arguments
    station_id, dynamic_df = args

    # Order by date, fill NaN values for any missing date
    dynamic_df.sort_index(inplace=True)
    dynamic_df = dynamic_df.asfreq('D')

    # Select the dynamic predictors
    dynamic_df = dynamic_df[[var.short_name for var in dynamic_cols]]

    # If tas and tasmin/tasmax are present, subtract tas from the latter
    for var in ['tasmin', 'tasmax']:
        if var in dynamic_df.columns and 'tas' in dynamic_df.columns:
            dynamic_df[var] = dynamic_df[var] - dynamic_df['tas']

    # Split between dynamic and target data and select the training range
    preds_df = dynamic_df.drop(columns=['swe']).copy()
    preds_df = preds_df.loc[config.START_WARMUP:config.END_DATE]

    swe_series = dynamic_df['swe'].copy()
    swe_series = swe_series.loc[config.START_DATE:config.END_DATE]

    # Retrieve the total number of snow years and observations
    clean_swe = swe_series.dropna()
    n_snow_years = len(clean_swe.groupby(clean_swe.index.map(get_snow_year)))
    n_observations = clean_swe.count()

    # Remove snow years with many missing SWE values
    swe_series, station_years_dropped, n_obs_dropped = remove_years_with_missing_months(swe_series, max_gap=max_gap_size)

    # Modify the start and end dates based on SWE availability
    new_start_date = swe_series.first_valid_index()
    new_end_date = swe_series.last_valid_index()

    # Check that there are new start and end dates are valid
    dropped_reason = None
    if new_start_date is None or new_end_date is None:
        dropped_reason = 'no_valid_range'
        return station_id, None, station_years_dropped, dropped_reason, n_snow_years, n_observations, n_obs_dropped

    # Check that there is at least 'min_range' valid SWE data
    if (new_end_date - new_start_date).days < min_range:
        dropped_reason = 'short_valid_range'
        return station_id, None, station_years_dropped, dropped_reason, n_snow_years, n_observations, n_obs_dropped

    # Check that there are at least 'min_targets' SWE measurements
    swe_new_range = swe_series.loc[new_start_date:new_end_date]
    if swe_new_range.count() < min_targets:
        dropped_reason = 'insufficient_swe'
        return station_id, None, station_years_dropped, dropped_reason, n_snow_years, n_observations, n_obs_dropped

    # Find the new start date for the warmup period
    warmup_length = pd.Timedelta(days=shared.WARMUP_DAYS)
    new_start_warmup = new_start_date - warmup_length

    # Check for large missing clusters in the predictor data
    preds_new_range = preds_df.loc[new_start_warmup:new_end_date]
    if has_large_missing_clusters(preds_new_range):
        dropped_reason = 'missing_clusters_preds'
        return station_id, None, station_years_dropped, dropped_reason, n_snow_years, n_observations, n_obs_dropped

    # Fill missing values in the predictor data using linear interpolation
    preds_new_range = preds_new_range.interpolate(method='linear')
    preds_new_range = preds_new_range.bfill().ffill()

    # Join the predictors and target back into a single DataFrame
    train_dyn_data = preds_new_range.join(swe_new_range, how='outer')

    return station_id, train_dyn_data, station_years_dropped, dropped_reason, n_snow_years, n_observations, n_obs_dropped

def save_training_sequences():

    # In test mode the outputs get the test prefix, so the real sequences are never overwritten
    prefix = shared.TEST_PREFIX

    # Load the metadata for the stations
    metadata_path = local_paths.BASE_PATH / 'data' / 'STATION_METADATA.csv'
    metadata_df = pd.read_csv(metadata_path, index_col=0)
    if config.TEST_MODE:
        metadata_df = metadata_df.loc[shared.TEST_STATION_IDS]
        print(len(metadata_df), "stations selected for testing.")

    print("Loaded station metadata")

    # Load the topographic data
    topo_path = local_paths.BASE_PATH / 'data' / 'topo_data' / 'NH_TOPO.csv'
    topo_df = pd.read_csv(topo_path, index_col=0)
    print("Loaded topographic data")

    # Merge into static data
    static_data = metadata_df.join(topo_df, how='left')

    # Load and merge the land cover data if a land cover predictor is used
    if any(var.endswith('_lc') for var in config.STATIC_VARS):
        lc_path = local_paths.BASE_PATH / 'data' / 'land_cover_data' / 'NH_LAND_COVER.csv'
        lc_df = pd.read_csv(lc_path, index_col=0)
        static_data = static_data.join(lc_df, how='left')
    print("Merged static data")

    # Check that all static columns defined in the config file are present in the static data
    missing_static_cols = [var for var in config.STATIC_VARS if var not in static_data.columns]
    if missing_static_cols:
        raise ValueError(f"Missing static columns in loaded data: {missing_static_cols}")

    # Remove the stations where the elevation reported in the metadata
    # differs from the elevation bounds in the topographic data by more than 500 m
    elev_diff_thresh = 500
    diff_from_min = np.abs(static_data['ELEV'] - static_data['elevation_mi'])
    diff_from_max = np.abs(static_data['ELEV'] - static_data['elevation_mx'])
    closest_diff = np.where(diff_from_min < diff_from_max, diff_from_min, diff_from_max)
    closest_diff[(static_data['ELEV'] >= static_data['elevation_mi']) & \
                 (static_data['ELEV'] <= static_data['elevation_mx'])] = 0
    stations_to_drop = static_data.index[np.abs(closest_diff) > elev_diff_thresh].tolist()
    static_data = static_data.drop(index=stations_to_drop)
    if stations_to_drop:
        print(f"Dropped {len(stations_to_drop)} stations due to elevation mismatch (>{elev_diff_thresh} m): {stations_to_drop}")

    # Iterate through the station IDs in the metadata DataFrame
    dynamic_data = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_station_dynamic, station_id): station_id for station_id in static_data.index}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading dynamic data"):
            dynamic_df, station_id = future.result()
            dynamic_data[station_id] = dynamic_df

    # Initialize dictionaries to store the training data and dropped stations/years
    dynamic_training = {}   # station_id -> dynamic DataFrame
    dropped_snow_years = {}      # station_id -> list of dropped snow years
    dropped_stations = {}   # station_id -> reason string

    # Initialize counters for the total number of snow years, observations and dropped observations
    n_snow_years = 0
    n_observations = 0
    n_obs_dropped = 0

    # Iterate through the dynamic predictors and select the training range in parallel
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_dynamic_station, (station_id, dynamic_df)): station_id for station_id, dynamic_df in dynamic_data.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing stations"):
            station_id, train_dyn_data, station_years_dropped, dropped_reason, n_station_snow_years, n_station_observations, n_station_obs_dropped = future.result()

            # Update global aggregates
            n_snow_years += n_station_snow_years
            n_observations += n_station_observations
            if station_years_dropped:
                dropped_snow_years[station_id] = station_years_dropped
                n_obs_dropped += n_station_obs_dropped

            if dropped_reason is not None:
                dropped_stations[station_id] = dropped_reason
                continue

            # Store successful training data
            dynamic_training[station_id] = train_dyn_data

    # Select the static predictors and drop the discarded stations
    static_predictors = static_data[[var.short_name for var in static_cols]]
    static_predictors = static_predictors.drop(index=list(dropped_stations.keys()))

    # Print the number of dropped snow years
    total_dropped_years = sum(len(years) for years in dropped_snow_years.values())
    print(f"Dropped a total of {total_dropped_years}/{n_snow_years} "
            f"({total_dropped_years/n_snow_years:.2%}) snow years across all stations due to large missing gaps in the SWE data during the snowpack season:\n{dropped_snow_years}")
    if total_dropped_years > 0:
        print(f"Total observations dropped due to missing snow years: {n_obs_dropped}/{n_observations} ({n_obs_dropped/n_observations:.2%})")

    # Print the dropped stations grouped by cause
    grouped = {}
    for sid, reason in dropped_stations.items():
        grouped.setdefault(reason, []).append(sid)

    ids = grouped['no_valid_range'] if 'no_valid_range' in grouped else []
    print(f"Dropped {len(ids)} stations due to no valid range of SWE data:\n{ids}")

    ids = grouped['short_valid_range'] if 'short_valid_range' in grouped else []
    print(f"Dropped {len(ids)} stations due to short valid range (< {min_range} days):\n{ids}")

    ids = grouped['insufficient_swe'] if 'insufficient_swe' in grouped else []
    print(f"Dropped {len(ids)} stations due to insufficient valid SWE data (< {min_targets}):\n{ids}")

    ids = grouped['missing_clusters_preds'] if 'missing_clusters_preds' in grouped else []
    print(f"Dropped {len(ids)} stations due to large missing clusters in dynamic predictors:\n{ids}")

    # Create a temporary data directory to save the training sequences
    output_path = local_paths.BASE_PATH/ 'data' / 'temp'
    output_path.mkdir(parents=True, exist_ok=True)

    # If not in test mode, clear the existing dynamic predictors files first
    if not config.TEST_MODE:
        print("Clearing training sequences directory...")
        for file in output_path.glob(f'dynamic_predictors_*.csv'):
            file.unlink()

    # Save the static predictors
    print("Saving training sequences...")
    static_predictors.to_csv(output_path / f'{prefix}static_predictors.csv')

    # Save the dynamic training data
    for station_id, train_dyn_data in dynamic_training.items():
        filename = output_path / f'{prefix}dynamic_predictors_ID_{station_id}.csv'
        train_dyn_data.to_csv(filename, index_label='Date')

    return

##############################################################################
# MAIN
##############################################################################

if __name__ == '__main__':
    save_training_sequences()