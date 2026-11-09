"""Station metrics of every run folder in results/final_model_comparison (paper Tables 3-5).

Settings below the imports: experiments_dir (None scores BASE_PATH/results itself), threshold
(predictions below it are set to 0) and apply_weights (weighting by snow class x elevation band with
BASE_PATH/data/CLASS_ELEV_WEIGHTS.csv). A folder named era5_land scores ERA5-Land instead of predictions.

Inputs: each run's station_splits.csv and predictions/NH_pSWE_{train,eval}.csv, the observations in
BASE_PATH/data/temp and STATION_METADATA.csv.
Output: <experiments_dir>/evaluation_metrics_thr_<threshold>_<weighted|unweighted>.csv.

Usage: python src/3_analysis/compute_metrics.py
"""
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared
import local_paths

# Set the number of workers for parallel processing
max_workers = 12

# Folder whose run folders are scored; None scores BASE_PATH/results itself
experiments_dir = local_paths.BASE_PATH / 'results' / 'final_model_comparison'

# Set up additional paths and global variables
prefix = shared.TEST_PREFIX
targets_dir = local_paths.BASE_PATH/ 'data' / 'temp'

# Threshold for snapping predicted SWE values below this value to zero.
threshold = 5.0 # 0.0 means no snapping

# Whether to weight metrics by (SnowClass, elevation band) strata. False means plain pooled metrics.
apply_weights = True

##############################################################################
# HELPER FUNCTIONS
##############################################################################

def _get_elev_band(elev):
    try:
        elev = float(elev)
    except Exception:
        return None
    for upper, label in zip(config.ELEVATION_BINS[1:-1], shared.ELEVATION_LABELS):
        if elev <= upper:
            return label
    return shared.ELEVATION_LABELS[-1]

def _get_snow_year(date):
    return date.year + 1 if date.month >= 10 else date.year

def _iter_weight_bins(weights_df):
    for swe_class in weights_df.index:
        for elev_band in weights_df.columns:
            key = (swe_class, elev_band)
            w_raw = float(weights_df.loc[swe_class, elev_band])
            yield key, w_raw

def _weighted_mean_over_available_bins(weights_df, value_sums, counts):
    weighted_sum = 0.0
    weight_sum = 0.0
    for key, w_raw in _iter_weight_bins(weights_df):
        n_bin = counts.get(key, 0)
        if n_bin == 0:
            continue
        weighted_sum += w_raw * (value_sums[key] / n_bin)
        weight_sum += w_raw
    if weight_sum == 0:
        return None
    return weighted_sum / weight_sum

def _longest_true_run(values):
    max_run = 0
    current_run = 0
    for value in values:
        if bool(value):
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run

def _compute_station_metrics(task):
    station_id, station_prediction, swe_class, elev, targets_dir, threshold, use_era5_swe = task

    elev_band = _get_elev_band(elev)
    if elev_band is None:
        print(f"Skipping station {station_id} due to invalid elevation band")
        return None

    # Load the observed SWE targets for this station
    filename = f"{shared.TEST_PREFIX}dynamic_predictors_ID_{station_id}.csv"
    val_target = pd.read_csv(targets_dir / filename, index_col=0, parse_dates=True)[['swe']]

    # Use ERA5 SWE as prediction for this station if use_era5_swe is True
    if use_era5_swe:
        station_prediction = pd.read_csv(targets_dir / filename, index_col=0, parse_dates=True)[['era5_swe']]
        station_prediction = station_prediction.rename(columns={'era5_swe': 'pred_swe'})

    # Snap predicted SWE values below threshold to zero
    station_prediction['pred_swe'] = station_prediction['pred_swe'].where(station_prediction['pred_swe'] >= threshold, 0.0)

    # Join targets and predictions on date index and drop NaNs
    df_joined = val_target.join(station_prediction, how='inner')
    assert not df_joined.empty, f"No overlapping dates for station {station_id}."
    df_joined = df_joined.dropna()

    # Compute errors and basic sums
    errors = (df_joined['pred_swe'] - df_joined['swe'])

    result = {
        'key': (swe_class, elev_band),
        'sq_errors': float((errors ** 2).sum()),
        'abs_errors': float(errors.abs().sum()),
        'err_sum': float(errors.sum()),
        'count': int(len(df_joined)),
        'obs_sum': float(df_joined['swe'].sum()),
        'obs_sq_sum': float((df_joined['swe'] ** 2).sum()),
        'snow_sq_errors': 0.0,
        'snow_err_sum': 0.0,
        'snow_count': 0,
        'yearly_peak_abs_error_sum': 0.0,
        'yearly_peak_error_sum': 0.0,
        'yearly_timing_abs_error_sum': 0.0,
        'yearly_timing_error_sum': 0.0,
        'yearly_count': 0,
        'yearly_snowpack_abs_error_sum': 0.0,
        'yearly_snowpack_error_sum': 0.0,
        'snow_tp': 0,
        'snow_tn': 0,
        'snow_fp': 0,
        'snow_fn': 0,
    }

    # Snow-on metrics (only where observed SWE > 0)
    df_snow_on = df_joined[df_joined['swe'] > 0]
    if not df_snow_on.empty:
        snow_errors = df_snow_on['pred_swe'] - df_snow_on['swe']
        result['snow_sq_errors'] = float((snow_errors ** 2).sum())
        result['snow_err_sum'] = float(snow_errors.sum())
        result['snow_count'] = int(len(df_snow_on))

    # Binary snow presence metrics (SWE > 0)
    obs_snow = df_joined['swe'] > 0
    pred_snow = df_joined['pred_swe'] > 0
    result['snow_tp'] = int((pred_snow & obs_snow).sum())
    result['snow_tn'] = int((~pred_snow & ~obs_snow).sum())
    result['snow_fp'] = int((pred_snow & ~obs_snow).sum())
    result['snow_fn'] = int((~pred_snow & obs_snow).sum())

    # Yearly metrics: require complete daily coverage over the whole snow year.
    grouped = df_joined.groupby(df_joined.index.map(_get_snow_year))
    for snow_year, group in grouped:
        year_start = f"{snow_year - 1}-10-01"
        year_end = f"{snow_year}-09-30"
        full_date_range = pd.date_range(start=year_start, end=year_end, freq='D')
        group_full = group.reindex(full_date_range)

        # Require full daily coverage for both observations and predictions over the snow year.
        if group_full[['swe', 'pred_swe']].isna().any().any():
            continue

        # Retrieve peak metrics within the full snow year
        obs_peak_date = group_full['swe'].idxmax()
        pred_peak_date = group_full['pred_swe'].idxmax()
        obs_peak_val = float(group_full.loc[obs_peak_date, 'swe'])
        pred_peak_val = float(group_full.loc[pred_peak_date, 'pred_swe'])

        peak_error = pred_peak_val - obs_peak_val
        timing_error_days = int((pred_peak_date - obs_peak_date).days)

        result['yearly_peak_abs_error_sum'] += abs(peak_error)
        result['yearly_peak_error_sum'] += peak_error
        result['yearly_timing_abs_error_sum'] += abs(timing_error_days)
        result['yearly_timing_error_sum'] += timing_error_days

        # Core snow season length metrics (longest continuous SWE>0 run) over the full year
        obs_snowpack_days = _longest_true_run((group_full['swe'] > 0).to_numpy())
        pred_snowpack_days = _longest_true_run((group_full['pred_swe'] > 0).to_numpy())
        snowpack_duration_error = pred_snowpack_days - obs_snowpack_days

        result['yearly_snowpack_abs_error_sum'] += abs(snowpack_duration_error)
        result['yearly_snowpack_error_sum'] += snowpack_duration_error

        # Use a single yearly count for all yearly metrics
        result['yearly_count'] += 1

    return result

##############################################################################
# CORE FUNCTIONS
##############################################################################

def compute_metrics(results_dir=local_paths.BASE_PATH / 'results', split_type='eval', use_era5_swe=False, threshold=5.0, apply_weights=True):

    # Load the metadata, of which select only the snow class and elevation
    metadata_path = local_paths.BASE_PATH / 'data' / 'STATION_METADATA.csv'
    metadata_df = pd.read_csv(metadata_path, index_col=0)
    metadata_df = metadata_df[['SnowClass', 'ELEV']]

    # Collect per-bin error statistics
    # Keyed by (SnowClass, elevation band)
    bin_sq_errors = {}      # sum((pred-obs)^2)
    bin_abs_errors = {}     # sum(|pred-obs|)
    bin_err_sums = {}       # sum(pred-obs)
    bin_counts = {}         # n
    bin_obs_sums = {}       # sum(obs)
    bin_obs_sq_sums = {}    # sum(obs^2)

    # Collect global snow-on (obs > 0) RMSE components
    bin_snow_sq_errors = {}   # sum((pred-obs)^2) where obs > 0
    bin_snow_err_sums = {}    # sum(pred-obs) where obs > 0
    bin_snow_counts = {}      # n where obs > 0

    # Collect yearly peak metrics (only years with full data in peak months)
    bin_yearly_peak_abs_error_sums = {}      # sum(|peak_pred - peak_obs|)
    bin_yearly_peak_error_sums = {}          # sum(peak_pred - peak_obs)
    bin_yearly_timing_abs_error_sums = {}    # sum(|timing error days|)
    bin_yearly_timing_error_sums = {}        # sum(timing error days)
    bin_yearly_counts = {}                    # n yearly peak comparisons
    
    # Collect yearly snowpack duration metrics (longest continuous SWE>0 run)
    bin_yearly_snowpack_abs_error_sums = {}  # sum(|pred_days - obs_days|)
    bin_yearly_snowpack_error_sums = {}      # sum(pred_days - obs_days)

    # Collect global binary snow presence metrics (SWE > 0)
    bin_snow_tp = {}
    bin_snow_tn = {}
    bin_snow_fp = {}
    bin_snow_fn = {}

    # Load the station IDs for the specified split type
    stations_split_file = results_dir / f'{prefix}station_splits.csv'
    stations_split_df = pd.read_csv(stations_split_file, index_col=0)
    station_ids = stations_split_df[stations_split_df['split'] == split_type].index.tolist()

    if not use_era5_swe:
        # Load the predictions for the split type
        filename = results_dir / 'predictions' / f'{prefix}NH_pSWE_{split_type}.csv'
        predictions_split = pd.read_csv(filename, parse_dates=['Date'])
        predictions_split['station_id'] = predictions_split['station_id'].astype(int)

        # Validate that the station ids set and the unique station ids in the predictions file match
        if set(station_ids) != set(predictions_split['station_id']):
            raise ValueError(
                f"Mismatch between station IDs in split file and predictions file for {split_type} set."
            )

        predictions_by_station = {
            int(sid): group[['Date', 'pred_swe']].set_index('Date')
            for sid, group in predictions_split.groupby('station_id')
        }
    station_tasks = [
        (
            int(station_id),
            predictions_by_station[int(station_id)] if not use_era5_swe else None,
            metadata_df.loc[int(station_id), 'SnowClass'],
            metadata_df.loc[int(station_id), 'ELEV'],
            targets_dir,
            threshold,
            use_era5_swe,
        )
        for station_id in station_ids
    ]
    print(f"  Prepared {len(station_tasks)} station tasks; using max_workers={max_workers}")

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            station_results = tqdm(
                executor.map(_compute_station_metrics, station_tasks),
                total=len(station_tasks),
                desc="Computing metrics",
                unit="station",
            )
            station_results = list(station_results)
    else:
        station_results = list(tqdm(map(_compute_station_metrics, station_tasks), total=len(station_tasks), desc="Computing metrics", unit="station"))

    for station_result in station_results:
        if station_result is None:
            continue
        key = station_result['key']
        bin_sq_errors[key] = bin_sq_errors.get(key, 0.0) + station_result['sq_errors']
        bin_abs_errors[key] = bin_abs_errors.get(key, 0.0) + station_result['abs_errors']
        bin_err_sums[key] = bin_err_sums.get(key, 0.0) + station_result['err_sum']
        bin_counts[key] = bin_counts.get(key, 0) + station_result['count']
        bin_obs_sums[key] = bin_obs_sums.get(key, 0.0) + station_result['obs_sum']
        bin_obs_sq_sums[key] = bin_obs_sq_sums.get(key, 0.0) + station_result['obs_sq_sum']

        if station_result['snow_count'] > 0:
            bin_snow_sq_errors[key] = bin_snow_sq_errors.get(key, 0.0) + station_result['snow_sq_errors']
            bin_snow_err_sums[key] = bin_snow_err_sums.get(key, 0.0) + station_result['snow_err_sum']
            bin_snow_counts[key] = bin_snow_counts.get(key, 0) + station_result['snow_count']

        if station_result['yearly_count'] > 0:
            bin_yearly_peak_abs_error_sums[key] = bin_yearly_peak_abs_error_sums.get(key, 0.0) + station_result['yearly_peak_abs_error_sum']
            bin_yearly_peak_error_sums[key] = bin_yearly_peak_error_sums.get(key, 0.0) + station_result['yearly_peak_error_sum']
            bin_yearly_timing_abs_error_sums[key] = bin_yearly_timing_abs_error_sums.get(key, 0.0) + station_result['yearly_timing_abs_error_sum']
            bin_yearly_timing_error_sums[key] = bin_yearly_timing_error_sums.get(key, 0.0) + station_result['yearly_timing_error_sum']
            bin_yearly_counts[key] = bin_yearly_counts.get(key, 0) + station_result['yearly_count']

            bin_yearly_snowpack_abs_error_sums[key] = bin_yearly_snowpack_abs_error_sums.get(key, 0.0) + station_result['yearly_snowpack_abs_error_sum']
            bin_yearly_snowpack_error_sums[key] = bin_yearly_snowpack_error_sums.get(key, 0.0) + station_result['yearly_snowpack_error_sum']

        bin_snow_tp[key] = bin_snow_tp.get(key, 0) + station_result['snow_tp']
        bin_snow_tn[key] = bin_snow_tn.get(key, 0) + station_result['snow_tn']
        bin_snow_fp[key] = bin_snow_fp.get(key, 0) + station_result['snow_fp']
        bin_snow_fn[key] = bin_snow_fn.get(key, 0) + station_result['snow_fn']

    print(f"  Collected results for {sum(bin_counts.values())} observations across {len(bin_counts)} bins")
    
    # If there are no observations, return

    # If weighting is disabled or in test mode, compute plain (unweighted) metrics across all data
    if not apply_weights or config.TEST_MODE:
        total_sq = sum(bin_sq_errors.values())
        total_abs = sum(bin_abs_errors.values())
        total_err = sum(bin_err_sums.values())
        total_obs = sum(bin_obs_sums.values())
        total_obs_sq = sum(bin_obs_sq_sums.values())
        total_count = sum(bin_counts.values())
        if total_count == 0:
            raise RuntimeError('No validation data available to compute metrics')

        mse = total_sq / total_count
        rmse = mse ** 0.5
        mae = total_abs / total_count
        bias = total_err / total_count
        if total_obs == 0:
            raise RuntimeError('Cannot compute %bias: sum(observations) is 0')
        pbias = 100.0 * (total_err / total_obs)
        denom = total_obs_sq - (total_obs ** 2) / total_count
        if denom == 0:
            raise RuntimeError('Cannot compute NSE: zero variance in observations')
        nse = 1.0 - (total_sq / denom)

        n_snow = int(sum(bin_snow_counts.values()))
        rmse_snow = None
        bias_snow = None
        if n_snow > 0:
            rmse_snow = (sum(bin_snow_sq_errors.values()) / n_snow) ** 0.5
            bias_snow = sum(bin_snow_err_sums.values()) / n_snow

        n_yearly = int(sum(bin_yearly_counts.values()))
        peak_mae_yearly = None
        peak_mb_yearly = None
        peak_timing_mae_yearly = None
        peak_timing_mb_yearly = None
        snowpack_duration_mae_yearly = None
        snowpack_duration_mb_yearly = None

        if n_yearly > 0:
            peak_mae_yearly = sum(bin_yearly_peak_abs_error_sums.values()) / n_yearly
            peak_mb_yearly = sum(bin_yearly_peak_error_sums.values()) / n_yearly
            peak_timing_mae_yearly = sum(bin_yearly_timing_abs_error_sums.values()) / n_yearly
            peak_timing_mb_yearly = sum(bin_yearly_timing_error_sums.values()) / n_yearly
            snowpack_duration_mae_yearly = sum(bin_yearly_snowpack_abs_error_sums.values()) / n_yearly
            snowpack_duration_mb_yearly = sum(bin_yearly_snowpack_error_sums.values()) / n_yearly

        total_tp = int(sum(bin_snow_tp.values()))
        total_tn = int(sum(bin_snow_tn.values()))
        total_fp = int(sum(bin_snow_fp.values()))
        total_fn = int(sum(bin_snow_fn.values()))
        total_class = total_tp + total_tn + total_fp + total_fn
        snow_accuracy = None if total_class == 0 else (total_tp + total_tn) / total_class
        snow_precision = None if (total_tp + total_fp) == 0 else total_tp / (total_tp + total_fp)
        snow_recall = None if (total_tp + total_fn) == 0 else total_tp / (total_tp + total_fn)

        print("  Returning unweighted metrics")

        return {
            'n_full': int(total_count),
            'mse_full': float(mse),
            'rmse_full': float(rmse),
            'mae_full': float(mae),
            'bias_full': float(bias),
            'pbias_full': float(pbias),
            'nse_full': float(nse),
            'n_snow': n_snow,
            'rmse_snow': None if rmse_snow is None else float(rmse_snow),
            'bias_snow': None if bias_snow is None else float(bias_snow),
            'n_yearly': n_yearly,
            'peak_mae_yearly': None if peak_mae_yearly is None else float(peak_mae_yearly),
            'peak_mb_yearly': None if peak_mb_yearly is None else float(peak_mb_yearly),
            'peak_timing_mae_yearly': None if peak_timing_mae_yearly is None else float(peak_timing_mae_yearly),
            'peak_timing_mb_yearly': None if peak_timing_mb_yearly is None else float(peak_timing_mb_yearly),
            'snowpack_duration_mae_yearly': None if snowpack_duration_mae_yearly is None else float(snowpack_duration_mae_yearly),
            'snowpack_duration_mb_yearly': None if snowpack_duration_mb_yearly is None else float(snowpack_duration_mb_yearly),
            'snow_accuracy': None if snow_accuracy is None else float(snow_accuracy),
            'snow_precision': None if snow_precision is None else float(snow_precision),
            'snow_recall': None if snow_recall is None else float(snow_recall),
        }

    # Weighted metrics (non-test mode).
    weights_path = local_paths.BASE_PATH / 'data' / 'CLASS_ELEV_WEIGHTS.csv'
    weights_df = pd.read_csv(weights_path, index_col=0)
    weights_arr = weights_df.astype(float).values
    total_weight = float(weights_arr.sum())
    if total_weight <= 0:
        raise RuntimeError('Weights sum to zero or negative')

    weighted_mse = 0.0
    weighted_mae = 0.0
    weighted_bias = 0.0
    pbias_num = 0.0
    pbias_den = 0.0
    ss_res_global = 0.0
    weighted_mean_obs = 0.0
    total_n = 0

    # First, compute the weighted mean of observations across bins
    for key, w_raw in _iter_weight_bins(weights_df):
        w = w_raw / total_weight
        if key not in bin_counts or bin_counts[key] == 0:
            print(f"Warning: Bin {key} has zero observations; cannot compute weighted metrics reliably. Returning None for all metrics.")
            return {}
        n_bin = bin_counts[key]
        total_n += n_bin
        mean_obs_b = bin_obs_sums[key] / n_bin
        weighted_mean_obs += w * mean_obs_b

    # Second, compute the weighted metrics
    for key, w_raw in _iter_weight_bins(weights_df):
        w = w_raw / total_weight
        n_bin = bin_counts[key]
        ss_res_b = bin_sq_errors[key]
        abs_b = bin_abs_errors[key]
        err_b = bin_err_sums[key]
        sum_obs_b = bin_obs_sums[key]

        weighted_mse += w * (ss_res_b / n_bin)
        weighted_mae += w * (abs_b / n_bin)
        weighted_bias += w * (err_b / n_bin)
        pbias_num += w * (err_b / n_bin)
        pbias_den += w * (sum_obs_b / n_bin)
        ss_res_global += w * (ss_res_b / n_bin)

    if pbias_den == 0:
        raise RuntimeError('Cannot compute weighted %bias: weighted sum of observations is 0')
    
    weighted_pbias = 100.0 * (pbias_num / pbias_den)
    weighted_rmse = weighted_mse ** 0.5
    ss_tot_global = 0.0
    mu = weighted_mean_obs
    for key, w_raw in _iter_weight_bins(weights_df):
        w = w_raw / total_weight
        n_bin = bin_counts[key]
        sum_obs_b = bin_obs_sums[key]
        sum_obs_sq_b = bin_obs_sq_sums[key]
        ss_tot_b = sum_obs_sq_b - 2.0 * mu * sum_obs_b + n_bin * (mu ** 2)
        ss_tot_global += w * (ss_tot_b / n_bin)

    if ss_tot_global == 0:
        raise RuntimeError('Cannot compute weighted NSE: zero variance in weighted observations')
    weighted_nse = 1.0 - (ss_res_global / ss_tot_global)

    weighted_mse_snow = _weighted_mean_over_available_bins(weights_df, bin_snow_sq_errors, bin_snow_counts)
    rmse_snow = None if weighted_mse_snow is None else weighted_mse_snow ** 0.5
    bias_snow = _weighted_mean_over_available_bins(weights_df, bin_snow_err_sums, bin_snow_counts)

    peak_mae_yearly = _weighted_mean_over_available_bins(weights_df, bin_yearly_peak_abs_error_sums, bin_yearly_counts)
    peak_mb_yearly = _weighted_mean_over_available_bins(weights_df, bin_yearly_peak_error_sums, bin_yearly_counts)
    peak_timing_mae_yearly = _weighted_mean_over_available_bins(weights_df, bin_yearly_timing_abs_error_sums, bin_yearly_counts)
    peak_timing_mb_yearly = _weighted_mean_over_available_bins(weights_df, bin_yearly_timing_error_sums, bin_yearly_counts)

    snowpack_duration_mae_yearly = _weighted_mean_over_available_bins(weights_df, bin_yearly_snowpack_abs_error_sums, bin_yearly_counts)
    snowpack_duration_mb_yearly = _weighted_mean_over_available_bins(weights_df, bin_yearly_snowpack_error_sums, bin_yearly_counts)

    # Weighted snow presence confusion matrix (per-bin rates combined with stratum weights)
    weighted_tp = 0.0
    weighted_tn = 0.0
    weighted_fp = 0.0
    weighted_fn = 0.0
    for key, w_raw in _iter_weight_bins(weights_df):
        w = w_raw / total_weight
        n_bin = bin_counts[key]
        weighted_tp += w * (bin_snow_tp[key] / n_bin)
        weighted_tn += w * (bin_snow_tn[key] / n_bin)
        weighted_fp += w * (bin_snow_fp[key] / n_bin)
        weighted_fn += w * (bin_snow_fn[key] / n_bin)

    total_class = weighted_tp + weighted_tn + weighted_fp + weighted_fn
    snow_accuracy = None if total_class == 0 else (weighted_tp + weighted_tn) / total_class
    snow_precision = None if (weighted_tp + weighted_fp) == 0 else weighted_tp / (weighted_tp + weighted_fp)
    snow_recall = None if (weighted_tp + weighted_fn) == 0 else weighted_tp / (weighted_tp + weighted_fn)

    print("  Computed weighted metrics")

    return {
        'n_full': int(total_n),
        'mse_full': float(weighted_mse),
        'rmse_full': float(weighted_rmse),
        'mae_full': float(weighted_mae),
        'bias_full': float(weighted_bias),
        'pbias_full': float(weighted_pbias),
        'nse_full': float(weighted_nse),
        'n_snow': int(sum(bin_snow_counts.values())),
        'rmse_snow': None if rmse_snow is None else float(rmse_snow),
        'bias_snow': None if bias_snow is None else float(bias_snow),
        'n_yearly': int(sum(bin_yearly_counts.values())),
        'peak_mae_yearly': None if peak_mae_yearly is None else float(peak_mae_yearly),
        'peak_mb_yearly': None if peak_mb_yearly is None else float(peak_mb_yearly),
        'peak_timing_mae_yearly': None if peak_timing_mae_yearly is None else float(peak_timing_mae_yearly),
        'peak_timing_mb_yearly': None if peak_timing_mb_yearly is None else float(peak_timing_mb_yearly),
        'snowpack_duration_mae_yearly': None if snowpack_duration_mae_yearly is None else float(snowpack_duration_mae_yearly),
        'snowpack_duration_mb_yearly': None if snowpack_duration_mb_yearly is None else float(snowpack_duration_mb_yearly),
        'snow_accuracy': None if snow_accuracy is None else float(snow_accuracy),
        'snow_precision': None if snow_precision is None else float(snow_precision),
        'snow_recall': None if snow_recall is None else float(snow_recall),
    }


##############################################################################
# MAIN SCRIPT
##############################################################################

def _main():
    if experiments_dir:
        results_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    else:
        results_dirs = [local_paths.BASE_PATH / 'results']

    rows = []
    if not results_dirs:
        raise RuntimeError("No results directories specified for metric computation")
    
    if experiments_dir:
        print(f"Computing metrics for {len(results_dirs)} directories in: {experiments_dir}")

    print(f"Using threshold={threshold} for snapping predicted SWE values to zero")
    
    for results_dir in results_dirs:
        for split_type in ('eval', 'train'):
            is_era5 = results_dir.name == 'era5_land'
            if split_type == 'train' and is_era5:
                print(f"Skipping train split for ERA5 Land")
                continue
            print(f"Processing folder: {results_dir.name}, split: {split_type}")

            metrics = compute_metrics(results_dir=results_dir, split_type=split_type, use_era5_swe=is_era5, threshold=threshold, apply_weights=apply_weights)

            # Append a row for this experiment and split
            row = {
                'experiment': results_dir.name,
                'split': split_type,
                'n_full': metrics.get('n_full'),
                'mse_full': metrics.get('mse_full'),
                'rmse_full': metrics.get('rmse_full'),
                'mae_full': metrics.get('mae_full'),
                'bias_full': metrics.get('bias_full'),
                'pbias_full': metrics.get('pbias_full'),
                'nse_full': metrics.get('nse_full'),
                'sp_accuracy_full': metrics.get('snow_accuracy'),
                'sp_precision_full': metrics.get('snow_precision'),
                'sp_recall_full': metrics.get('snow_recall'),
                'n_snow': metrics.get('n_snow'),
                'rmse_snow': metrics.get('rmse_snow'),
                'bias_snow': metrics.get('bias_snow'),
                'n_yearly': metrics.get('n_yearly'),
                'peak_mae_yearly': metrics.get('peak_mae_yearly'),
                'peak_mb_yearly': metrics.get('peak_mb_yearly'),
                'peak_timing_mae_yearly': metrics.get('peak_timing_mae_yearly'),
                'peak_timing_mb_yearly': metrics.get('peak_timing_mb_yearly'),
                'snowpack_duration_mae_yearly': metrics.get('snowpack_duration_mae_yearly'),
                'snowpack_duration_mb_yearly': metrics.get('snowpack_duration_mb_yearly'),
            }
            rows.append(row)

    # Save the metrics to a CSV file
    weighted_str = 'weighted' if apply_weights else 'unweighted'
    out_dir = experiments_dir if experiments_dir else results_dirs[0]
    out_path = out_dir / f'{prefix}evaluation_metrics_thr_{threshold}_{weighted_str}.csv'
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"Saved evaluation metrics to: {out_path}")


if __name__ == '__main__':
    _main()