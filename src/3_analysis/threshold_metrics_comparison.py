"""Snow-presence and error metrics of the runs in results/final_model_comparison over a range of thresholds.

A check of the threshold used in compute_metrics.py; not in the paper. Always weighted by stratum;
snow presence is SWE > threshold for both observations and predictions.

Inputs: as compute_metrics.py.
Output: results/final_model_comparison/threshold_metrics_comparison_<split>.csv

Usage: python src/3_analysis/threshold_metrics_comparison.py [--split {eval,train}]
           [--thresholds 0,1,5 | --start 0 --end 5 --step 0.5]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared
import local_paths

max_workers = 12

# Folder whose run folders are scored
experiments_dir = local_paths.BASE_PATH / "results" / "final_model_comparison"
targets_dir = local_paths.BASE_PATH / "data" / "temp"


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


def _build_thresholds(start, end, step):
    if step <= 0:
        raise ValueError("step must be > 0")
    n_steps = int(round((end - start) / step))
    thresholds = [start + i * step for i in range(n_steps + 1)]
    return [round(t, 6) for t in thresholds]


def _compute_station_snow_presence_metrics(task):
    (
        station_id,
        station_prediction,
        swe_class,
        elev,
        targets_dir,
        thresholds,
        use_era5_swe,
    ) = task

    elev_band = _get_elev_band(elev)
    if elev_band is None:
        print(f"Skipping station {station_id} due to invalid elevation band")
        return None

    filename = f"{shared.TEST_PREFIX}dynamic_predictors_ID_{station_id}.csv"
    val_target = pd.read_csv(targets_dir / filename, index_col=0, parse_dates=True)[["swe"]]

    if use_era5_swe:
        station_prediction = pd.read_csv(targets_dir / filename, index_col=0, parse_dates=True)[["era5_swe"]]
        station_prediction = station_prediction.rename(columns={"era5_swe": "pred_swe"})

    df_joined = val_target.join(station_prediction, how="inner")
    assert not df_joined.empty, f"No overlapping dates for station {station_id}."
    df_joined = df_joined.dropna()

    obs = df_joined["swe"]
    obs_sum = float(obs.sum())
    obs_sq_sum = float((obs ** 2).sum())

    result = {
        "key": (swe_class, elev_band),
        "count": int(len(df_joined)),
        "obs_sum": obs_sum,
        "obs_sq_sum": obs_sq_sum,
        "threshold_counts": {},
        "threshold_errors": {},
    }

    for threshold in thresholds:
        obs_snow = df_joined["swe"] > threshold
        pred_snow = df_joined["pred_swe"] > threshold
        result["threshold_counts"][threshold] = {
            "tp": int((pred_snow & obs_snow).sum()),
            "tn": int((~pred_snow & ~obs_snow).sum()),
            "fp": int((pred_snow & ~obs_snow).sum()),
            "fn": int((~pred_snow & obs_snow).sum()),
        }

        snapped_pred = df_joined["pred_swe"].where(df_joined["pred_swe"] >= threshold, 0.0)
        errors = snapped_pred - obs
        result["threshold_errors"][threshold] = {
            "sq": float((errors ** 2).sum()),
            "abs": float(errors.abs().sum()),
            "err": float(errors.sum()),
        }

    return result


##############################################################################
# CORE FUNCTIONS
##############################################################################

def compute_snow_presence_threshold_metrics(
    results_dir=local_paths.BASE_PATH / 'results',
    split_type="eval",
    thresholds=None,
    use_era5_swe=False,
):
    if thresholds is None:
        thresholds = _build_thresholds(0.0, 5.0, 0.5)

    weights_path = local_paths.BASE_PATH / "data" / "CLASS_ELEV_WEIGHTS.csv"
    weights_df = pd.read_csv(weights_path, index_col=0)

    metadata_path = local_paths.BASE_PATH / "data" / "STATION_METADATA.csv"
    metadata_df = pd.read_csv(metadata_path, index_col=0)
    metadata_df = metadata_df[["SnowClass", "ELEV"]]

    bin_counts = {}
    bin_obs_sums = {}
    bin_obs_sq_sums = {}
    bin_tp = {threshold: {} for threshold in thresholds}
    bin_tn = {threshold: {} for threshold in thresholds}
    bin_fp = {threshold: {} for threshold in thresholds}
    bin_fn = {threshold: {} for threshold in thresholds}
    bin_sq_errors = {threshold: {} for threshold in thresholds}
    bin_abs_errors = {threshold: {} for threshold in thresholds}
    bin_err_sums = {threshold: {} for threshold in thresholds}

    stations_split_file = results_dir / f"{shared.TEST_PREFIX}station_splits.csv"
    stations_split_df = pd.read_csv(stations_split_file, index_col=0)
    station_ids = stations_split_df[stations_split_df["split"] == split_type].index.tolist()

    if not use_era5_swe:
        filename = results_dir / "predictions" / f"{shared.TEST_PREFIX}NH_pSWE_{split_type}.csv"
        predictions_split = pd.read_csv(filename, parse_dates=["Date"])
        predictions_split["station_id"] = predictions_split["station_id"].astype(int)

        if (set(station_ids) != set(predictions_split["station_id"])):
            raise ValueError(
                f"Mismatch between station IDs in split file and predictions file for {split_type} set."
            )

        predictions_by_station = {
            int(sid): group[["Date", "pred_swe"]].set_index("Date")
            for sid, group in predictions_split.groupby("station_id")
        }
    station_tasks = [
        (
            int(station_id),
            predictions_by_station[int(station_id)] if not use_era5_swe else None,
            metadata_df.loc[int(station_id), "SnowClass"],
            metadata_df.loc[int(station_id), "ELEV"],
            targets_dir,
            thresholds,
            use_era5_swe,
        )
        for station_id in station_ids
    ]
    print(f"  Prepared {len(station_tasks)} station tasks; using max_workers={max_workers}")

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            station_results = tqdm(
                executor.map(_compute_station_snow_presence_metrics, station_tasks),
                total=len(station_tasks),
                desc="Computing snow presence metrics",
                unit="station",
            )
            station_results = list(station_results)
    else:
        station_results = list(
            tqdm(
                map(_compute_station_snow_presence_metrics, station_tasks),
                total=len(station_tasks),
                desc="Computing snow presence metrics",
                unit="station",
            )
        )

    for station_result in station_results:
        if station_result is None:
            continue
        key = station_result["key"]
        bin_counts[key] = bin_counts.get(key, 0) + station_result["count"]
        bin_obs_sums[key] = bin_obs_sums.get(key, 0.0) + station_result["obs_sum"]
        bin_obs_sq_sums[key] = bin_obs_sq_sums.get(key, 0.0) + station_result["obs_sq_sum"]

        for threshold, counts in station_result["threshold_counts"].items():
            bin_tp[threshold][key] = bin_tp[threshold].get(key, 0) + counts["tp"]
            bin_tn[threshold][key] = bin_tn[threshold].get(key, 0) + counts["tn"]
            bin_fp[threshold][key] = bin_fp[threshold].get(key, 0) + counts["fp"]
            bin_fn[threshold][key] = bin_fn[threshold].get(key, 0) + counts["fn"]

        for threshold, errors in station_result["threshold_errors"].items():
            bin_sq_errors[threshold][key] = bin_sq_errors[threshold].get(key, 0.0) + errors["sq"]
            bin_abs_errors[threshold][key] = bin_abs_errors[threshold].get(key, 0.0) + errors["abs"]
            bin_err_sums[threshold][key] = bin_err_sums[threshold].get(key, 0.0) + errors["err"]

    print(f"  Collected results for {sum(bin_counts.values())} observations across {len(bin_counts)} bins")

    weights_arr = weights_df.astype(float).values
    total_weight = float(weights_arr.sum())
    if total_weight <= 0:
        raise RuntimeError("Weights sum to zero or negative")

    for key, _ in _iter_weight_bins(weights_df):
        if key not in bin_counts or bin_counts[key] == 0:
            print(
                f"Warning: Bin {key} has zero observations; cannot compute weighted metrics reliably."
            )
            return {}

    weighted_mean_obs = 0.0
    pbias_den = 0.0
    for key, w_raw in _iter_weight_bins(weights_df):
        w = w_raw / total_weight
        n_bin = bin_counts[key]
        mean_obs_b = bin_obs_sums[key] / n_bin
        weighted_mean_obs += w * mean_obs_b
        pbias_den += w * bin_obs_sums[key] / n_bin

    if pbias_den == 0:
        raise RuntimeError("Cannot compute weighted %bias: weighted sum of observations is 0")

    mu = weighted_mean_obs
    ss_tot_global = 0.0
    for key, w_raw in _iter_weight_bins(weights_df):
        w = w_raw / total_weight
        n_bin = bin_counts[key]
        sum_obs_b = bin_obs_sums[key]
        sum_obs_sq_b = bin_obs_sq_sums[key]
        ss_tot_b = sum_obs_sq_b - 2.0 * mu * sum_obs_b + n_bin * (mu ** 2)
        ss_tot_global += w * ss_tot_b / n_bin

    if ss_tot_global == 0:
        raise RuntimeError("Cannot compute weighted NSE: zero variance in weighted observations")

    rows = []
    for threshold in thresholds:
        acc_num = {}
        acc_den = {}
        rec_num = {}
        rec_den = {}
        fpr_num = {}
        fpr_den = {}
        weighted_mse = 0.0
        weighted_mae = 0.0
        weighted_bias = 0.0
        pbias_num = 0.0
        ss_res_global = 0.0

        for key in bin_counts:
            tp = bin_tp[threshold].get(key, 0)
            tn = bin_tn[threshold].get(key, 0)
            fp = bin_fp[threshold].get(key, 0)
            fn = bin_fn[threshold].get(key, 0)

            total = tp + tn + fp + fn
            acc_num[key] = tp + tn
            acc_den[key] = total
            rec_num[key] = tp
            rec_den[key] = tp + fn
            fpr_num[key] = fp
            fpr_den[key] = fp + tn

            n_bin = bin_counts[key]
            sq = bin_sq_errors[threshold].get(key, 0.0)
            abs_err = bin_abs_errors[threshold].get(key, 0.0)
            err = bin_err_sums[threshold].get(key, 0.0)
            w = float(weights_df.loc[key[0], key[1]]) / total_weight
            weighted_mse += w * (sq / n_bin)
            weighted_mae += w * (abs_err / n_bin)
            weighted_bias += w * (err / n_bin)
            pbias_num += w * err / n_bin
            ss_res_global += w * sq / n_bin

        weighted_accuracy = _weighted_mean_over_available_bins(weights_df, acc_num, acc_den)
        weighted_recall = _weighted_mean_over_available_bins(weights_df, rec_num, rec_den)
        weighted_fpr = _weighted_mean_over_available_bins(weights_df, fpr_num, fpr_den)
        weighted_rmse = weighted_mse ** 0.5
        weighted_pbias = 100.0 * (pbias_num / pbias_den)
        weighted_nse = 1.0 - (ss_res_global / ss_tot_global)

        rows.append(
            {
                "threshold_mm": threshold,
                "sp_accuracy": None if weighted_accuracy is None else float(weighted_accuracy),
                "sp_recall": None if weighted_recall is None else float(weighted_recall),
                "sp_false_positive_rate": None if weighted_fpr is None else float(weighted_fpr),
                "rmse": float(weighted_rmse),
                "mae": float(weighted_mae),
                "bias": float(weighted_bias),
                "pbias": float(weighted_pbias),
                "nse": float(weighted_nse),
            }
        )

    return rows


##############################################################################
# MAIN SCRIPT
##############################################################################

def _parse_args():
    parser = argparse.ArgumentParser(description="Compute weighted snow presence metrics by SWE threshold.")
    parser.add_argument("--split", default="eval", choices=["train", "eval"], help="Split type")
    parser.add_argument("--thresholds", default=None, help="Comma-separated threshold values in mm")
    parser.add_argument("--start", type=float, default=0.0, help="Threshold range start (mm)")
    parser.add_argument("--end", type=float, default=5.0, help="Threshold range end (mm)")
    parser.add_argument("--step", type=float, default=0.5, help="Threshold step (mm)")
    return parser.parse_args()


def _main():
    args = _parse_args()
    if args.thresholds:
        thresholds = [float(v.strip()) for v in args.thresholds.split(",") if v.strip()]
    else:
        thresholds = _build_thresholds(args.start, args.end, args.step)

    results_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]

    rows = []
    if not results_dirs:
        raise RuntimeError("No results directories specified for metric computation")

    for results_dir in results_dirs:
        print(f"Processing folder: {results_dir.name}, split: {args.split}")

        metrics_rows = compute_snow_presence_threshold_metrics(
            results_dir=results_dir,
            split_type=args.split,
            thresholds=thresholds,
            use_era5_swe=True if results_dir.name == "era5_land" else False,
        )

        for metrics in metrics_rows:
            row = {
                "experiment": results_dir.name,
                "split": args.split,
                **metrics,
            }
            rows.append(row)
    
    out_path = experiments_dir / f"{shared.TEST_PREFIX}threshold_metrics_comparison_{args.split}.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"Saved snow presence threshold metrics to: {out_path}")


if __name__ == "__main__":
    _main()
