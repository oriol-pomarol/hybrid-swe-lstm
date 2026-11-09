#!/usr/bin/env bash
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH --job-name=snow_analysis
#SBATCH --output=out/%x_%A_%a.log
# Add a --partition line for your cluster.
# The gridded analyses for both sources. Usage, from the repository root: mkdir -p out && sbatch jobs/run_dataset_analysis_array.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

case "$SLURM_ARRAY_TASK_ID" in
  0) script="compute_longterm_trends.py"; source="era5_land"; args=(--variables peak_swe css_length) ;;
  1) script="compute_doy_swe.py"; source="era5_land"; args=() ;;
  2) script="compute_metrics_norswe.py"; source="era5_land"; args=() ;;
  3) script="compute_avg_peak_swe_map.py"; source="era5_land"; args=() ;;
  4) script="compute_longterm_trends.py"; source="hybrid"; args=(--variables peak_swe css_length) ;;
  5) script="compute_doy_swe.py"; source="hybrid"; args=() ;;
  6) script="compute_metrics_norswe.py"; source="hybrid"; args=() ;;
  7) script="compute_avg_peak_swe_map.py"; source="hybrid"; args=() ;;
  *) echo "Unknown SLURM array task ID: $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "Running $script for source=$source"
PYTHONUNBUFFERED=1 "$PYTHON" "src/3_analysis/$script" --source "$source" "${args[@]}"
