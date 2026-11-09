#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=predictor_comparison
#SBATCH --output=out/%x_%j.log
# Add a --partition line with a GPU partition of your cluster.
# Preprocessing and model runs for the configs listed in experiment_configs/run_order.txt, one after the other.
# Usage, from the repository root: mkdir -p out && sbatch jobs/run_predictors_sequential.sh [station_set]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

PYTHON="${PYTHON:-python}"
STATION_SET="${1:-subset}"
CFG_DIR="experiment_configs"
RUN_LIST="$CFG_DIR/run_order.txt"
CONFIG="src/common/config.py"

echo "WARNING: this job overwrites $CONFIG and the training sequences in data/temp." >&2

while IFS= read -r cfg || [[ -n "$cfg" ]]; do
  cfg="${cfg//$'\r'/}"
  [[ -z "$cfg" || "$cfg" =~ ^# ]] && continue

  run_id="${cfg%.py}_$(date +%Y%m%d_%H%M%S)"
  echo "=== RUN $run_id ==="

  cp "$CFG_DIR/$cfg" "$CONFIG"
  "$PYTHON" src/1_preprocessing/save_training_sequences.py
  "$PYTHON" src/2_modelling/main.py --station-set "$STATION_SET" --results-subdir "predictor_comparison/$run_id"

done < "$RUN_LIST"

echo "All experiment runs finished."
