#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=single_run
#SBATCH --output=out/%x_%j.log
# Add a --partition line with a GPU partition of your cluster.
# One run of main.py. Usage, from the repository root: mkdir -p out && sbatch jobs/single_modelling_run.sh [main.py arguments]

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON="${PYTHON:-python}"

PYTHONUNBUFFERED=1 "$PYTHON" src/2_modelling/main.py "$@"
