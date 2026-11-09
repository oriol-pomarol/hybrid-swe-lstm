#!/usr/bin/env bash
# Final training runs of one model variant, seeds 1..N in parallel in one SLURM job.
# Usage, from the repository root: bash jobs/run_seeds_parallel.sh <run_name> [num_seeds]
set -euo pipefail

RUN_NAME="${1:?usage: run_seeds_parallel.sh <run_name> [num_seeds]}"
NUM_SEEDS="${2:-5}"

# Hyperparameters of the final runs
HPARAMS_BASE='{"learning_rate": 0.001, "optimizer_name": "RMSprop", "n_epochs": 200, "batch_size": 32, "hidden_size": 128, "dropout": 0.0}'

RUN_LIST="temp/seed_runs_${RUN_NAME}.tsv"
mkdir -p temp out
: > "$RUN_LIST"

# Generate run entries: results_subdir<TAB>seed
for (( s=1; s<=NUM_SEEDS; s++ )); do
  printf '%s\t%s\n' "train_seeded_runs/${RUN_NAME}_seed${s}" "$s" >> "$RUN_LIST"
done

total_runs=$(wc -l < "$RUN_LIST")
echo "Submitting one SLURM job that runs $total_runs seeds of '$RUN_NAME' in parallel."

export HPARAMS_BASE RUN_LIST

sbatch --job-name="final_${RUN_NAME}" --export=ALL <<'SLURM'
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=120:00:00
#SBATCH --output=out/%x_%j.log
# Add a --partition line for your cluster.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running seed experiments listed in $RUN_LIST"

pids=()
while IFS=$'\t' read -r subdir seed; do
  outf="out/$(basename "$subdir").log"
  echo "Starting $subdir (seed=$seed)" | tee -a "$outf"
  PYTHONUNBUFFERED=1 "$PYTHON" src/2_modelling/main.py --hparams "$HPARAMS_BASE" --train-seed "$seed" \
    --station-set full --results-subdir "$subdir" >>"$outf" 2>&1 &
  pids+=("$!")
done < "$RUN_LIST"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if (( fail )); then
  echo "One or more seed runs failed." >&2
  exit 1
fi

echo "All seed runs finished successfully."
SLURM

echo "Submitted SLURM job. Check out/ for the job log and the per-run logs."
