#!/usr/bin/env bash
# Hyperparameter search on the 'tune' station set, as a SLURM array.
# Usage, from the repository root: bash jobs/run_hp_parallel.sh <run_name> [runs_per_job]
set -euo pipefail

RUN_NAME="${1:?usage: run_hp_parallel.sh <run_name> [runs_per_job]}"

# How many HP combinations each SLURM array task should execute
RUNS_PER_JOB="${2:-12}"

# Where to store the generated list of runs
RUN_LIST="temp/hp_runs_${RUN_NAME}.tsv"
mkdir -p temp out
: > "$RUN_LIST"

# Define "fixed" hyperparameters (same for all runs)
EPOCHS=200
TIMEOUT=119 # in hours, just below the 120 h time limit of the job

# Hyperparameter grids
LRS=(0.00005 0.0001 0.0005 0.001 0.005)
HSS=("32" "64" "128" "[32,2]") # hidden_size or [hidden_size, num_layers]
OPTS=(Adam RMSprop)
BATCHES=(32 64 128)
DROPOUTS=(0.0 0.2)

# Build full run list once: results_subdir<TAB>hparams_json
for lr in "${LRS[@]}"; do
  for hs in "${HSS[@]}"; do
    for opt in "${OPTS[@]}"; do
      for bs in "${BATCHES[@]}"; do
        for dr in "${DROPOUTS[@]}"; do
          hparams_json=$(printf '{"learning_rate": %s, "optimizer_name": "%s", "n_epochs": %s, "batch_size": %s, "hidden_size": %s, "dropout": %s, "max_training_hours": %s}' "$lr" "$opt" "$EPOCHS" "$bs" "$hs" "$dr" "$TIMEOUT")
          run_hs=${hs//[/}; run_hs=${run_hs//]/}; run_hs=${run_hs//,/x}
          run_id="lr${lr//./}_opt${opt}_bs${bs}_hs${run_hs}_d${dr//./}"
          printf '%s\t%s\n' "hp_tuning_${RUN_NAME}/${run_id}" "$hparams_json" >> "$RUN_LIST"
        done
      done
    done
  done
done

total_runs=$(wc -l < "$RUN_LIST")
n_jobs=$(( (total_runs + RUNS_PER_JOB - 1) / RUNS_PER_JOB ))

echo "Total runs: $total_runs"
echo "Runs per job: $RUNS_PER_JOB"
echo "Submitting $n_jobs array jobs."

export RUN_LIST RUNS_PER_JOB

sbatch --array=0-$((n_jobs-1)) --job-name="hp_${RUN_NAME}" --export=ALL <<'SLURM'
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=120:00:00
#SBATCH --output=out/%x_%A_%a.log
# Add a --partition line for your cluster.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

total_runs=$(wc -l < "$RUN_LIST")
start=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_JOB + 1 ))
end=$(( start + RUNS_PER_JOB - 1 ))
(( start > total_runs )) && exit 0
(( end > total_runs )) && end=$total_runs

echo "Task $SLURM_ARRAY_TASK_ID handles lines $start..$end"

# Parallel inside each job: redirect each run's stdout/stderr to separate files
pids=()
while IFS=$'\t' read -r subdir hparams_json; do
  outf="out/${subdir//\//_}.log"
  echo "Starting $subdir"
  echo "HPARAMS: $hparams_json" >> "$outf"
  PYTHONUNBUFFERED=1 "$PYTHON" src/2_modelling/main.py --hparams "$hparams_json" --station-set tune \
    --results-subdir "$subdir" >>"$outf" 2>&1 &
  pids+=("$!")
done < <(sed -n "${start},${end}p" "$RUN_LIST")

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if (( fail )); then
  echo "Task $SLURM_ARRAY_TASK_ID had one or more failed runs."
  exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID finished."
SLURM
