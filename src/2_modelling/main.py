"""Train one LSTM and predict SWE at its training and evaluation stations.

Inputs: the sequences in BASE_PATH/data/temp (save_training_sequences.py) and the settings in
src/common/config.py (HYBRID_MODE, predictors).

Outputs (BASE_PATH/results[/<results-subdir>]): run_config.json, station_splits.csv,
lstm_epoch_losses.csv, models/ and predictions/NH_pSWE_{train,eval}.csv.

Usage: python src/2_modelling/main.py [--hparams JSON] [--station-set {full,tune,subset}]
                                      [--train-seed N] [--results-subdir NAME]
"""
from create_data_loaders import create_data_loaders
from predict_from_loader import predict_from_loader
from train_lstm import train_lstm, resolve_hparams
import gc
import time
import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import local_paths
import shared

# Define helper functions for timing steps and the whole script
def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def run_step(name, func, *args, **kwargs):
    print(f"START {name}: {_now()}")
    s = time.time()
    out = func(*args, **kwargs)
    e = time.time()
    print(f"END   {name}: {_now()} (duration {e-s:.2f}s)")
    return out

def write_run_config(run_dir, hparams, args):
    """Write the settings of this run to run_dir/run_config.json."""
    run_config = {
        'hparams': hparams,
        'train_seed': args.train_seed,
        'station_set': args.station_set,
        'HYBRID_MODE': config.HYBRID_MODE,
        'PREDICT_dSWE': config.PREDICT_dSWE,
        'DYN_VARS': config.DYN_VARS,
        'STATIC_VARS': config.STATIC_VARS,
        'TEST_MODE': config.TEST_MODE,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / f'{shared.TEST_PREFIX}run_config.json'
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved run configuration to {config_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-subdir', help='optional run id', default=None)
    parser.add_argument('--hparams', help='JSON string with hyperparameters', default=None)
    parser.add_argument('--train-seed', help='random seed for reproducibility', type=int, default=10)
    parser.add_argument('--station-set', choices=shared.STATION_SETS, default='full',
                        help="stations to train and evaluate on: 'full' (final runs: 70%% train, held-out 20%% eval), "
                             "'tune' (hyperparameter tuning: 50%% train, another 20%% eval) or "
                             "'subset' (~100 training stations, the 'tune' eval stations)")
    args = parser.parse_args()

    # Folder for this run's outputs: results/, or results/<subdir> if a subdir is provided
    run_dir = local_paths.BASE_PATH / 'results'
    if args.results_subdir:
        run_dir = run_dir / args.results_subdir

    # Load the hyperparameters, apply the defaults and save the run configuration
    raw = json.loads(args.hparams) if args.hparams else {}
    hparams = resolve_hparams(raw, station_set=args.station_set)
    write_run_config(run_dir, hparams, args)

    # Script-level timing
    print(f"SCRIPT START: {_now()}")
    _script_s = time.time()

    # Create the train and validation data loaders
    data_loaders = run_step("Creating data loaders", create_data_loaders,
                            run_dir=run_dir, station_set=args.station_set, batch_size=hparams['batch_size'])

    # Train and save the LSTM model
    model = run_step("Training LSTM model", train_lstm,
             data_loaders, run_dir=run_dir, save_model=True, hparams=hparams, train_seed=args.train_seed)

    # The validation loader is not needed after training
    data_loaders.pop('val', None)
    gc.collect()

    # Predict on training stations
    run_step("Predicting training stations", predict_from_loader,
             data_loaders['train_pred'], run_dir=run_dir, model=model, split='train')

    # Predict on evaluation stations
    run_step("Predicting evaluation stations", predict_from_loader,
             data_loaders['test'], run_dir=run_dir, model=model, split='eval')

    _script_e = time.time()
    print(f"SCRIPT END: {_now()} (duration {_script_e - _script_s:.2f}s)")


if __name__ == '__main__':
    main()
