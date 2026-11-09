import torch
import torch.nn as nn
from tqdm import tqdm
import math
import time
import random
import numpy as np
from data_scaler import DataScaler
from lstm_definition import LSTM
from sequence_runner import SequenceRunner
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared

def resolve_hparams(hparams=None, station_set='full'):
    """
    Return the full hyperparameter dict: `hparams` plus the defaults of `shared.HPARAM_DEFAULTS`
    for the missing keys. Test-mode runs and the 'subset' station set use the smaller
    `shared.SMALL_RUN_HPARAM_DEFAULTS`. Unknown keys and optimizers raise a ValueError.
    """

    hp = {} if hparams is None else dict(hparams)
    valid_keys = list(shared.HPARAM_DEFAULTS) + ['patience']
    unknown = sorted(set(hp) - set(valid_keys))
    if unknown:
        raise ValueError(f"Unknown hyperparameters {unknown}; valid keys are {valid_keys}")

    defaults = dict(shared.HPARAM_DEFAULTS)
    if config.TEST_MODE or station_set == 'subset':
        defaults.update(shared.SMALL_RUN_HPARAM_DEFAULTS)
    merged = {**defaults, **hp}

    n_epochs = int(merged['n_epochs'])
    resolved = {
        'batch_size': int(merged['batch_size']),
        'hidden_size': merged['hidden_size'],
        'learning_rate': float(merged['learning_rate']),
        'optimizer_name': merged['optimizer_name'],
        'n_epochs': n_epochs,
        'chunk_size': int(merged['chunk_size']),
        'patience': int(merged.get('patience', max(1, int(n_epochs*0.1)))),
        'min_delta': float(merged['min_delta']),
        'dropout': float(merged['dropout']),
        'max_training_hours': float(merged['max_training_hours']),
    }
    if resolved['optimizer_name'] not in shared.OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{resolved['optimizer_name']}'; choose one of {list(shared.OPTIMIZERS)}")
    return resolved


def train_lstm(data_loaders, run_dir, save_model=True, hparams=None, train_seed=None):
    """
    Train the LSTM model using the provided training data loader.

    Args:
        data_loaders (dict): Dictionary containing at least the `'train'` and
            `'val'` DataLoaders returned by `create_data_loaders`.
        run_dir (Path): Folder of this run; the model and losses are written here.
        save_model (bool): Save the best model to `run_dir/models/lstm_model.pt`.
        hparams (dict): Hyperparameters; missing ones get the defaults of `resolve_hparams`.
        train_seed (int): Seed of the random number generators (None: unseeded).

    Returns:
        The trained model, with the weights of the epoch with the lowest validation loss.
    """

    prefix = shared.TEST_PREFIX

    # Set the HP values
    hp = resolve_hparams(hparams)
    hidden_size = hp['hidden_size']
    learning_rate = hp['learning_rate']
    optimizer_name = hp['optimizer_name']
    n_epochs = hp['n_epochs']
    chunk_size = hp['chunk_size']
    patience = hp['patience']
    min_delta = hp['min_delta']
    dropout = hp['dropout']
    max_training_hours = hp['max_training_hours']

    print(f"Set hyperparameters: hidden_size={hidden_size}, learning_rate={learning_rate}, optimizer={optimizer_name}, n_epochs={n_epochs}, chunk_size={chunk_size}, patience={patience}, min_delta={min_delta}, dropout={dropout}, max_training_hours={max_training_hours}")

    if train_seed is not None:
        print(f"Setting training seed: {train_seed}")
        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_seed)
        # Favor deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Check that there is a gpu available
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu}")
        device = torch.device("cuda")
    else:
        print("No GPU available")
        device = torch.device("cpu")

    # Extract the loaders we need
    train_loader, val_loader = data_loaders['train'], data_loaders['val']

    # Initialize the model and move it to the device
    input_size = len(config.DYN_VARS) + len(config.STATIC_VARS) + (1 * config.HYBRID_MODE)
    model = LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
    model.to(device)

    # Set the optimizer and loss function
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum')

    # Initialize a dict to hold epoch losses
    epoch_losses = {'train': [], 'val': []}

    # Initialize simple meters for train and validation
    train_meter = AverageMeter()
    val_meter = AverageMeter()

    # Load the transformers and retrieve the transformed zero value
    data_scaler = DataScaler(run_dir)
    data_scaler.load_transformers()
    trans_zero = data_scaler.get_transformed_zero()

    # Initialize early-stopping variables
    best_val = float('inf')
    no_improve = 0
    stop_early = False
    timeout_reached = False
    best_state_dict = None
    training_start_time = time.time()
    max_training_seconds = max_training_hours * 3600.0

    for epoch in range(n_epochs):

        # Reset meters at the start of the epoch
        train_meter.reset()
        val_meter.reset()

        # Set model to training mode
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch"):

            elapsed_seconds = time.time() - training_start_time
            if elapsed_seconds > max_training_seconds:
                print(f"Training timeout reached after {elapsed_seconds/3600.0:.2f}h (limit={max_training_hours:.2f}h). Stopping training.")
                timeout_reached = True
                break

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Set up runner and run warmup
            train_runner = SequenceRunner(batch, model, device, trans_zero)
            train_runner.run_warmup(shared.WARMUP_DAYS)

            # Get targets for the prediction period
            targets = batch['target'].squeeze(-1).to(device)
            prediction_steps = batch['length'].item() - shared.WARMUP_DAYS

            # Truncated BPTT over chunks
            for step_cursor in range(0, prediction_steps, chunk_size):
                steps_this_chunk = min(chunk_size, prediction_steps - step_cursor)
                preds_chunk = train_runner.run_predictions(steps_this_chunk)
                targets_chunk = targets[:, step_cursor: step_cursor + steps_this_chunk]

                try:
                    chunk_loss, valid_count = calculate_loss(preds_chunk, targets_chunk, criterion)
                except ValueError as e:
                    print(f"Warning: skipping loss computation for station(s) {batch['station_id']}:\n{e}")
                    train_runner.detach_state()
                    continue

                train_meter.update(chunk_loss.item(), valid_count)

                mean_loss = chunk_loss / valid_count
                mean_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Detach carried state to truncate backprop
                train_runner.detach_state()

                # Free the chunk tensors before the next chunk
                del preds_chunk, targets_chunk, mean_loss, chunk_loss

        if timeout_reached:
            print("Training stopped due to timeout.")
            break

        # Set model to evaluation mode and disable grad for running validation
        model.eval()
        with torch.no_grad():

            # Compute the validation loss at the end of the epoch
            for batch in val_loader:

                # Generate predictions for the batch
                prediction_steps = batch['length'].item() - shared.WARMUP_DAYS
                val_runner = SequenceRunner(batch, model, device, trans_zero)
                val_runner.run_warmup(shared.WARMUP_DAYS)
                val_predictions = val_runner.run_predictions(prediction_steps)

                # Get targets for the prediction period
                val_targets = batch['target'].squeeze(-1).to(device)

                # Calculate the loss between predictions and targets
                try:
                    val_sum_loss, val_valid_count = calculate_loss(val_predictions, val_targets, criterion)
                except ValueError as e:
                    print(f"Warning: skipping validation loss computation for station(s) {batch['station_id']}:\n{e}")
                    continue

                # Update validation meter with the sum loss and number of valid samples
                val_meter.update(val_sum_loss, val_valid_count)

            # Calculate and print epoch losses from meters
            train_mse = train_meter.compute()
            val_mse = val_meter.compute()
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {train_mse:.4f}, Validation Loss: {val_mse:.4f}")
            epoch_losses['train'].append(train_mse)
            epoch_losses['val'].append(val_mse)

            # Minimal early-stopping: store best weights in memory and stop when no improvement
            if not math.isnan(val_mse):
                if val_mse + min_delta < best_val:
                    best_val = val_mse
                    no_improve = 0
                    # store a CPU clone of the state dict to avoid GPU I/O and mutation
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(f"Early stopping: no improvement for {patience} epochs")
                    stop_early = True

            if stop_early:
                break

    # If we have best weights stored in memory, restore them and optionally save once
    if best_state_dict is not None:
        # load onto the current device
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        print(f"Restored best model from memory (val={best_val:.4f})")
        if save_model:
            model_dir = run_dir / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f'{prefix}lstm_model.pt'
            torch.save(best_state_dict, model_path)
            print(f"Saved model to {model_path}")
    else:
        print("Warning: No model state available in memory; training completed without saving.")

    # Save epoch losses to a csv file
    losses_path = run_dir / f'{prefix}lstm_epoch_losses.csv'
    with open(losses_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        for epoch in range(len(epoch_losses['train'])):
            f.write(f"{epoch+1},{epoch_losses['train'][epoch]},{epoch_losses['val'][epoch]}\n")
    print(f"Saved epoch losses to {losses_path}")

    return model


def calculate_loss(predictions, targets, criterion):
    """Calculate the loss between predictions and targets, ignoring NaNs in targets."""

    # Check that predictions and targets have the same shape
    if predictions.shape != targets.shape:
        raise ValueError(
            "Predictions shape {} does not match targets shape {}"
            .format(predictions.shape, targets.shape)
        )

    # Filter out NaN targets
    prediction_mask = ~torch.isnan(targets)
    masked_preds = predictions[prediction_mask]
    masked_targets = targets[prediction_mask]

    # Check for NaNs in predictions
    if torch.isnan(masked_preds).any():
        raise ValueError("Predictions contain NaN values")

    # Get number of valid samples
    n_samples = masked_targets.numel()

    # Calculate loss only if there are valid target values
    if n_samples > 0:
        loss = criterion(masked_preds, masked_targets)
    else:
        # Raise an error if there are no valid samples
        raise ValueError("No valid target samples available for loss calculation.")

    return loss, n_samples

class AverageMeter:
    """Simple helper to accumulate sums and counts and compute an average.

    Stores a running sum (e.g., sum of squared errors) and a count of valid
    samples. `compute()` returns sum/count or NaN when count == 0.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value_sum, n=1):
        self.sum += float(value_sum)
        self.count += int(n)

    def compute(self):
        return self.sum / self.count if self.count > 0 else float('nan')