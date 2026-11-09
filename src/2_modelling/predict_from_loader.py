import json
import torch
import pandas as pd
import numpy as np
from data_scaler import DataScaler
from lstm_definition import LSTM
from sequence_runner import SequenceRunner
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared

def predict_from_loader(data_loader, run_dir, model=None, split=None, hidden_size=None, return_predictions=False):
    """
    Predict SWE for every station in `data_loader`.

    If `split` is given (e.g. 'train' or 'eval'), the predictions of all stations are written
    to one long-format file `run_dir/predictions/NH_pSWE_<split>.csv` with the columns
    `Date`, `pred_swe` and `station_id`. With `return_predictions=True`, a dict
    {station_id: DataFrame} is returned as well; otherwise None is returned.
    """

    # Define paths
    prefix = shared.TEST_PREFIX
    model_path = run_dir / 'models' / f'{prefix}lstm_model.pt'
    output_path = run_dir / 'predictions'
    output_path.mkdir(parents=True, exist_ok=True)

    # Check that there is a gpu available
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu}")
        device = torch.device("cuda")
    else:
        print("No GPU available")
        device = torch.device("cpu")

    # Load the model if not provided
    if model is None:

        # Take the hidden size from the run configuration if there is one
        run_config_path = run_dir / f'{prefix}run_config.json'
        if run_config_path.exists():
            with open(run_config_path) as f:
                hidden_size = json.load(f)['hparams']['hidden_size']
        if hidden_size is None:
            raise ValueError(f"No {run_config_path.name} in {run_dir}: a model or the hidden size of the saved model must be provided.")

        # If the path exists, load the model
        if model_path.exists():
            print("Loading trained LSTM model...")

            # Load the model
            input_size = len(config.DYN_VARS) + len(config.STATIC_VARS) + (1 * config.HYBRID_MODE)
            model = LSTM(input_size=input_size, hidden_size=hidden_size)
            model.to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

        # If the path does not exist, raise an error
        else:
            raise FileNotFoundError(f"No model provided and no saved model found at {model_path}")

    # Load the transformers
    data_scaler = DataScaler(run_dir)
    data_scaler.load_transformers()

    # Retrieve the target transformer
    target_ct = data_scaler.get_transformers()['target']
    target_transformer = None
    for _, est, cols in target_ct.transformers_:
        if cols:
            target_transformer = est
            break
    if target_transformer is None:
        raise ValueError("Target transformer not found in DataScaler.")

    # Get the transformed zero value for SWE
    trans_zero = data_scaler.get_transformed_zero()

    # Keep the predictions in memory only when requested
    station_predictions = {} if return_predictions else None

    # If saving predictions, stream the rows of each station to one CSV per split
    output_file = None
    wrote_header = False
    if split:
        output_file = output_path / f'{prefix}NH_pSWE_{split}.csv'
        if output_file.exists():
            output_file.unlink()

    # Evaluate the model on the loader data
    model.eval()
    with torch.no_grad():
        for batch in data_loader:

            # Generate predictions for the batch
            prediction_steps = batch['length'].item() - shared.WARMUP_DAYS
            runner = SequenceRunner(batch, model, device, trans_zero=trans_zero)
            runner.run_warmup(shared.WARMUP_DAYS)
            predictions = runner.run_predictions(prediction_steps, verbose=True)

            # For each station in the batch, store predictions and start date
            for i, station_id in enumerate(batch['station_id']):
                seq_len = batch['seq_lengths'][i]
                station_preds = predictions[i, :seq_len].cpu().numpy().squeeze()

                # Inverse transform the predictions and snap small values to zero
                inv_trans_preds = target_transformer.inverse_transform(pd.DataFrame(station_preds, columns=['swe']))
                zero_snapped_preds = np.where(inv_trans_preds < 0.1, 0, inv_trans_preds).flatten()

                # Get the date range for the predictions
                dates_pred = pd.date_range(
                    start=batch['start_date'][i],
                    periods=len(station_preds),
                    freq='D'
                ).normalize()

                # Join dates and predictions into a long-format DataFrame
                predictions_df = pd.DataFrame(
                    {'Date': dates_pred, 'pred_swe': zero_snapped_preds}
                )
                predictions_df['station_id'] = station_id
                if return_predictions:
                    station_predictions[station_id] = predictions_df
                if output_file is not None:
                    predictions_df.to_csv(output_file, mode='a', header=not wrote_header, index=False)
                    wrote_header = True

    # Report the saved output path
    if output_file is not None:
        print(f"Saved predictions to {output_file}")

    return station_predictions
