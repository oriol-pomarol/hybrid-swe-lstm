import gc
import random
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader
from data_scaler import DataScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config
import shared
import local_paths


def create_data_loaders(run_dir, batch_size, station_set='full'):
    """
    Create the data loaders of one run: training, validation (loss) and evaluation stations.

    `station_set` selects the stations; 20% of them are always
    held out, and 10% monitor the loss:
        'full':   train on the remaining 70%, evaluate on the held-out 20% (final runs);
        'tune':   train on 50%, evaluate on another 20% (hyperparameter tuning);
        'subset': train on ~100 stations of those 50%, evaluate on the same 20% as 'tune'.
    The station splits and the fitted transformers are written to `run_dir`.

    Returns:
        dict: A dictionary with keys `'train'`, `'train_pred'`, `'val'` and `'test'` mapping to the
              training (shuffled batches), training (one batch, for prediction), validation
              (loss monitoring) and test DataLoaders.
    """

    if station_set not in shared.STATION_SETS:
        raise ValueError(f"Unknown station set '{station_set}'; choose one of {shared.STATION_SETS}")

    ## 1. Load the static and dynamic data and store as a list of samples
    # Load the static predictors
    temp_dir = local_paths.BASE_PATH / 'data' / 'temp'
    static_predictors = pd.read_csv(temp_dir / f'{shared.TEST_PREFIX}static_predictors.csv', index_col=0)

    # Check that all static columns defined in the config file are present in the static data
    missing_static_cols = [var for var in config.STATIC_VARS if var not in static_predictors.columns]
    if missing_static_cols:
        raise ValueError(f"Missing static columns in loaded data: {missing_static_cols}")

    # Define a function to load one dynamic dataframe
    def load_dynamic_dataframe(station_id):
        dynamic_df = pd.read_csv(temp_dir / f'{shared.TEST_PREFIX}dynamic_predictors_ID_{station_id}.csv',
                                 index_col=0, parse_dates=[0])
        return station_id, dynamic_df

    # Column names of each variable group
    dynamic_cols = [v for v in config.DYN_VARS if v != 'swe']
    col_names = {
        'dynamic': dynamic_cols + ['era5_swe'] if config.HYBRID_MODE else dynamic_cols,
        'static': config.STATIC_VARS,
        'target': ['swe']
    }

    # Load the dynamic dataframes in parallel and build one sample per station as each one arrives
    all_samples = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_dynamic_dataframe, station_id): station_id for station_id in static_predictors.index}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading dynamic dataframes"):
            try:
                station_id, dynamic_df = future.result()
            except Exception as e:
                print(f"Error loading dynamic dataframe for station {futures[future]}: {str(e)}")
                continue

            # Raise an error if there are missing values in the dynamic predictors
            if dynamic_df.drop(columns=['swe']).isnull().any().any():
                raise ValueError(f"Missing values found in dynamic predictors for station {station_id}")

            # Store the sample in the dictionary
            all_samples[station_id] = {
                'dynamic': dynamic_df[col_names['dynamic']],
                'static': static_predictors.loc[[station_id]][col_names['static']],
                'target': dynamic_df.iloc[shared.WARMUP_DAYS:][['swe']],
                'length': len(dynamic_df),
                'station_id': station_id,
                'start_date': dynamic_df['swe'].first_valid_index(),
            }

    ## 2. Perform a train/tune/test split with stratified sampling
    # Load the station metadata and select only the stations present in the static predictors
    metadata_path = local_paths.BASE_PATH / 'data' / 'STATION_METADATA.csv'
    metadata_df = pd.read_csv(metadata_path, index_col=0)
    metadata_df = metadata_df.loc[static_predictors.index]

    # Create elevation bands
    metadata_df['elevation_band'] = pd.cut(metadata_df['ELEV'], bins=config.ELEVATION_BINS, labels=False)

    # Define a function for stratified sampling
    def _stratified_sample(df, cols, split_frac=0.5):
        '''
        Perform stratified sampling on the dataframe based on the specified columns.
        When in test mode, performs random sampling instead.
        Args:
            df (pd.DataFrame): DataFrame containing station metadata.
            cols (list): List of column names to use for stratification.
            split_frac (float): Fraction of samples to include in the smaller split.
        Returns:
            large_split (list): List of df indices in the larger split.
            small_split (list): List of df indices in the smaller split.
        '''

        # Check that the split fraction is valid
        if split_frac < 0.0 or split_frac > 0.5:
            raise ValueError("split_frac must be between 0 and 0.5")

        # In test mode (a handful of stations), perform random sampling
        if config.TEST_MODE:
            group_n = max(1, int(round(split_frac * len(df))))
            small_split = random.sample(df.index.tolist(), group_n)
            large_split = df.index.difference(small_split).tolist()
            return large_split, small_split

        # Perform stratified sampling
        large_split = []
        small_split = []

        # If SnowClass is one of the columns, move any Ocean class to the large split directly
        if 'SnowClass' in cols:
            ocean_stations = df.index[df['SnowClass'] == 'Ocean'].tolist()
            large_split.extend(ocean_stations)
            df = df.drop(index=ocean_stations)

            # Recalculate the split fraction based on remaining stations
            split_frac = split_frac * len(df) / (len(df) + len(ocean_stations))

        for _, group in df.groupby(cols):
            # Determine number of samples for each group
            group_n = max(1, int(round(split_frac * len(group))))
            group_sample = group.sample(n=group_n, random_state=10).index.tolist()
            small_split.extend(group_sample)
            large_split.extend(group.drop(index=group_sample).index.tolist())

        return large_split, small_split
        
    # Set a random seed
    random.seed(10)

    # Select a stratified sample of 20% of the stations for final evaluation
    train_stations, eval_stations = _stratified_sample(
        df=metadata_df,
        cols=['elevation_band', 'SnowClass'],
        split_frac=0.2)

    # Select a stratified sample of 10% of the stations for loss monitoring
    train_stations, loss_monitoring_stations = _stratified_sample(
        df=metadata_df.loc[train_stations],
        cols=['elevation_band', 'SnowClass'],
        split_frac=0.125)

    # Select a stratified sample of 20% of the stations for HP tuning
    tune_train_stations, tune_eval_stations = _stratified_sample(
        df=metadata_df.loc[train_stations],
        cols=['elevation_band', 'SnowClass'],
        split_frac=2/7)
    
    # In the 'subset' station set, take a smaller subset of those for training (~100 stations)
    if station_set == 'subset':
        _, tune_train_stations = _stratified_sample(
            df=metadata_df.loc[tune_train_stations],
            cols=['elevation_band', 'SnowClass'],
            split_frac=0.025)

    # Select the correct train and eval stations for the tune and subset station sets
    if station_set in ('tune', 'subset'):
        eval_stations = tune_eval_stations
        train_stations = tune_train_stations
    
    # Indicate the type of split for each station in a dataframe
    station_splits_df = pd.DataFrame({
        'NHSWEID': metadata_df.index,
        'split': ['train' if stn in train_stations else 'loss' if stn in loss_monitoring_stations else 'eval' if stn in eval_stations else 'unused' for stn in metadata_df.index]
    })
    
    # Save the station splits to a CSV file with station ID as index
    station_splits_df.set_index('NHSWEID', inplace=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    station_splits_df.to_csv(run_dir / f'{shared.TEST_PREFIX}station_splits.csv')

    ## 3. Normalize the data and create the data loaders
    #  Select the train/val/test sample lists
    test_samples = [all_samples[stn] for stn in eval_stations]
    val_samples = [all_samples[stn] for stn in loss_monitoring_stations]
    train_samples = [all_samples[stn] for stn in train_stations]

    # Initialize the data scaler
    data_scaler = DataScaler(run_dir)
    data_scaler.initialize_transformers(col_names=col_names)

    # Fit the transformers on the training samples
    data_scaler.fit_samples(train_samples)

    # Transform all samples
    trans_train_samples = data_scaler.transform_samples(train_samples)
    trans_val_samples = data_scaler.transform_samples(val_samples)
    trans_test_samples = data_scaler.transform_samples(test_samples)

    # Release the large temporary containers before creating the DataLoaders
    del all_samples, train_samples, val_samples, test_samples, static_predictors, metadata_df
    gc.collect()

    # Save the transformers
    data_scaler.save_transformers()

    # Define a collate function that joins the sequences into batches
    def collate_fn(batch):
        # Sort the batch by sequence length in descending order
        batch = sorted(batch, key=lambda x: x['length'], reverse=True)

        # Calculate the new length as the maximum length of the sequences
        max_length = batch[0]['length']

        def _pad_with_nan(x, pad):
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            return torch.nn.functional.pad(x, (0, 0, 0, pad), value=float('nan')) if pad > 0 else x

        # Create the padded torch batches
        dyn_list, static_list, target_list = [], [], []
        for s in batch:
            dyn_list.append(_pad_with_nan(s['dynamic'], max_length - s['length']).unsqueeze(0))
            target_list.append(_pad_with_nan(s['target'], max_length - s['length']).unsqueeze(0))
            static_list.append(s['static'].unsqueeze(0) if s['static'].dim() == 1 else s['static'])
        dynamic_batch = torch.cat(dyn_list, dim=0)
        target_batch = torch.cat(target_list, dim=0)
        static_batch = torch.cat(static_list, dim=0)

        return {
            'dynamic': dynamic_batch,
            'static': static_batch,
            'target': target_batch,
            'length': torch.tensor(max_length, dtype=torch.int64),
            'station_id': [stn['station_id'] for stn in batch],
            'start_date': [stn['start_date'] for stn in batch],
            'seq_lengths': [stn['length'] - shared.WARMUP_DAYS for stn in batch],
            'alive_counts': torch.tensor(
                [sum(1 for sample in batch if sample['length'] > t) for t in range(max_length)],
                dtype=torch.int64
            )
        }

    # Check if there is GPU and the number of CPU cores available
    is_cuda = torch.cuda.is_available()

    # Create the train, validation, and test data loaders
    train_data_loader = DataLoader(
        trans_train_samples,
        batch_size=int(batch_size),
        shuffle=True,
        pin_memory=True if is_cuda else False,
        collate_fn=collate_fn
    )
    train_pred_loader = DataLoader(
        trans_train_samples,
        batch_size=len(trans_train_samples),
        pin_memory=is_cuda,
        collate_fn=collate_fn
    )

    val_data_loader = DataLoader(
        trans_val_samples,
        batch_size=len(trans_val_samples),
        pin_memory=is_cuda,
        collate_fn=collate_fn
    )
    test_data_loader = DataLoader(
        trans_test_samples,
        batch_size=len(trans_test_samples),
        pin_memory=is_cuda,
        collate_fn=collate_fn
    )

    return {
        'train': train_data_loader,
        'train_pred': train_pred_loader,
        'val': val_data_loader,
        'test': test_data_loader,
    }