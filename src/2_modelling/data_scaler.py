import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
import sys

import torch
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import shared
from variable_attrs import VARIABLE_ATTRS

class DataScaler:
    """Scaling of the dynamic, static and target columns, fitted on the training stations.

    Each group gets an sklearn ColumnTransformer: log1p and standardisation for the variables with
    `log_scaling` in VARIABLE_ATTRS, standardisation for the others. The fitted transformers are pickled
    to `run_dir/models/<group>_transformer.pkl`; `get_transformed_zero()` gives the scaled 0 mm SWE.
    """
    def __init__(self, run_dir):
        self.output_path = run_dir / 'models'
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.prefix = shared.TEST_PREFIX
        self.transformers = {}
    
    def log1p_clip(self, X):
        return np.log1p(np.clip(X, 0, None))
    
    def initialize_transformers(self, col_names):
        
        for var_group, cols in col_names.items():
        
            # Retrieve the indices for log-scaled and standard-scaled columns
            log_std_indices = []
            std_indices = []
            for i, col in enumerate(cols):
                var_attrs = VARIABLE_ATTRS.get(col)
                if var_attrs is None:
                    raise ValueError(f"Column '{col}' not found in VARIABLE_ATTRS.")
                elif var_attrs.log_scaling == True:
                    log_std_indices.append(i)
                elif var_attrs.train_norm == True:
                    std_indices.append(i)

            # Define the log + standard scaling pipeline
            log_std_pipeline = Pipeline(steps=[
                ('log_transform', FunctionTransformer(func=self.log1p_clip, inverse_func=np.expm1)),
                ('std_scaler', StandardScaler())
            ])

            # Build the ColumnTransformer with appropriate pipelines
            transformer =  ColumnTransformer(
                transformers=[
                    ('log_std_scaler', log_std_pipeline, log_std_indices),
                    ('std_scaler', StandardScaler(), std_indices)
                ],
                remainder='passthrough'
            )
            self.transformers[var_group] = transformer
        
    def fit_samples(self, data_samples):
        for var_group, transformer in self.transformers.items():
            concat_df = pd.concat([sample[var_group] for sample in data_samples], axis=0)
            if var_group == 'target':
                concat_df = concat_df.dropna()
            self.transformers[var_group] = transformer.fit(concat_df)

    def to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
        return torch.from_numpy(np.asarray(x, dtype=np.float32))

    def transform_samples(self, data_samples):
        if not self.transformers:
            raise ValueError("Transformers have not been fitted or loaded.")
        for sample in data_samples:
            sample['dynamic'] = self.to_tensor(self.transformers['dynamic'].transform(sample['dynamic']))
            sample['static'] = self.to_tensor(self.transformers['static'].transform(sample['static']))
            sample['target'] = self.to_tensor(self.transformers['target'].transform(sample['target']))
        return data_samples
    
    def save_transformers(self):
        for var_group, transformer in self.transformers.items():
            file_path = self.output_path / f'{self.prefix}{var_group}_transformer.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(transformer, f)

    def load_transformers(self):
        for var_group in ['dynamic', 'static', 'target']:
            file_path = self.output_path / f'{self.prefix}{var_group}_transformer.pkl'
            with open(file_path, 'rb') as f:
                self.transformers[var_group] = pickle.load(f)

    def get_transformers(self):
        return self.transformers
    
    def get_transformed_zero(self):
        target_transformer = self.transformers.get('target')
        if target_transformer is None:
            raise ValueError("Target transformer has not been fitted or loaded.")
        zero_df = pd.DataFrame({'swe': [0.0]})
        transformed_zero = target_transformer.transform(zero_df)
        return float(transformed_zero[0, 0])