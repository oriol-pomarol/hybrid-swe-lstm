from pathlib import Path
import torch.nn as nn
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import config

class LSTM(nn.Module):
    """LSTM with a linear output layer that predicts SWE (or its daily change, if config.PREDICT_dSWE) for each step.

    `hidden_size` is an int or [hidden_size, num_layers]. `dropout` is applied to the output and between
    stacked layers. `forward(inputs, hidden)` takes inputs of shape (batch, steps, input_size), whose last
    feature is the previous SWE, and returns (output, hidden).
    """
    def __init__(self, input_size, hidden_size=64, dropout=0.0):
        super().__init__()

        # Accept either an int or a list/tuple for hidden_size
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size, 1]

        # Dropout of the model output, also applied between stacked LSTM layers
        self.dropout = float(dropout)

        # Use a single nn.LSTM with num_layers
        self.predict_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size[0],
            batch_first=True,
            num_layers=hidden_size[1],
            dropout=self.dropout if hidden_size[1] > 1 else 0.0
        )

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size[0], 1)

    def forward(self, inputs, hidden=None):

        # Process through LSTM
        lstm_out, hidden = self.predict_lstm(inputs, hidden)
        output = self.fc(lstm_out)

        # Apply dropout if specified
        if self.dropout > 0:
            output = self.dropout_layer(output)

        # If predicting dSWE, add the output to the previous SWE
        if config.PREDICT_dSWE:
            prev_swe = inputs[..., -1:]  # Last input feature is prev_swe
            output += prev_swe

        return output, hidden