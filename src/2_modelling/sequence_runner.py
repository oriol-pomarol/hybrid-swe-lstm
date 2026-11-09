import torch
from tqdm import tqdm

class SequenceRunner:
    """Run autoregressive LSTM predictions for a batch.

    This class encapsulates warmup and stepwise prediction logic so both
    training and inference can share the same implementation.
    """

    def __init__(self, batch, model, device, trans_zero):

        # Cache the model, device, and transformed zero value for SWE
        self.model = model
        self.device = device
        self.trans_zero = trans_zero

        # Move data to device and cache frequently used fields
        self.dynamic = batch['dynamic'].to(device)
        self.static = batch['static'].to(device)
        self.alive_counts = batch['alive_counts']
        self.max_length = batch['length'].item()
        self.batch_size = self.dynamic.shape[0]

        # Initialize SWE as 0 at t=0, and hidden state as None
        self.t = 0
        self.hidden = None
        self.prev_swe = torch.full((self.batch_size, 1, 1), self.trans_zero, device=device)

    def step(self):
        """Advance one time step and return predictions for alive sequences."""

        # Slice the predictors to alive sequences at the current time step
        n_valid_seq = self.alive_counts[self.t].item()
        current_dynamic = self.dynamic[:n_valid_seq, self.t].unsqueeze(1)
        current_static = self.static[:n_valid_seq].unsqueeze(1)

        # For all but the first step, slice previous SWE and hidden state
        if self.t > 0:
            self.prev_swe = self.prev_swe[:n_valid_seq]
            self.hidden = (self.hidden[0][:, :n_valid_seq, :],
                           self.hidden[1][:, :n_valid_seq, :])
            
        # Concatenate predictors and previous SWE for the LSTM input
        lstm_inputs = torch.cat([current_dynamic, current_static, self.prev_swe], dim=-1)

        # Forward predict one step
        self.prev_swe, self.hidden = self.model.forward(
            lstm_inputs, hidden=self.hidden
        )

        self.t += 1
        return self.prev_swe
    
    def run_warmup(self, n_steps):
        """Run a warmup period of `n_steps` steps to initialize swe and hidden state."""
        for _ in range(n_steps):
            self.step()

    def run_predictions(self, n_steps, verbose=False):
        """Run predictions for up to `n_steps` steps starting at current t.
           Returns a tensor of shape `(batch_size, n_steps)`."""

        # Initialize a tensor filled with NaNs to hold predictions
        valid_steps = min(n_steps, self.max_length - self.t)
        predictions = torch.full(
            (self.batch_size, valid_steps),
            float('nan'),
            dtype=torch.float32,
            device=self.device,
        )

        # Run stepwise predictions
        iterable = range(valid_steps)
        if verbose:
            iterable = tqdm(iterable, desc="Running predictions", unit="step")
        for step_idx in iterable:
            n_valid_seq = self.alive_counts[self.t].item()
            pred_swe = self.step()
            predictions[:n_valid_seq, step_idx] = pred_swe.squeeze()

        return predictions
    
    def detach_state(self):
        """Detach hidden state and previous SWE to truncate backprop."""
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.prev_swe = self.prev_swe.detach()
