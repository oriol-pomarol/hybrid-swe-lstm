import math
import sys
import unittest
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR / "2_modelling"))


class _CumulativeModel:
    """Fake model: output = previous SWE (last input) + first dynamic predictor; counts steps in the hidden state."""

    def __init__(self):
        self.inputs = []

    def forward(self, inputs, hidden=None):
        self.inputs.append(inputs.clone())
        output = inputs[..., -1:] + inputs[..., :1]
        n_seq = inputs.shape[0]
        h = torch.zeros(1, n_seq, 1) if hidden is None else hidden[0]
        h = h + 1
        return output, (h, h.clone())


def _make_batch(lengths, dynamic_values):
    """Batch as built by collate_fn: rows sorted by length (longest first), padded with NaN."""
    max_length = max(lengths)
    n_dyn, n_static = 2, 1
    dynamic = torch.full((len(lengths), max_length, n_dyn), float("nan"))
    for row, (length, value) in enumerate(zip(lengths, dynamic_values)):
        dynamic[row, :length, 0] = value
        dynamic[row, :length, 1] = -1.0
    static = torch.arange(len(lengths), dtype=torch.float32).unsqueeze(1).repeat(1, n_static) + 10.0
    alive_counts = torch.tensor([sum(1 for length in lengths if length > t) for t in range(max_length)],
                                dtype=torch.int64)
    return {
        "dynamic": dynamic,
        "static": static,
        "length": torch.tensor(max_length, dtype=torch.int64),
        "alive_counts": alive_counts,
    }


class TestSequenceRunner(unittest.TestCase):
    def setUp(self):
        from sequence_runner import SequenceRunner
        self.SequenceRunner = SequenceRunner
        self.lengths = [4, 3, 1]
        self.batch = _make_batch(self.lengths, dynamic_values=[1.0, 2.0, 3.0])
        self.trans_zero = 0.5

    def _runner(self, model):
        return self.SequenceRunner(self.batch, model, torch.device("cpu"), self.trans_zero)

    def test_inputs_are_dynamic_static_then_previous_swe(self):
        model = _CumulativeModel()
        runner = self._runner(model)
        runner.step()
        runner.step()

        first = model.inputs[0]
        self.assertEqual(tuple(first.shape), (3, 1, 4))
        expected_first = torch.tensor([[[1.0, -1.0, 10.0, 0.5]],
                                       [[2.0, -1.0, 11.0, 0.5]],
                                       [[3.0, -1.0, 12.0, 0.5]]])
        self.assertTrue(torch.equal(first, expected_first))

        second = model.inputs[1]
        self.assertEqual(tuple(second.shape), (2, 1, 4))
        self.assertTrue(torch.equal(second[:, 0, -1], torch.tensor([1.5, 2.5])))

    def test_predictions_are_autoregressive_and_nan_after_the_end(self):
        runner = self._runner(_CumulativeModel())
        predictions = runner.run_predictions(n_steps=10)

        nan = float("nan")
        expected = [[1.5, 2.5, 3.5, 4.5],
                    [2.5, 4.5, 6.5, nan],
                    [3.5, nan, nan, nan]]
        self.assertEqual(tuple(predictions.shape), (3, 4))
        for row, expected_row in enumerate(expected):
            for col, value in enumerate(expected_row):
                actual = predictions[row, col].item()
                if math.isnan(value):
                    self.assertTrue(math.isnan(actual), msg=f"row {row}, step {col}")
                else:
                    self.assertAlmostEqual(actual, value, places=6, msg=f"row {row}, step {col}")
        self.assertEqual(runner.t, 4)

    def test_hidden_state_follows_the_alive_sequences(self):
        runner = self._runner(_CumulativeModel())
        runner.run_predictions(n_steps=4)
        self.assertEqual(tuple(runner.hidden[0].shape), (1, 1, 1))
        self.assertEqual(runner.hidden[0].item(), 4.0)
        self.assertEqual(tuple(runner.prev_swe.shape), (1, 1, 1))

    def test_warmup_advances_without_returning_predictions(self):
        runner = self._runner(_CumulativeModel())
        self.assertIsNone(runner.run_warmup(2))
        self.assertEqual(runner.t, 2)

        predictions = runner.run_predictions(n_steps=10)
        self.assertEqual(tuple(predictions.shape), (3, 2))
        self.assertAlmostEqual(predictions[0, 0].item(), 3.5, places=6)
        self.assertAlmostEqual(predictions[1, 0].item(), 6.5, places=6)
        self.assertTrue(math.isnan(predictions[1, 1].item()))
        self.assertTrue(torch.isnan(predictions[2]).all())

    def test_chunks_continue_where_the_previous_one_stopped(self):
        runner = self._runner(_CumulativeModel())
        first = runner.run_predictions(n_steps=2)
        second = runner.run_predictions(n_steps=2)
        self.assertTrue(torch.allclose(torch.cat([first, second], dim=1)[0], torch.tensor([1.5, 2.5, 3.5, 4.5])))

    def test_detach_state_cuts_the_graph(self):
        from lstm_definition import LSTM
        torch.manual_seed(0)
        model = LSTM(input_size=4, hidden_size=8)
        runner = self._runner(model)
        runner.run_predictions(n_steps=2)
        self.assertTrue(runner.prev_swe.requires_grad and runner.prev_swe.grad_fn is not None)
        runner.detach_state()
        self.assertIsNone(runner.prev_swe.grad_fn)
        self.assertIsNone(runner.hidden[0].grad_fn)
        self.assertIsNone(runner.hidden[1].grad_fn)

    def test_real_lstm_gives_finite_values_only_where_sequences_are_alive(self):
        from lstm_definition import LSTM
        torch.manual_seed(0)
        model = LSTM(input_size=4, hidden_size=[8, 2])
        with torch.no_grad():
            predictions = self._runner(model).run_predictions(n_steps=4)
        for row, length in enumerate(self.lengths):
            self.assertTrue(torch.isfinite(predictions[row, :length]).all())
            self.assertTrue(torch.isnan(predictions[row, length:]).all())


if __name__ == "__main__":
    unittest.main()
