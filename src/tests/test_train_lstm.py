import importlib
import sys
import types
import unittest
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR / "common"))
sys.path.append(str(SRC_DIR / "2_modelling"))
import config
import shared

_ADDED_MODULES = ("train_lstm", "data_scaler", "lstm_definition", "sequence_runner")


def setUpModule():
    global train_lstm, _original_variable_attrs

    # data_scaler imports VARIABLE_ATTRS, which opens rasters at import time
    _original_variable_attrs = sys.modules.get("variable_attrs")
    fake_variable_attrs = types.ModuleType("variable_attrs")
    fake_variable_attrs.VARIABLE_ATTRS = None
    sys.modules["variable_attrs"] = fake_variable_attrs
    try:
        train_lstm = importlib.import_module("train_lstm")
    finally:
        if _original_variable_attrs is None:
            sys.modules.pop("variable_attrs", None)
        else:
            sys.modules["variable_attrs"] = _original_variable_attrs


def tearDownModule():
    for name in _ADDED_MODULES:
        sys.modules.pop(name, None)


class TestCalculateLoss(unittest.TestCase):
    def setUp(self):
        self.criterion = torch.nn.MSELoss(reduction="sum")

    def test_nan_targets_are_ignored(self):
        predictions = torch.tensor([[1.0, 2.0, 3.0], [0.0, 7.0, 1.0]])
        targets = torch.tensor([[1.0, float("nan"), 5.0], [float("nan"), float("nan"), 2.0]])
        loss, n_samples = train_lstm.calculate_loss(predictions, targets, self.criterion)
        self.assertEqual(n_samples, 3)
        self.assertAlmostEqual(loss.item(), 0.0 + 4.0 + 1.0, places=6)

    def test_nan_predictions_on_nan_targets_are_allowed(self):
        predictions = torch.tensor([[1.0, float("nan")]])
        targets = torch.tensor([[2.0, float("nan")]])
        loss, n_samples = train_lstm.calculate_loss(predictions, targets, self.criterion)
        self.assertEqual(n_samples, 1)
        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_nan_prediction_on_a_valid_target_raises(self):
        predictions = torch.tensor([[float("nan"), 1.0]])
        targets = torch.tensor([[1.0, 1.0]])
        with self.assertRaises(ValueError):
            train_lstm.calculate_loss(predictions, targets, self.criterion)

    def test_all_nan_targets_raise(self):
        predictions = torch.zeros(2, 3)
        targets = torch.full((2, 3), float("nan"))
        with self.assertRaises(ValueError):
            train_lstm.calculate_loss(predictions, targets, self.criterion)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            train_lstm.calculate_loss(torch.zeros(2, 3), torch.zeros(2, 4), self.criterion)


@unittest.skipIf(config.TEST_MODE, "set TEST_MODE = False in config.py to run these tests")
class TestResolveHparams(unittest.TestCase):
    def test_defaults(self):
        hp = train_lstm.resolve_hparams()
        for key, value in shared.HPARAM_DEFAULTS.items():
            self.assertEqual(hp[key], value, msg=key)
        self.assertEqual(hp["patience"], max(1, int(shared.HPARAM_DEFAULTS["n_epochs"] * 0.1)))

    def test_tune_uses_the_full_defaults(self):
        self.assertEqual(train_lstm.resolve_hparams(station_set="tune"), train_lstm.resolve_hparams())

    def test_subset_uses_the_small_run_defaults(self):
        hp = train_lstm.resolve_hparams(station_set="subset")
        for key, value in shared.SMALL_RUN_HPARAM_DEFAULTS.items():
            self.assertEqual(hp[key], value, msg=key)
        self.assertEqual(hp["learning_rate"], shared.HPARAM_DEFAULTS["learning_rate"])
        self.assertEqual(hp["patience"], max(1, int(shared.SMALL_RUN_HPARAM_DEFAULTS["n_epochs"] * 0.1)))

    def test_given_values_override_the_defaults(self):
        given = {"batch_size": 128, "hidden_size": [32, 2], "optimizer_name": "RMSprop", "dropout": 0.2}
        hp = train_lstm.resolve_hparams(given, station_set="subset")
        self.assertEqual(hp["batch_size"], 128)
        self.assertEqual(hp["hidden_size"], [32, 2])
        self.assertEqual(hp["optimizer_name"], "RMSprop")
        self.assertEqual(hp["dropout"], 0.2)

    def test_patience_is_ten_percent_of_the_epochs_and_at_least_one(self):
        self.assertEqual(train_lstm.resolve_hparams({"n_epochs": 200})["patience"], 20)
        self.assertEqual(train_lstm.resolve_hparams({"n_epochs": 5})["patience"], 1)
        self.assertEqual(train_lstm.resolve_hparams({"n_epochs": 200, "patience": 7})["patience"], 7)

    def test_resolving_twice_changes_nothing(self):
        hp = train_lstm.resolve_hparams({"learning_rate": 0.0005, "n_epochs": 50})
        self.assertEqual(train_lstm.resolve_hparams(hp), hp)

    def test_input_is_not_modified(self):
        given = {"learning_rate": 0.0005}
        train_lstm.resolve_hparams(given)
        self.assertEqual(given, {"learning_rate": 0.0005})

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            train_lstm.resolve_hparams({"lr": 0.001})

    def test_unknown_optimizer_raises(self):
        with self.assertRaises(ValueError):
            train_lstm.resolve_hparams({"optimizer_name": "Adagrad"})


if __name__ == "__main__":
    unittest.main()
