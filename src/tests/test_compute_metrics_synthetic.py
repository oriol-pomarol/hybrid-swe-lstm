import importlib
import importlib.util
import math
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR / "common"))
import config

COMPUTE_METRICS_PATH = SRC_DIR / "3_analysis" / "compute_metrics.py"


def _load_compute_metrics_module(module_path, base_path):
    fake_local_paths = types.ModuleType("local_paths")
    fake_local_paths.BASE_PATH = base_path

    original_local_paths = sys.modules.get("local_paths")
    sys.modules["local_paths"] = fake_local_paths
    try:
        spec = importlib.util.spec_from_file_location("compute_metrics_under_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if original_local_paths is None:
            sys.modules.pop("local_paths", None)
        else:
            sys.modules["local_paths"] = original_local_paths


@contextmanager
def _import_compute_metrics_module_for_multiprocessing(module_path, base_path):
    fake_modules_dir = base_path / "_fake_modules"
    fake_modules_dir.mkdir(parents=True, exist_ok=True)

    fake_local_paths_path = fake_modules_dir / "local_paths.py"
    fake_local_paths_path.write_text(
        "from pathlib import Path\n"
        f"BASE_PATH = Path({str(base_path)!r})\n",
        encoding="utf-8",
    )

    original_sys_path = list(sys.path)
    original_compute_metrics = sys.modules.pop("compute_metrics", None)
    original_local_paths = sys.modules.pop("local_paths", None)
    try:
        sys.path.insert(0, str(fake_modules_dir))
        sys.path.insert(1, str(module_path.parent))
        module = importlib.import_module("compute_metrics")
        yield module
    finally:
        sys.modules.pop("compute_metrics", None)
        sys.modules.pop("local_paths", None)
        if original_compute_metrics is not None:
            sys.modules["compute_metrics"] = original_compute_metrics
        if original_local_paths is not None:
            sys.modules["local_paths"] = original_local_paths
        sys.path[:] = original_sys_path


@unittest.skipIf(config.TEST_MODE, "set TEST_MODE = False in config.py to run these tests")
class TestComputeMetricsSynthetic(unittest.TestCase):
    def test_weighted_metrics_single_station(self):
        with TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)

            # Minimal folder layout expected by compute_metrics.py
            (base_path / "data" / "temp").mkdir(parents=True, exist_ok=True)
            run_dir = base_path / "results" / "synthetic_run"
            (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

            weights_df = pd.DataFrame({"500-": [1.0]}, index=["Prairie"])
            weights_df.to_csv(base_path / "data" / "CLASS_ELEV_WEIGHTS.csv")

            metadata_df = pd.DataFrame(
                {
                    "SnowClass": ["Prairie"],
                    "ELEV": [100.0],
                },
                index=[1],
            )
            metadata_df.to_csv(base_path / "data" / "STATION_METADATA.csv")

            dates = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"])
            target_df = pd.DataFrame({"swe": [0.0, 2.0, 1.0]}, index=dates)
            target_df.to_csv(base_path / "data" / "temp" / "dynamic_predictors_ID_1.csv")

            pred_df = pd.DataFrame(
                {
                    "station_id": [1, 1, 1],
                    "Date": dates,
                    "pred_swe": [0.0, 1.0, 1.0],
                }
            )
            pred_df.to_csv(run_dir / "predictions" / "NH_pSWE_eval.csv", index=False)

            station_splits_df = pd.DataFrame({"split": ["eval"]}, index=[1])
            station_splits_df.to_csv(run_dir / "station_splits.csv")

            compute_metrics_module = _load_compute_metrics_module(COMPUTE_METRICS_PATH, base_path)

            # Keep test deterministic and avoid multiprocessing in unit tests.
            compute_metrics_module.max_workers = 1
            metrics = compute_metrics_module.compute_metrics(
                results_dir=run_dir, split_type="eval", threshold=0.0, apply_weights=True
            )

            self.assertEqual(metrics["n_full"], 3)
            self.assertTrue(math.isclose(metrics["mse_full"], 1.0 / 3.0, rel_tol=1e-12))
            self.assertTrue(math.isclose(metrics["rmse_full"], (1.0 / 3.0) ** 0.5, rel_tol=1e-12))
            self.assertTrue(math.isclose(metrics["mae_full"], 1.0 / 3.0, rel_tol=1e-12))
            self.assertTrue(math.isclose(metrics["bias_full"], -1.0 / 3.0, rel_tol=1e-12))
            self.assertTrue(math.isclose(metrics["pbias_full"], -100.0 / 3.0, rel_tol=1e-12))
            self.assertTrue(math.isclose(metrics["nse_full"], 0.5, rel_tol=1e-12))

            self.assertEqual(metrics["n_snow"], 2)
            self.assertTrue(math.isclose(metrics["rmse_snow"], (0.5) ** 0.5, rel_tol=1e-12))
            self.assertTrue(math.isclose(metrics["bias_snow"], -0.5, rel_tol=1e-12))

            self.assertEqual(metrics["n_yearly"], 0)
            self.assertIsNone(metrics["peak_mae_yearly"])
            self.assertIsNone(metrics["peak_mb_yearly"])
            self.assertIsNone(metrics["peak_timing_mae_yearly"])
            self.assertIsNone(metrics["peak_timing_mb_yearly"])
            self.assertIsNone(metrics["snowpack_duration_mae_yearly"])
            self.assertIsNone(metrics["snowpack_duration_mb_yearly"])

    def test_weighted_metrics_multiprocessing_matches_serial(self):
        with TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)

            # Minimal folder layout expected by compute_metrics.py
            (base_path / "data" / "temp").mkdir(parents=True, exist_ok=True)
            run_dir = base_path / "results" / "synthetic_run"
            (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

            weights_df = pd.DataFrame({"500-": [1.0]}, index=["Prairie"])
            weights_df.to_csv(base_path / "data" / "CLASS_ELEV_WEIGHTS.csv")

            metadata_df = pd.DataFrame(
                {
                    "SnowClass": ["Prairie", "Prairie"],
                    "ELEV": [100.0, 200.0],
                },
                index=[1, 2],
            )
            metadata_df.to_csv(base_path / "data" / "STATION_METADATA.csv")

            dates = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"])
            target_1_df = pd.DataFrame({"swe": [0.0, 2.0, 1.0]}, index=dates)
            target_1_df.to_csv(base_path / "data" / "temp" / "dynamic_predictors_ID_1.csv")
            target_2_df = pd.DataFrame({"swe": [1.0, 0.0, 1.0]}, index=dates)
            target_2_df.to_csv(base_path / "data" / "temp" / "dynamic_predictors_ID_2.csv")

            pred_df = pd.DataFrame(
                {
                    "station_id": [1, 1, 1, 2, 2, 2],
                    "Date": list(dates) + list(dates),
                    "pred_swe": [0.0, 1.0, 1.0, 1.0, 0.0, 2.0],
                }
            )
            pred_df.to_csv(run_dir / "predictions" / "NH_pSWE_eval.csv", index=False)

            station_splits_df = pd.DataFrame({"split": ["eval", "eval"]}, index=[1, 2])
            station_splits_df.to_csv(run_dir / "station_splits.csv")

            with _import_compute_metrics_module_for_multiprocessing(COMPUTE_METRICS_PATH, base_path) as compute_metrics_module:
                compute_metrics_module.max_workers = 1
                serial_metrics = compute_metrics_module.compute_metrics(
                    results_dir=run_dir, split_type="eval", threshold=0.0, apply_weights=True
                )

                compute_metrics_module.max_workers = 2
                parallel_metrics = compute_metrics_module.compute_metrics(
                    results_dir=run_dir, split_type="eval", threshold=0.0, apply_weights=True
                )

            self.assertEqual(parallel_metrics["n_full"], 6)
            self.assertTrue(math.isclose(parallel_metrics["mse_full"], 1.0 / 3.0, rel_tol=1e-12))
            self.assertEqual(parallel_metrics["n_snow"], 4)

            for key, serial_val in serial_metrics.items():
                parallel_val = parallel_metrics[key]
                if isinstance(serial_val, float):
                    self.assertTrue(
                        math.isclose(parallel_val, serial_val, rel_tol=1e-12, abs_tol=1e-12),
                        msg=f"Mismatch for metric '{key}'",
                    )
                else:
                    self.assertEqual(parallel_val, serial_val)


if __name__ == "__main__":
    unittest.main()
