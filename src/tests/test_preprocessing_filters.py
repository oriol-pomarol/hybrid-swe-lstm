import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
SAVE_TRAINING_SEQUENCES_PATH = SRC_DIR / "1_preprocessing" / "save_training_sequences.py"


class _FakeVariableAttrs:
    def find(self, **kwargs):
        return []


def _load_save_training_sequences_module():
    fake_local_paths = types.ModuleType("local_paths")
    fake_local_paths.BASE_PATH = Path("unused_base_path")
    fake_variable_attrs = types.ModuleType("variable_attrs")
    fake_variable_attrs.VARIABLE_ATTRS = _FakeVariableAttrs()

    fakes = {"local_paths": fake_local_paths, "variable_attrs": fake_variable_attrs}
    originals = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        spec = importlib.util.spec_from_file_location("save_training_sequences_under_test", SAVE_TRAINING_SEQUENCES_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def setUpModule():
    global sts
    sts = _load_save_training_sequences_module()


def _nan_rows_frame(nan_rows, n_rows=20):
    df = pd.DataFrame({"pr": np.ones(n_rows), "tas": np.ones(n_rows)},
                      index=pd.date_range("2000-01-01", periods=n_rows, freq="D"))
    df.iloc[list(nan_rows), 0] = np.nan
    return df


class TestHasLargeMissingClusters(unittest.TestCase):
    def test_no_missing_values(self):
        self.assertFalse(sts.has_large_missing_clusters(_nan_rows_frame([])))

    def test_run_of_five_missing_days_is_kept(self):
        self.assertFalse(sts.has_large_missing_clusters(_nan_rows_frame(range(3, 8))))

    def test_run_of_six_missing_days_is_dropped(self):
        self.assertTrue(sts.has_large_missing_clusters(_nan_rows_frame(range(3, 9))))

    def test_scattered_missing_days_are_kept(self):
        self.assertFalse(sts.has_large_missing_clusters(_nan_rows_frame([1, 3, 5, 7, 9, 11, 13])))

    def test_missing_value_in_one_column_counts(self):
        df = _nan_rows_frame([])
        df.iloc[2:8, 1] = np.nan
        self.assertTrue(sts.has_large_missing_clusters(df))


class TestFindLargestConsecutiveGap(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(sts.find_largest_consecutive_gap(pd.DatetimeIndex([])), 0)

    def test_single_date(self):
        self.assertEqual(sts.find_largest_consecutive_gap(pd.DatetimeIndex(["2000-01-01"])), 1)

    def test_consecutive_dates(self):
        dates = pd.date_range("2000-01-01", periods=10, freq="D")
        self.assertEqual(sts.find_largest_consecutive_gap(dates), 10)

    def test_longest_of_several_runs(self):
        dates = pd.date_range("2000-01-01", periods=3, freq="D").append(
            pd.date_range("2000-01-10", periods=5, freq="D"))
        self.assertEqual(sts.find_largest_consecutive_gap(dates), 5)


class TestRemoveYearsWithMissingMonths(unittest.TestCase):
    def setUp(self):
        # Snow years 2001, 2002 and 2003 (1 Oct 2000 to 30 Sep 2003), all observed
        dates = pd.date_range("2000-10-01", "2003-09-30", freq="D")
        self.series = pd.Series(1.0, index=dates, name="swe")
        # 2001: a 121-day winter gap (1 Nov 2000 to 1 Mar 2001)
        self.series.loc["2000-11-01":"2001-03-01"] = np.nan
        # 2002: a 120-day winter gap (1 Nov 2001 to 28 Feb 2002)
        self.series.loc["2001-11-01":"2002-02-28"] = np.nan
        # 2003: a 153-day summer gap (1 May to 30 Sep 2003)
        self.series.loc["2003-05-01":"2003-09-30"] = np.nan

    def test_gap_lengths_of_the_fixture(self):
        self.assertEqual(int(self.series.loc["2000-10-01":"2001-09-30"].isna().sum()), 121)
        self.assertEqual(int(self.series.loc["2001-10-01":"2002-09-30"].isna().sum()), 120)

    def test_only_the_year_with_a_winter_gap_over_120_days_is_dropped(self):
        series, dropped_years, n_obs_dropped = sts.remove_years_with_missing_months(self.series.copy())

        self.assertEqual(dropped_years, [2001])
        self.assertEqual(n_obs_dropped, 365 - 121)

        self.assertTrue(series.loc["2000-10-01":"2001-09-30"].isna().all())
        self.assertEqual(int(series.loc["2001-10-01":"2002-09-30"].notna().sum()), 365 - 120)
        self.assertEqual(int(series.loc["2002-10-01":"2003-09-30"].notna().sum()), 365 - 153)

    def test_no_gaps_drops_nothing(self):
        series = pd.Series(1.0, index=pd.date_range("2000-10-01", "2002-09-30", freq="D"))
        cleaned, dropped_years, n_obs_dropped = sts.remove_years_with_missing_months(series.copy())
        self.assertEqual(dropped_years, [])
        self.assertEqual(n_obs_dropped, 0)
        pd.testing.assert_series_equal(cleaned, series)


if __name__ == "__main__":
    unittest.main()
