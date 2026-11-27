"""Protocol invariants that protect the scientific comparison, using tiny data."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.deployment_protocol import (
    GridCondition, config_digest, coverage_summary, degrade_window, degrade_windows,
    iter_conditions, load_config, noise_seeds_for_condition, standardized_noise,
    validate_config, window_quality,
)
from research.degradation import downsample_window


class DeploymentProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()

    def window(self):
        # A complete half-open 120s reference window sampled once per second.
        return pd.DataFrame({"datetime": pd.date_range("2008-01-01", periods=120, freq="s"),
                             "x": np.arange(120, dtype=float) * 2.0,
                             "y": np.arange(120, dtype=float) * 3.0})

    def test_grid_and_stress_boundary(self):
        conditions = iter_conditions(self.cfg)
        self.assertEqual(len(conditions), 16)
        self.assertEqual(sum(not c.is_stress for c in conditions), 12)
        self.assertEqual(len({c.name for c in conditions}), 16)
        w = self.window()
        for c in conditions:
            d = degrade_window(w, c, window_id=17)
            self.assertEqual(window_quality(d, self.cfg)["usable"], not c.is_stress)
            if c.is_stress:
                self.assertEqual(len(d), 4)

    def test_bucket_first_and_copy_match_legacy(self):
        w = self.window().iloc[[0, 4, 5, 9, 10, 19, 20]].copy()
        before = w.copy(deep=True)
        c = GridCondition("ds5_n0", 5, 0)
        got = degrade_window(w, c, window_id="a")
        pd.testing.assert_frame_equal(got, downsample_window(w, 5))
        pd.testing.assert_frame_equal(w, before)

    def test_noise_pairs_original_points_across_sampling_and_magnitudes(self):
        w = self.window()
        full = degrade_window(w, GridCondition("ds0_n5", 0, 5), 2, window_id=42)
        sparse5 = degrade_window(w, GridCondition("ds10_n5", 10, 5), 2, window_id=42)
        sparse20 = degrade_window(w, GridCondition("ds10_n20", 10, 20), 2, window_id=42)
        take = np.arange(0, 120, 10)
        np.testing.assert_allclose(full[["x", "y"]].to_numpy()[take], sparse5[["x", "y"]].to_numpy())
        native = w[["x", "y"]].to_numpy()[take]
        np.testing.assert_allclose(sparse20[["x", "y"]].to_numpy() - native,
                                   4 * (sparse5[["x", "y"]].to_numpy() - native), atol=1e-12)

    def test_noise_does_not_depend_on_window_order(self):
        w = self.window()
        c = GridCondition("ds10_n10", 10, 10)
        left, qleft = degrade_windows([w, w], ["001", "002"], c, 0, self.cfg)
        right, _ = degrade_windows([w, w], ["002", "001"], c, 0, self.cfg)
        pd.testing.assert_frame_equal(left[0], right[1])
        self.assertFalse(np.array_equal(left[0].x.to_numpy(), left[1].x.to_numpy()))
        self.assertEqual(coverage_summary(qleft), {"n_reference": 2, "n_usable": 2, "retention": 1.0})
        self.assertFalse(np.array_equal(standardized_noise("001", 120, 0), standardized_noise("001", 120, 1)))

    def test_noisy_coordinates_do_not_change_coverage(self):
        w = self.window()
        for ds in (0, 5, 10, 30):
            quiet = degrade_window(w, GridCondition("quiet", ds, 0), window_id=1)
            noisy = degrade_window(w, GridCondition("noisy", ds, 20), 1, window_id=1)
            self.assertEqual(window_quality(quiet, self.cfg), window_quality(noisy, self.cfg))
        broken = w.iloc[::10].copy()
        broken.loc[broken.index[-1], "datetime"] += pd.Timedelta(seconds=100)
        self.assertEqual(window_quality(broken, self.cfg)["reason"], "gap_too_large")

    def test_repeat_budget_and_configuration_guards(self):
        for c in iter_conditions(self.cfg):
            expected = [0, 1, 2] if c.name == "ds10_n10" else [0]
            self.assertEqual(noise_seeds_for_condition(self.cfg, c), expected)
        changed = copy.deepcopy(self.cfg)
        changed["degradation"]["main_downsample_sec"] = [0, 10, 30]
        with self.assertRaises(ValueError):
            validate_config(changed)
        changed = copy.deepcopy(self.cfg)
        changed["seeds"]["repeat_conditions"] = ["ds0_n0"]
        with self.assertRaises(ValueError):
            validate_config(changed)
        changed = copy.deepcopy(self.cfg)
        changed["split"]["method"] = "random_window_split"
        with self.assertRaises(ValueError):
            validate_config(changed)
        self.assertEqual(config_digest(self.cfg), config_digest(dict(reversed(list(self.cfg.items())))))

    def test_ids_and_order_validation(self):
        w = self.window()
        c = GridCondition("ds0_n0", 0, 0)
        with self.assertRaises(ValueError):
            degrade_windows([w, w], [1, 1], c, 0, self.cfg)
        with self.assertRaises(ValueError):
            degrade_windows([w], [], c, 0, self.cfg)
        with self.assertRaises(ValueError):
            degrade_window(w.iloc[::-1], c, window_id=1)


if __name__ == "__main__":
    unittest.main()
