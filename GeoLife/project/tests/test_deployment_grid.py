"""Small independent fixtures; no GeoLife training or real research F1 output.

Run: python -m unittest discover -s project/tests -p test_deployment_grid.py -v
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from research import deployment_grid as grid
from research.deployment_protocol import load_config
from research.segmentation import segment_windows


def fixture():
    cfg = copy.deepcopy(load_config())
    # These changes are isolated to a synthetic fixture. The production loader
    # rejects such changes to the frozen protocol, and all outputs are temporary.
    cfg["models"]["rf"]["n_estimators"] = 2
    cfg["models"]["gru"].update(epochs=1, hidden=4, batch_size=5)
    frames = []
    for i in range(10):
        t = np.arange(115)
        frames.append(pd.DataFrame({
            "datetime": pd.Timestamp("2020-01-01") + pd.to_timedelta(t, unit="s"),
            "x": t * (i % 5 + 1.0), "y": np.sin(t / 10) * 3}))
    meta = pd.DataFrame({"window_id": np.arange(10), "user_id": np.repeat(["010", "020"], 5),
                         "label": cfg["classes"] * 2})
    support = {f"ds{s}": np.full(10, s != 30, dtype=bool) for s in [0, 5, 10, 30]}
    support["common"] = np.ones(10, dtype=bool)
    data = SimpleNamespace(meta=meta, support=support, frame=lambda i: frames[i])
    return cfg, data, grid._extract(frames, 128)


class DeploymentGridTests(unittest.TestCase):
    def test_fast_segmentation_preserves_original_window_ids(self):
        cfg = load_config()
        rng = np.random.default_rng(12)
        points = pd.DataFrame({
            "user_id": np.repeat(["010", "020"], 300), "traj_id": "trip",
            "datetime": pd.Timestamp("2020-01-01") + pd.to_timedelta(
                np.tile(np.cumsum(rng.integers(0, 7, 300)), 2), unit="s"),
            "label": np.tile(np.repeat(["walk", None, "walk", "bike", None, "bus"], 50), 2),
            "x": rng.normal(size=600), "y": rng.normal(size=600),
        }).sample(frac=1, random_state=42).reset_index(drop=True)
        packed, fast, starts, counts = grid._fast_reference(points, cfg)
        legacy, legacy_frames = segment_windows(points)
        pd.testing.assert_frame_equal(fast, legacy.reset_index(), check_dtype=False)
        for start, count, expected in zip(starts, counts, legacy_frames):
            pd.testing.assert_frame_equal(packed.iloc[start:start + count].reset_index(drop=True),
                                          expected, check_dtype=False)

    def test_inference_only_fails_before_building_features(self):
        cfg, data, clean = fixture()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(grid, "ARTIFACTS", Path(tmp) / "artifacts"), \
                patch.object(grid, "prepare", return_value={"fingerprint": "fixture-only"}), \
                patch.object(grid, "Windows", return_value=data), \
                patch.object(grid, "_clean_features") as extract:
            with self.assertRaisesRegex(FileNotFoundError, "No complete saved model"):
                grid.evaluate(cfg, Path(tmp) / "out", ["010"], threads=2,
                              inference_only=True, smoke=True)
            extract.assert_not_called()

    def test_training_reload_resume_partial_metrics_and_corruption(self):
        cfg, data, clean = fixture()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(grid, "ARTIFACTS", Path(tmp) / "artifacts"), \
                patch.object(grid, "prepare", return_value={"fingerprint": "fixture-only"}), \
                patch.object(grid, "Windows", return_value=data), \
                patch.object(grid, "_clean_features", return_value=clean), \
                patch.object(torch.cuda, "is_available", return_value=False):
            out = Path(tmp) / "out"
            run = grid.evaluate(cfg, out, ["010"], threads=2, smoke=True)
            fold = run / "folds" / "010"
            before = {p.name: p.stat().st_mtime_ns for p in fold.iterdir()}
            grid.evaluate(cfg, out, ["010"], threads=2, inference_only=True, smoke=True)
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in fold.iterdir()})
            result = pd.read_csv(out / "partial_grid_results.csv")
            self.assertEqual(len(result), 36)  # 18 condition/seed evaluations x two cohorts
            self.assertTrue(result["scope"].eq("partial_louo").all())
            self.assertFalse((out / "grid_results.csv").exists())
            self.assertTrue(result.loc[result["is_stress"], "rf_macro_f1"].isna().all())
            self.assertTrue(result.loc[result["is_stress"], "gru_macro_f1"].isna().all())
            self.assertTrue(result.loc[~result["is_stress"], "n_scored"].eq(5).all())
            # Reuse is conditional on a matching receipt and model hashes.
            receipt = next(fold.glob("*.csv.gz.json"))
            receipt.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Prediction cache corrupted"):
                grid.evaluate(cfg, out, ["010"], threads=2, inference_only=True, smoke=True)


if __name__ == "__main__":
    unittest.main()
