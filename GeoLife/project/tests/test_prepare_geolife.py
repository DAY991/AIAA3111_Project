"""Synthetic parser/IO regression checks, not experimental result validation.

From GeoLife/: python -m unittest discover -s project/tests -p test_prepare_geolife.py -v
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_geolife.py"
SPEC = importlib.util.spec_from_file_location("prepare_geolife_cli", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def make_user(root: Path, user: str, intervals, seconds=(0, 5, 10, 15), date="2008-01-01"):
    directory = root / user
    trajectory = directory / "Trajectory"
    trajectory.mkdir(parents=True)
    headers = ["Geolife trajectory", "WGS 84", "Altitude is in Feet", "Reserved 3", "0,2,255,", "0"]
    rows = [f"39.900{index},116.300{index},0,100,0,{date},00:00:{second:02d}"
            for index, second in enumerate(seconds)]
    (trajectory / "20080101000000.plt").write_text("\n".join(headers + rows) + "\n", encoding="utf-8")
    labels = ["Start Time\tEnd Time\tTransportation Mode"]
    labels.extend(f"{start}\t{end}\t{mode}" for start, end, mode in intervals)
    (directory / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")


class PrepareGeoLifeTest(unittest.TestCase):
    def test_taxi_exclusion_mismatched_user_and_new_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "private-location-name" / "Data"
            make_user(raw, "001", [("2008-01-01 00:00:00", "2008-01-01 00:00:05", "taxi"),
                                   ("2008-01-01 00:00:10", "2008-01-01 00:00:15", "train")])
            make_user(raw, "002", [("2007-01-01 00:00:00", "2007-01-01 00:00:15", "walk")])
            make_user(raw, "003", [("2008-01-01 00:00:00", "2008-01-01 00:00:05", "walk")], seconds=(0, 5))
            output = root / "new" / "processed"
            self.assertFalse(output.exists())
            record = prepare.prepare_points(raw, output)
            points = pd.read_csv(output / "points.csv", dtype={"user_id": str, "traj_id": str})
            self.assertEqual(len(points), 6)
            self.assertEqual(set(points["user_id"]), {"001", "003"})
            self.assertEqual(points["label"].eq("car").sum(), 2)
            self.assertEqual(points["label"].eq("walk").sum(), 2)
            self.assertEqual(points["label"].isna().sum(), 2)
            self.assertFalse(points["label"].isin(["taxi", "train"]).any())
            self.assertTrue(np.isfinite(points[["x", "y"]].to_numpy()).all())
            self.assertEqual(record["counts"]["n_users"], 2)
            self.assertEqual(record["counts"]["n_trajectories"], 2)
            self.assertEqual(record["counts"]["n_excluded_or_unlabelled_points"], 2)
            self.assertEqual(record["output"]["sha256"], hashlib.sha256((output / "points.csv").read_bytes()).hexdigest())
            metadata = (output / "data_preparation.json").read_text(encoding="utf-8")
            self.assertEqual(json.loads(metadata), record)
            self.assertNotIn("private-location-name", metadata)
            self.assertTrue(record["not_an_experiment_summary"])
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["data_preparation.json", "points.csv"])

    def test_refuses_overwrite_before_loading_and_allows_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_user(root / "Data", "001", [("2008-01-01 00:00:00", "2008-01-01 00:00:15", "bike")])
            output = root / "out"
            output.mkdir()
            (output / "points.csv").write_bytes(b"existing data\n")
            with patch.object(prepare.loader, "load_raw_geolife") as load:
                with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                    prepare.prepare_points(root / "Data", output)
                load.assert_not_called()
            self.assertEqual((output / "points.csv").read_bytes(), b"existing data\n")
            record = prepare.prepare_points(root / "Data", output, overwrite=True)
            self.assertEqual(record["counts"]["class_point_counts"]["bike"], 4)
            self.assertNotEqual((output / "points.csv").read_bytes(), b"existing data\n")

    def test_missing_empty_misaligned_or_only_excluded_data_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                prepare.prepare_points(root / "absent", root / "out")
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "official Data directory"):
                prepare.prepare_points(empty, root / "out")
            make_user(root / "misaligned", "001", [("2007-01-01 00:00:00", "2007-01-01 00:00:15", "walk")])
            with self.assertRaises(FileNotFoundError):
                prepare.prepare_points(root / "misaligned", root / "out")
            make_user(root / "excluded", "001", [("2008-01-01 00:00:00", "2008-01-01 00:00:15", "airplane")])
            with self.assertRaisesRegex(ValueError, "supported five-class label"):
                prepare.prepare_points(root / "excluded", root / "out")
            self.assertFalse((root / "out" / "points.csv").exists())

    def test_cli_default_directory_and_failure_exit_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_user(root / "Data", "001", [("2008-01-01 00:00:00", "2008-01-01 00:00:15", "bus")])
            with patch.object(prepare.config, "DATA_PROCESSED", root / "default-output"), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prepare.main(["--raw-root", str(root / "Data")]), 0)
            self.assertTrue((root / "default-output" / "points.csv").is_file())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(prepare.main(["--raw-root", str(root / "absent"), "--output-dir", str(root / "other")]), 2)


if __name__ == "__main__":
    unittest.main()
