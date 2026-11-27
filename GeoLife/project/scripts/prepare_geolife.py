"""Convert the official GeoLife Data directory into the frozen points.csv input.

From GeoLife/:
    python project/scripts/prepare_geolife.py --raw-root "/path/to/Geolife Trajectories 1.3/Data"

This prepares local point data only. Window construction, coverage and model
evaluation belong to the separate deployment-grid entry point. Unsupported labels
remain missing in the point table, as in the frozen loader; segmentation later
excludes them while preserving same-label-run boundaries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
import time


GEOLIFE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GEOLIFE_ROOT / "tools"))
from member_b_bootstrap import bootstrap_research

bootstrap_research(GEOLIFE_ROOT)
import numpy as np
import pandas as pd
from research import config, loader


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_points(raw_root: str | Path, output_dir: str | Path | None = None,
                   overwrite: bool = False) -> dict:
    """Write points.csv and a local preparation record; return that record.

    A complete CSV is written and hashed in the output directory before replace,
    so readers never observe a partially written points.csv. Metadata is prepared
    before replacement and records the final CSV hash. It is not an A/B experiment
    summary and does not claim to reproduce any previously reported score.
    """
    raw_root = Path(raw_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else config.DATA_PROCESSED
    points_path = output_dir / "points.csv"
    metadata_path = output_dir / "data_preparation.json"
    if points_path.exists() and not overwrite:
        raise FileExistsError(f"{points_path} already exists; pass --overwrite to replace it")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw GeoLife directory not found: {raw_root}")
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()
    try:
        points = loader.load_raw_geolife(raw_root)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "No usable labelled trajectory points were found. --raw-root must name the official Data directory "
            "containing user folders such as 000/Trajectory/*.plt and 000/labels.txt; label times must overlap the points."
        ) from error
    if points.empty or not points["label"].isin(config.CLASSES).any():
        raise ValueError("No points with a supported five-class label matched the raw trajectories")
    # Preserve all rows returned by the frozen loader, including missing labels.
    # Dropping them here could join otherwise separated same-label runs.
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_csv = temporary_json = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output_dir,
                                         prefix=".points.", suffix=".csv.tmp", delete=False) as handle:
            temporary_csv = Path(handle.name)
            points.to_csv(handle, index=False)
        missing_count = int(points["label"].isna().sum())
        class_counts = {label: int(points["label"].eq(label).sum()) for label in config.CLASSES}
        record = {
            "schema_version": 1,
            "artifact_type": "local_raw_points_preparation",
            "not_an_experiment_summary": True,
            "input": {
                "raw_root_name": raw_root.name,
                "format": "Official GeoLife Data/<user>/Trajectory/*.plt plus tab-separated labels.txt",
                "absolute_input_path_recorded": False,
            },
            "output": {"path": "points.csv", "sha256": sha256_file(temporary_csv),
                       "size_bytes": temporary_csv.stat().st_size},
            "counts": {
                "n_points": int(len(points)), "n_users": int(points["user_id"].nunique()),
                "n_trajectories": int(points.groupby(["user_id", "traj_id"]).ngroups),
                "n_supported_label_points": int(sum(class_counts.values())),
                "n_excluded_or_unlabelled_points": missing_count,
                "class_point_counts": class_counts,
            },
            "label_handling": "Taxi merges into car; unsupported labels become missing. Missing-label rows are retained until labelled-window segmentation.",
            "implementation_sha256": {name: sha256_file(GEOLIFE_ROOT / "project" / "research" / name)
                                      for name in ["loader.py", "labels.py", "geo.py", "config.py"]},
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "preparation_seconds": time.perf_counter() - started_clock,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__, "pandas": pd.__version__, "torch_required": False},
        }
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output_dir,
                                         prefix=".data_preparation.", suffix=".json.tmp", delete=False) as handle:
            temporary_json = Path(handle.name)
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        # Check again after loading: never silently overwrite a file that appeared
        # while a large raw dataset was being parsed.
        if points_path.exists() and not overwrite:
            raise FileExistsError(f"{points_path} appeared during preparation; pass --overwrite to replace it")
        temporary_csv.replace(points_path)
        temporary_csv = None
        temporary_json.replace(metadata_path)
        temporary_json = None
        return record
    finally:
        for path in [temporary_csv, temporary_json]:
            if path is not None:
                path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path,
                        help="Official GeoLife Data directory containing per-user Trajectory/ and labels.txt")
    parser.add_argument("--output-dir", type=Path, default=config.DATA_PROCESSED,
                        help="Destination directory; defaults to project/data/processed")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing points.csv")
    args = parser.parse_args(argv)
    try:
        record = prepare_points(args.raw_root, args.output_dir, args.overwrite)
    except (OSError, ValueError, pd.errors.ParserError, KeyError) as error:
        # Escaped output remains usable with Windows consoles using a legacy
        # encoding; the UTF-8 metadata itself preserves human-readable text.
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "prepared", "output": record["output"], "counts": record["counts"],
                      "metadata": "data_preparation.json", "not_an_experiment_summary": True},
                     ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
