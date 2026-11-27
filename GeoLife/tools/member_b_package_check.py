"""Verify the overlaid member-B handoff bundles, optionally smoke-test RF.

At the extracted GeoLife_MemberB_Handoff root:
    python tools/member_b_package_check.py
    python tools/member_b_package_check.py --mode all --smoke

Core verification uses only the standard library. The optional smoke test uses
NumPy, pandas, scikit-learn 1.8.0, PyYAML, joblib and threadpoolctl, without Torch.
The smoke test compares a small common-support batch against the original
held-out predictions in all four representative conditions; it does not run PI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

from member_b_bootstrap import bootstrap_research


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
FOUR_CONDITIONS = ["ds0_n0", "ds10_n0", "ds0_n20", "ds10_n20"]


class PackageError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise PackageError(f"Invalid package-relative path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise PackageError(f"Path leaves the package root: {value!r}")
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise PackageError(f"Path resolves outside the package root: {value!r}") from error
    return path


def load_manifest(root: Path) -> dict:
    path = root / "PACKAGE_MANIFEST.json"
    if not path.is_file():
        raise PackageError(f"Missing {path}; run from the extracted package root or pass --root")
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("run_sha256", ""))):
        raise PackageError("Unsupported manifest version or invalid run_sha256")
    if not isinstance(manifest.get("expected"), dict) or not isinstance(manifest.get("files"), list):
        raise PackageError("Manifest requires expected counts and a files list")
    for name in ["n_reference", "n_common", "n_common_users", "n_folds", "n_packaged_rf_models"]:
        value = manifest["expected"].get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise PackageError(f"Invalid expected.{name}")
    names = []
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise PackageError("Every manifest file entry must be a mapping")
        name = entry.get("path")
        package_path(root, name)
        names.append(name)
        if entry.get("part") not in ("core", "models"):
            raise PackageError(f"Unknown package part for {name}")
        size = entry.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise PackageError(f"Invalid file size for {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise PackageError(f"Invalid SHA256 for {name}")
    if len(names) != len(set(names)):
        raise PackageError("Duplicate paths in PACKAGE_MANIFEST.json")
    packaged_rf = sum(entry["part"] == "models" and PurePosixPath(entry["path"]).name == "rf.joblib"
                      for entry in manifest["files"])
    if packaged_rf != manifest["expected"]["n_packaged_rf_models"]:
        raise PackageError("The RF model inventory count differs from expected.n_packaged_rf_models")
    return manifest


def verify_files(root: Path, entries: list[dict], label: str) -> dict:
    failures = []
    total_bytes = 0
    for index, entry in enumerate(entries, 1):
        path = package_path(root, entry["path"])
        if not path.is_file():
            failures.append(f"MISSING [{entry['part']}] {entry['path']}")
            continue
        actual_size = path.stat().st_size
        if actual_size != entry["size_bytes"]:
            failures.append(f"SIZE [{entry['part']}] {entry['path']}: {actual_size} != {entry['size_bytes']}")
            continue
        if sha256_file(path) != entry["sha256"]:
            failures.append(f"SHA256 [{entry['part']}] {entry['path']}")
            continue
        total_bytes += actual_size
        if index % 100 == 0:
            print(f"Verified {label}: {index}/{len(entries)} files", flush=True)
    if failures:
        raise PackageError("File verification failed:\n" + "\n".join(failures))
    return {"files": len(entries), "bytes": total_bytes}


def smoke_test(root: Path, manifest: dict, user: str, batch_size: int) -> dict:
    # Set before numerical imports; also constrain already-loaded native pools.
    for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[variable] = "1"
    torch_was_loaded = "torch" in sys.modules
    try:
        import joblib
        import numpy as np
        import pandas as pd
        import sklearn
        import yaml
        from threadpoolctl import threadpool_limits
    except ImportError as error:
        raise PackageError("--smoke needs the member-B numerical requirements (no Torch required): " + str(error)) from error
    if sklearn.__version__ != "1.8.0":
        raise PackageError(f"--smoke requires scikit-learn==1.8.0; found {sklearn.__version__}")
    bootstrap_research(root)
    from research import features
    from research.deployment_grid import Windows
    from research.deployment_protocol import load_config, iter_conditions, degrade_window, window_quality

    entries = {entry["path"]: entry for entry in manifest["files"]}
    run_id = manifest["run_sha256"]
    cfg = load_config(root / "现实关联" / "degradation_config.yaml")
    if cfg["seeds"]["noise_primary"] != 0:
        raise PackageError("The handoff smoke test expects the frozen primary noise seed 0")
    data = Windows(cfg)
    actual = {"n_reference": len(data.meta), "n_common": int(data.support["common"].sum()),
              "n_common_users": int(data.meta.loc[data.support["common"], "user_id"].nunique()),
              "n_folds": int(data.meta["user_id"].nunique())}
    for key, value in actual.items():
        if value != manifest["expected"][key]:
            raise PackageError(f"Actual {key}={value} differs from manifest {manifest['expected'][key]}")
    user = user.zfill(3)
    selected = np.flatnonzero(data.meta["user_id"].eq(user).to_numpy() & data.support["common"])[:batch_size]
    if not len(selected):
        raise PackageError(f"User {user} has no common-support windows; choose one of the 48 common users")
    fold = f"project/artifacts/member_a/run_{run_id[:16]}/folds/{user}"
    rf_relative = f"{fold}/rf.joblib"
    model_path = package_path(root, rf_relative)
    if rf_relative not in entries or not model_path.is_file():
        raise PackageError(f"--smoke requires the RF model from the models bundle: {rf_relative}\n"
                           "Extract the models ZIP into the same package root, then rerun --mode all --smoke.")
    model_entry = entries[rf_relative]
    # Core mode does not check all model files; it still verifies the model used
    # by smoke BEFORE loading its pickle. Only this supplied handoff is trusted.
    verify_files(root, [model_entry], "smoke model")
    metadata_relative = f"{fold}/models.json"
    if metadata_relative not in entries:
        raise PackageError(f"Model metadata is absent from the manifest: {metadata_relative}")
    metadata = json.loads(package_path(root, metadata_relative).read_text(encoding="utf-8"))
    identity = metadata["identity"]
    if identity["run_identity"] != run_id or identity["test_user"] != user or user in identity["training_users"]:
        raise PackageError("The chosen RF model is not the held-out-user model for this run")
    if model_entry["sha256"] != metadata["rf_sha256"]:
        raise PackageError("Package/model-metadata RF hashes differ")
    window_ids = data.meta.iloc[selected]["window_id"].to_numpy(dtype=int)
    conditions = {condition.name: condition for condition in iter_conditions(cfg, include_stress=False)}
    reports = []
    with threadpool_limits(limits=1):
        model = joblib.load(model_path)
        model.n_jobs = 1
        for name in FOUR_CONDITIONS:
            prediction_relative = f"{fold}/{name}_seed0.csv.gz"
            if prediction_relative not in entries:
                raise PackageError(f"Saved predictions absent from the manifest: {prediction_relative}")
            saved = pd.read_csv(package_path(root, prediction_relative), dtype={"user_id": str})
            if saved["window_id"].duplicated().any():
                raise PackageError(f"Duplicate prediction window IDs: {prediction_relative}")
            expected = saved.set_index("window_id").loc[window_ids]
            if (not expected["survive"].eq(True).all() or not expected["common"].eq(True).all()
                    or not expected["user_id"].eq(user).all() or not expected["condition"].eq(name).all()
                    or not expected["noise_seed"].eq(0).all()):
                raise PackageError(f"Saved prediction identity/support mismatch: {name}")
            windows = [degrade_window(data.frame(int(index)), conditions[name], noise_seed=0,
                                      window_id=int(window_id)) for index, window_id in zip(selected, window_ids)]
            if not all(window_quality(window, cfg)["usable"] for window in windows):
                raise PackageError(f"Unexpected unusable smoke windows: {name}")
            matrix = features.build_core_a_matrix(windows).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
            actual_prediction = model.predict(matrix)
            if not np.array_equal(actual_prediction, expected["pred_rf"].to_numpy(dtype=int)):
                mismatches = window_ids[actual_prediction != expected["pred_rf"].to_numpy(dtype=int)].tolist()
                raise PackageError(f"RF smoke predictions differ at {name}, window IDs {mismatches}")
            reports.append({"condition": name, "n_windows": len(selected), "exact_rf_prediction_match": True})
            print(f"RF smoke PASS: user={user}, condition={name}, windows={len(selected)}, n_jobs=1", flush=True)
    if not torch_was_loaded and "torch" in sys.modules:
        raise PackageError("Unexpected Torch import: the member-B smoke path must remain Torch-free")
    return {"actual_counts": actual, "test_user": user, "n_windows_per_condition": len(selected),
            "conditions": reports, "torch_imported_by_smoke": not torch_was_loaded and "torch" in sys.modules,
            "environment": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                            "PyYAML": yaml.__version__, "joblib": joblib.__version__}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mode", choices=["core", "all"], default="core")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--user", default="010", help="Held-out user for smoke; must have common-support windows")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 1024:
        parser.error("--batch-size must be in 1..1024")
    root = args.root.resolve()
    try:
        manifest = load_manifest(root)
        required = [entry for entry in manifest["files"] if args.mode == "all" or entry["part"] == "core"]
        checked = verify_files(root, required, args.mode)
        expected = manifest["expected"]
        print("Manifest counts: " + ", ".join(f"{key}={value}" for key, value in expected.items()), flush=True)
        print(f"{args.mode} file check PASS: {checked['files']} files, {checked['bytes']} bytes", flush=True)
        missing_models = [entry["path"] for entry in manifest["files"]
                          if entry["part"] == "models" and not package_path(root, entry["path"]).is_file()]
        if args.mode == "core" and missing_models:
            print(f"Models bundle missing ({len(missing_models)} paths; not required for the core-only file check):", flush=True)
            for path in missing_models:
                print(f"  {path}", flush=True)
        result = {"status": "passed", "mode": args.mode, "run_sha256": manifest["run_sha256"],
                  "verified": checked, "manifest_counts": expected, "missing_model_paths": missing_models}
        if args.smoke:
            result["smoke"] = smoke_test(root, manifest, args.user, args.batch_size)
        print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
        return 0
    except (PackageError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"PACKAGE CHECK FAILED: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
