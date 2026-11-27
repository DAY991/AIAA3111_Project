"""Independently verify member A's complete archived experiment, without models.

This tool hashes model bytes; it never unpickles, imports torch, retrains, or
changes the frozen research implementation or existing experimental tables.
It verifies every prediction receipt and re-scores all OOF predictions using
scikit-learn. Outputs are member_a_verification.json and a 54-fold inventory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "project" / "artifacts" / "member_a"
OUTPUTS = ROOT / "现实关联" / "outputs"
DEFAULT_RUN = "run_04d2f3dee1b64ac6"
TZ = timezone(timedelta(hours=8))


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def object_hash(value, compact=False):
    options = {"sort_keys": True, "ensure_ascii": False, "default": str}
    if compact:
        options.update(separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(json.dumps(value, **options).encode("utf-8")).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def conditions(cfg, repeats=True):
    deg, seeds = cfg["degradation"], cfg["seeds"]
    result = []
    for stress, steps in [(False, deg["main_downsample_sec"]), (True, deg["stress_downsample_sec"])]:
        for step in steps:
            for sigma in deg["noise_sigma_m"]:
                name = f"ds{step}_n{sigma:g}"
                noise_seeds = (seeds["noise_repeats"] if repeats and sigma > 0 and not stress
                               and name in seeds["repeat_conditions"] else [seeds["noise_primary"]])
                result.extend({"condition": name, "downsample_sec": step, "noise_sigma_m": sigma,
                               "noise_seed": seed, "is_stress": stress} for seed in noise_seeds)
    return result


class Verifier:
    def __init__(self, run_name, output_dir, atol, input_dir=OUTPUTS):
        self.started = time.perf_counter()
        self.run_dir = ARTIFACTS / run_name
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.atol = atol
        self.hash_count = 0
        self.hash_bytes = 0
        self.checks = []
        self.table_comparisons = {}
        self.inventory = []
        self.proof = {}
        self.prediction_file_proofs = []

    def sha256(self, path, expected=None):
        path = Path(path)
        require(path.is_file(), f"Missing file: {path}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        result = digest.hexdigest()
        self.hash_count += 1
        self.hash_bytes += size
        if expected is not None:
            require(result == expected, f"SHA256 mismatch: {path}; expected {expected}, got {result}")
        return result

    def compare_table(self, name, expected, path, keys, dtype=None):
        actual = pd.read_csv(path, dtype=dtype)
        require(set(actual.columns) == set(expected.columns), f"{name}: unexpected columns")
        require(len(actual) == len(expected), f"{name}: row count {len(actual)} != {len(expected)}")
        require(not actual.duplicated(keys).any(), f"{name}: duplicate keys {keys}")
        expected = expected.sort_values(keys).reset_index(drop=True)
        actual = actual.sort_values(keys).reset_index(drop=True)
        errors = {}
        for column in expected:
            if pd.api.types.is_numeric_dtype(expected[column]):
                a, b = actual[column].to_numpy(dtype=float), expected[column].to_numpy(dtype=float)
                require(np.allclose(a, b, rtol=0, atol=self.atol, equal_nan=True),
                        f"{name}: values differ in {column}")
                finite = np.isfinite(a) & np.isfinite(b)
                errors[column] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            else:
                require(actual[column].tolist() == expected[column].tolist(),
                        f"{name}: values differ in {column}")
        self.table_comparisons[name] = {
            "path": relative(path), "sha256": self.sha256(path), "rows": len(actual),
            "max_absolute_numeric_error": max(errors.values(), default=0.0),
            "absolute_tolerance": self.atol,
        }

    def verify_identity_and_inputs(self):
        manifest_path = self.run_dir / "run_manifest.json"
        run = read_json(manifest_path)
        cfg_path = ROOT / "现实关联" / "degradation_config.yaml"
        with cfg_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        require(run["config"] == cfg, "Current YAML differs from archived run configuration")
        config_hash = object_hash(cfg, compact=True)
        require(run["config_sha256"] == config_hash, "Run config hash mismatch")
        payload = {key: run[key] for key in ["preparation_fingerprint", "config_sha256", "environment"]}
        self.run_hash = object_hash(payload)
        require(self.run_hash == run["run_sha256"], "Run identity is not reproducible")
        require(self.run_dir.name == "run_" + self.run_hash[:16], "Run directory does not match identity")
        require(cfg["classes"] == ["walk", "bike", "bus", "car", "subway"], "Unexpected class order")
        require(cfg["split"] == {"method": "leave_one_user_out", "group_column": "user_id"}, "Not user-disjoint LOUO")
        require(cfg["segmentation"] == {"window_sec": 120, "min_points": 10, "max_gap_sec": 30}, "Changed window protocol")
        preparation = read_json(ARTIFACTS / "prepare_metadata.json")
        require(preparation["config_sha256"] == config_hash, "Preparation config mismatch")
        identity = {key: preparation[key] for key in ["config_sha256", "points_sha256", "implementation_sha256"]}
        require(object_hash(identity) == preparation["fingerprint"] == run["preparation_fingerprint"], "Preparation identity mismatch")
        source_hashes = {}
        for name, digest in preparation["implementation_sha256"].items():
            source_hashes[name] = self.sha256(ROOT / "project" / "research" / name, digest)
        require(set(source_hashes) == {"deployment_grid.py", "deployment_protocol.py", "features.py", "model_seq.py",
                                       "degradation.py", "segmentation.py", "config.py", "model.py"}, "Missing frozen source hashes")
        paths = {
            "points": (ROOT / cfg["data"]["points_csv"], "points_sha256"),
            "reference_manifest": (ROOT / cfg["data"]["reference_manifest"], "manifest_sha256"),
            "packed_windows": (ARTIFACTS / "windows.npz", "windows_sha256"),
            "support": (ARTIFACTS / "support.npz", "support_sha256"),
        }
        inputs = {key: {"path": relative(path), "sha256": self.sha256(path, preparation[field])}
                  for key, (path, field) in paths.items()}
        require(read_json(self.input_dir / "prepare_summary.json") == preparation, "Published preparation summary differs from cache metadata")
        self.cfg, self.preparation = cfg, preparation
        self.meta = pd.read_csv(paths["reference_manifest"][0], dtype={"user_id": str, "traj_id": str})
        meta = self.meta
        require(len(meta) == preparation["n_reference"] == 90120, "Reference set is not the expected 90,120 windows")
        require(meta["window_id"].is_unique and meta["window_id"].notna().all(), "Reference IDs are not unique")
        require(meta["usable"].eq(True).all(), "Reference includes clean-ineligible windows")
        require(set(meta["label"]) == set(cfg["classes"]), "Reference contains unexpected labels")
        self.users = sorted(meta["user_id"].unique())
        require(len(self.users) == preparation["n_users"] == 54, "Expected 54 users")
        self.n = len(meta)
        self.y = meta["label"].map({label: i for i, label in enumerate(cfg["classes"])}).to_numpy(dtype=np.int64)
        self.proof.update({"run_manifest_sha256": self.sha256(manifest_path), "run_environment": run["environment"],
                           "config_sha256": config_hash, "config_yaml_file_sha256": self.sha256(cfg_path),
                           "preparation_fingerprint": preparation["fingerprint"], "input_files": inputs,
                           "frozen_source_sha256": source_hashes})
        self.checks.append("run_config_source_points_and_preparation_hashes")

    def verify_support_and_features(self):
        with np.load(ARTIFACTS / "support.npz", allow_pickle=False) as packed:
            supports = {name: packed[name] for name in packed.files}
        with np.load(ARTIFACTS / "windows.npz", allow_pickle=False) as packed:
            stamps, xy, offsets = packed["time_ns"], packed["xy"], packed["offsets"]
        require(offsets.shape == (self.n + 1,) and offsets[0] == 0 and offsets[-1] == len(stamps), "Invalid packed offsets")
        require(np.array_equal(np.diff(offsets), self.meta["n_points"]), "Packed window counts differ from manifest")
        require(xy.shape == (len(stamps), 2) and np.isfinite(xy).all(), "Invalid packed coordinates")
        starts = pd.to_datetime(self.meta["start_time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        ends = pd.to_datetime(self.meta["end_time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        require(np.all(stamps[offsets[:-1]] >= starts) and np.all(stamps[offsets[1:] - 1] < ends), "Packed points leave their declared windows")
        for step in self.cfg["degradation"]["main_downsample_sec"] + self.cfg["degradation"]["stress_downsample_sec"]:
            key = f"ds{step}"
            require(supports[key].shape == (self.n,) and supports[key].dtype == np.bool_, f"Malformed support array {key}")
            usable = np.empty(self.n, dtype=bool)
            counts = np.empty(self.n, dtype=np.int32)
            for i, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
                t = stamps[begin:end]
                require(np.all(np.diff(t) >= 0), f"Unsorted timestamps in reference window {i}")
                if step:
                    buckets = (t - t[0]) // (step * 1_000_000_000)
                    t = t[np.r_[True, buckets[1:] != buckets[:-1]]]
                counts[i] = len(t)
                gap = float(np.diff(t).max()) / 1_000_000_000 if len(t) > 1 else 0.0
                usable[i] = len(t) >= 10 and gap <= 30
            require(np.array_equal(counts, supports[f"counts_{key}"]), f"Retained-point counts differ for {key}")
            require(np.array_equal(usable, supports[key]), f"Recomputed timestamp eligibility differs for {key}")
        common = np.logical_and.reduce([supports[f"ds{s}"] for s in self.cfg["degradation"]["main_downsample_sec"]])
        require(np.array_equal(common, supports["common"]), "Common support is not the main-grid intersection")
        require(int(common.sum()) == 71941, "Expected 71,941 common windows")
        require(supports["ds0"].all() and not supports["ds30"].any(), "Clean/stress coverage boundary mismatch")
        self.supports = supports
        folder = ARTIFACTS / ("features_" + self.preparation["fingerprint"][:16])
        info = read_json(folder / "metadata.json")
        require(info["fingerprint"] == self.preparation["fingerprint"], "Feature cache fingerprint mismatch")
        require(set(info["files"]) == {"core_a.npy", "core_b.npy", "mask.npy"}, "Feature file manifest mismatch")
        feature_proofs = {}
        for name, digest in info["files"].items():
            feature_proofs[name] = self.sha256(folder / name, digest)
            array = np.load(folder / name, mmap_mode="r", allow_pickle=False)
            shape = {"core_a.npy": (self.n, 17), "core_b.npy": (self.n, 128, 5), "mask.npy": (self.n, 128)}[name]
            require(array.shape == shape and array.dtype == np.float32, f"Unexpected feature layout: {name}")
            require(np.isfinite(array).all(), f"Nonfinite feature cache: {name}")
            if name == "mask.npy":
                expected_mask = np.arange(128)[None, :] < np.minimum(self.meta["n_points"].to_numpy(), 128)[:, None]
                require(np.array_equal(array, expected_mask), "Sequence padding mask differs from original point count")
        self.proof["feature_file_sha256"] = feature_proofs
        self.proof["feature_metadata_sha256"] = self.sha256(folder / "metadata.json")
        self.proof["coverage_by_step"] = {f"ds{s}": int(supports[f"ds{s}"].sum()) for s in [0, 5, 10, 30]}
        self.proof["common_windows"] = int(common.sum())
        self.checks.append("packed_timestamp_eligibility_common_support_and_feature_hashes")

    def verify_folds_and_predictions(self):
        fold_manifest = read_json(ARTIFACTS / "fold_manifest.json")
        require(fold_manifest["method"] == "leave_one_user_out" and fold_manifest["reference_fingerprint"] == self.preparation["fingerprint"], "Fold manifest identity mismatch")
        entries = fold_manifest["folds"]
        require(len(entries) == 54 and sorted(e["test_user"] for e in entries) == self.users, "Fold manifest user mismatch")
        by_user = {e["test_user"]: e for e in entries}
        actual_users = sorted(p.name for p in (self.run_dir / "folds").iterdir() if p.is_dir())
        require(actual_users == self.users, "Missing or extra fold directories")
        self.jobs = conditions(self.cfg)
        require(len(self.jobs) == 18, "Expected 18 condition/seed jobs, including stress boundary")
        oof = {(c["condition"], c["noise_seed"]): np.full((self.n, 2), -999, dtype=np.int64) for c in self.jobs}
        seen = {key: np.zeros(self.n, dtype=bool) for key in oof}
        total_rows = 0
        for count, user in enumerate(self.users, 1):
            test = self.meta["user_id"].to_numpy() == user
            positions = np.flatnonzero(test)
            training_users = [u for u in self.users if u != user]
            require(by_user[user] == {"test_user": user, "training_users": training_users, "n_test": len(positions)}, f"Invalid fold manifest for {user}")
            expected_identity = {"run_identity": self.run_hash, "test_user": user, "training_users": training_users,
                                 "train_window_ids_sha256": object_hash(self.meta.loc[~test, "window_id"].tolist()),
                                 "test_window_ids_sha256": object_hash(self.meta.loc[test, "window_id"].tolist())}
            folder = self.run_dir / "folds" / user
            metadata_path = folder / "models.json"
            metadata = read_json(metadata_path)
            require(metadata["identity"] == expected_identity, f"Train/test user or window-ID mismatch in fold {user}")
            require(len(training_users) == 53 and user not in training_users, f"User leakage in fold {user}")
            rf_path, gru_path = folder / "rf.joblib", folder / "gru.pt"
            rf_hash = self.sha256(rf_path, metadata["rf_sha256"])
            gru_hash = self.sha256(gru_path, metadata["gru_sha256"])
            expected_files = {f"{c['condition']}_seed{c['noise_seed']}.csv.gz" for c in self.jobs}
            require({p.name for p in folder.glob("*.csv.gz")} == expected_files, f"Prediction files differ for fold {user}")
            require({p.name for p in folder.glob("*.csv.gz.json")} == {name + ".json" for name in expected_files}, f"Prediction receipts differ for fold {user}")
            expected_frame = self.meta.iloc[positions][["window_id", "user_id", "label"]].reset_index(drop=True)
            for c in self.jobs:
                key = c["condition"], c["noise_seed"]
                path = folder / f"{c['condition']}_seed{c['noise_seed']}.csv.gz"
                receipt_path = path.with_name(path.name + ".json")
                receipt = read_json(receipt_path)
                require(receipt["run_sha256"] == self.run_hash and receipt["rf_sha256"] == rf_hash
                        and receipt["gru_sha256"] == gru_hash, f"Receipt/model/run binding mismatch: {path}")
                digest = self.sha256(path, receipt["prediction_sha256"])
                frame = pd.read_csv(path, dtype={"window_id": np.int64, "user_id": str, "label": str,
                                                "noise_seed": np.int64, "pred_rf": np.int64, "pred_gru": np.int64})
                require(list(frame.columns) == ["window_id", "user_id", "label", "survive", "common", "condition", "noise_seed", "pred_rf", "pred_gru"], f"Unexpected prediction schema: {path}")
                require(len(frame) == len(positions), f"Prediction rows missing: {path}")
                for column in ["window_id", "user_id", "label"]:
                    require(np.array_equal(frame[column], expected_frame[column]), f"Prediction {column}/order mismatch: {path}")
                require(frame["condition"].eq(c["condition"]).all() and frame["noise_seed"].eq(c["noise_seed"]).all(), f"Prediction condition/seed mismatch: {path}")
                usable = self.supports[f"ds{c['downsample_sec']}"][positions]
                require(np.array_equal(frame["survive"], usable), f"Prediction survival mismatch: {path}")
                require(np.array_equal(frame["common"], self.supports["common"][positions]), f"Prediction common-support mismatch: {path}")
                pred = frame[["pred_rf", "pred_gru"]].to_numpy(dtype=np.int64)
                require(((pred[usable] >= 0) & (pred[usable] < 5)).all(), f"Invalid eligible prediction: {path}")
                require((pred[~usable] == -1).all(), f"Unusable windows have predictions: {path}")
                require(not seen[key][positions].any(), f"Duplicate OOF window IDs: {path}")
                oof[key][positions] = pred
                seen[key][positions] = True
                total_rows += len(frame)
                self.prediction_file_proofs.append({"path": relative(path), "sha256": digest,
                                                   "receipt_sha256": self.sha256(receipt_path)})
            self.inventory.append({"test_user": user, "run_sha256": self.run_hash,
                                   "training_user_count": len(training_users), "train_window_count": int((~test).sum()),
                                   "test_window_count": len(positions), "common_test_window_count": int(self.supports["common"][positions].sum()),
                                   "train_window_ids_sha256": expected_identity["train_window_ids_sha256"],
                                   "test_window_ids_sha256": expected_identity["test_window_ids_sha256"],
                                   "rf_path": relative(rf_path), "rf_sha256": rf_hash,
                                   "gru_path": relative(gru_path), "gru_sha256": gru_hash,
                                   "models_metadata_path": relative(metadata_path), "models_metadata_sha256": self.sha256(metadata_path),
                                   "rf_training_seconds": metadata["rf_training_seconds"], "gru_training_seconds": metadata["gru_training_seconds"],
                                   "prediction_receipts_verified": len(self.jobs)})
            print(f"Verified fold {count}/54 user={user}: model hashes + 18 prediction files/receipts", flush=True)
        for key, mask in seen.items():
            require(mask.all(), f"Incomplete OOF set: {key}")
        require(total_rows == self.n * len(self.jobs), "OOF total-row count mismatch")
        self.oof = oof
        self.proof.update({"fold_manifest_sha256": self.sha256(ARTIFACTS / "fold_manifest.json"),
                           "models_verified": 2 * len(self.users), "prediction_files_verified": len(self.prediction_file_proofs),
                           "prediction_receipts_verified": len(self.prediction_file_proofs), "oof_condition_seed_sets": len(oof),
                           "oof_rows_per_set": self.n, "prediction_rows_verified": total_rows,
                           "prediction_and_receipt_inventory_sha256": object_hash(self.prediction_file_proofs)})
        self.checks.append("108_model_hashes_972_prediction_hashes_receipt_bindings_and_all_oof_rows")

    def recompute_grid(self):
        rows = []
        for c in self.jobs:
            usable = self.supports[f"ds{c['downsample_sec']}"]
            pred = self.oof[c["condition"], c["noise_seed"]]
            for cohort in ["condition", "common"]:
                selected = usable & self.supports["common"] if cohort == "common" else usable
                n_scored = int(selected.sum())
                row = dict(c, cohort=cohort, scope="full_louo", n_reference=self.n, n_usable=int(usable.sum()),
                           retention=float(usable.mean()), n_scored=n_scored,
                           status="evaluated" if n_scored else "no_usable_windows", run_sha256=self.run_hash)
                for index, model in enumerate(["rf", "gru"]):
                    if n_scored:
                        per_class = f1_score(self.y[selected], pred[selected, index], labels=list(range(5)), average=None, zero_division=0)
                        row[f"{model}_macro_f1"] = float(per_class.mean())
                        row[f"{model}_accuracy"] = float(accuracy_score(self.y[selected], pred[selected, index]))
                    else:
                        per_class = np.full(5, np.nan)
                        row[f"{model}_macro_f1"], row[f"{model}_accuracy"] = np.nan, np.nan
                    row.update({f"{model}_f1_{label}": float(value) for label, value in zip(self.cfg["classes"], per_class)})
                row["gru_minus_rf"] = row["gru_macro_f1"] - row["rf_macro_f1"]
                row["gru_minus_rf_percentage_points"] = 100 * row["gru_minus_rf"]
                rows.append(row)
        self.grid = pd.DataFrame(rows)
        require(len(self.grid) == 36, "Expected 36 grid rows")
        for name, path in [("grid_results", self.input_dir / "grid_results.csv"), ("run_grid_results", self.run_dir / "grid_results.csv")]:
            self.compare_table(name, self.grid, path, ["condition", "noise_seed", "cohort"])
        summary = read_json(self.input_dir / "evaluation_summary.json")
        require(summary["run_sha256"] == self.run_hash and summary["scope"] == "full_louo"
                and summary["smoke_overrides"] is False and summary["all_conditions_complete"] is True,
                "Evaluation summary is incomplete, partial, or belongs to another run")
        require(summary["completed_users"] == summary["all_users"] == self.users and summary["n_reference"] == self.n,
                "Evaluation summary user/window counts mismatch")
        require(summary["seed_std_scope"] == "noise_realizations_only_fixed_models_and_users", "Seed uncertainty scope changed")
        self.proof["evaluation_summary_sha256"] = self.sha256(self.input_dir / "evaluation_summary.json")
        self.checks.append("36_grid_rows_independently_recomputed_with_fixed_five_class_metrics")

    def verify_seed_and_interaction_tables(self):
        # NumPy arithmetic deliberately does not reuse the producer's groupby.agg.
        metrics = [name for name in self.grid if name.endswith("macro_f1")
                   or name.startswith(("rf_f1_", "gru_f1_")) or name == "gru_minus_rf"]
        rows = []
        for (condition, cohort), group in self.grid.groupby(["condition", "cohort"], sort=False):
            row = {"condition": condition, "cohort": cohort, "scope": "full_louo"}
            for metric in metrics:
                values = group[metric].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                row[metric + "_mean"] = float(values.mean()) if len(values) else np.nan
                row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
                row[metric + "_count"] = len(values)
            rows.append(row)
        self.compare_table("grid_seed_summary", pd.DataFrame(rows), self.input_dir / "grid_seed_summary.csv", ["condition", "cohort", "scope"])
        common = self.grid[(self.grid["cohort"] == "common") & (self.grid["noise_seed"] == self.cfg["seeds"]["noise_primary"])]
        primary = {row["condition"]: row for row in common.to_dict("records")}
        rows = []
        for step in [5, 10]:
            for noise in [5, 10, 20]:
                condition = f"ds{step}_n{noise}"
                for model in ["RF", "GRU"]:
                    column = model.lower() + "_macro_f1"
                    joint, ds, ns, native = [primary[key][column] for key in [condition, f"ds{step}_n0", f"ds0_n{noise}", "ds0_n0"]]
                    effect = joint - ds - ns + native
                    rows.append({"condition": condition, "model": model, "downsample_sec": step, "noise_sigma_m": noise,
                                 "noise_seed": self.cfg["seeds"]["noise_primary"], "scope": "full_louo", "cohort": "common",
                                 "n_scored": 71941, "run_sha256": self.run_hash, "f_joint": joint,
                                 "f_downsample_only": ds, "f_noise_only": ns, "f_native": native,
                                 "interaction_f1": effect, "interaction_percentage_points": effect * 100, "status": "evaluated"})
        self.compare_table("interaction_contrasts", pd.DataFrame(rows), self.input_dir / "interaction_contrasts.csv", ["condition", "model"])
        self.checks.append("seed_means_sample_standard_deviations_counts_and_12_interaction_contrasts")

    def verify_coverage_tables(self):
        overall, classes, users = [], [], []
        for c in conditions(self.cfg, repeats=False):
            usable = self.supports[f"ds{c['downsample_sec']}"]
            row = {key: c[key] for key in ["condition", "is_stress", "downsample_sec", "noise_sigma_m"]}
            row.update(n_reference=self.n, n_usable=int(usable.sum()), retention=float(usable.mean()),
                       n_common=71941, status="coverage_only" if usable.any() else "no_usable_windows",
                       config_sha256=self.proof["config_sha256"])
            overall.append(row)
            for column, output in [("label", classes), ("user_id", users)]:
                for value, group in self.meta.groupby(column, sort=True):
                    positions = group.index.to_numpy()
                    output.append({"condition": c["condition"], column: value, "n_reference": len(positions),
                                   "n_usable": int(usable[positions].sum()), "retention": float(usable[positions].mean()),
                                   "n_common": int(self.supports["common"][positions].sum())})
        self.compare_table("retention_pilot", pd.DataFrame(overall), self.input_dir / "retention_pilot.csv", ["condition"])
        self.compare_table("retention_by_class", pd.DataFrame(classes), self.input_dir / "retention_by_class.csv", ["condition", "label"])
        self.compare_table("retention_by_user", pd.DataFrame(users), self.input_dir / "retention_by_user.csv", ["condition", "user_id"], dtype={"user_id": str})
        self.checks.append("overall_per_class_and_per_user_coverage_tables")

    def run(self):
        print("Verifying archived run identity, frozen source files and full input/cache hashes...", flush=True)
        self.verify_identity_and_inputs()
        self.verify_support_and_features()
        print("Inputs, feature caches and independently recomputed eligibility verified. Hashing 108 models...", flush=True)
        self.verify_folds_and_predictions()
        self.recompute_grid()
        self.verify_seed_and_interaction_tables()
        self.verify_coverage_tables()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        inventory_path = self.output_dir / "member_a_fold_inventory.csv"
        temporary = inventory_path.with_name(inventory_path.name + ".tmp")
        pd.DataFrame(self.inventory).to_csv(temporary, index=False)
        temporary.replace(inventory_path)
        inventory_hash = self.sha256(inventory_path)
        report = {"status": "passed", "verification_version": "1.0.0", "generated_at_local": datetime.now(TZ).isoformat(timespec="seconds"),
                  "run_sha256": self.run_hash, "run_directory": relative(self.run_dir),
                  "verifier_source_sha256": self.sha256(Path(__file__)),
                  "verification_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                               "pandas": pd.__version__, "sklearn": sklearn.__version__},
                  "checks_passed": self.checks, "evidence": self.proof, "table_comparisons": self.table_comparisons,
                  "fold_inventory": {"path": relative(inventory_path), "sha256": inventory_hash, "rows": len(self.inventory)},
                  "files_sha256_computed": self.hash_count, "bytes_hashed": self.hash_bytes,
                  "elapsed_seconds": round(time.perf_counter() - self.started, 3),
                  "scope": "Archived file integrity, split membership, every saved OOF prediction, and independent aggregate scoring; no model deserialization or new inference."}
        write_json(self.output_dir / "member_a_verification.json", report)
        print(f"PASS: 54 folds, 108 model hashes, 972 predictions/receipts; 36 grid rows independently verified in {report['elapsed_seconds']:.1f}s.", flush=True)
        return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="Run directory name; defaults to the completed evaluation_summary.json run")
    parser.add_argument("--input-dir", type=Path, default=OUTPUTS, help="Completed result directory to verify")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the input directory")
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    if args.run is None:
        summary = read_json(args.input_dir / "evaluation_summary.json")
        run_hash = summary.get("run_sha256", "")
        require(isinstance(run_hash, str) and len(run_hash) == 64
                and all(c in "0123456789abcdef" for c in run_hash), "Invalid completed run identity")
        args.run = "run_" + run_hash[:16]
    require(Path(args.run).name == args.run and args.run.startswith("run_"), "--run must be an archived run directory name")
    require(np.isfinite(args.atol) and 0 <= args.atol <= 1e-9, "Numeric tolerance must be between 0 and 1e-9")
    verifier = Verifier(args.run, args.output_dir or args.input_dir, args.atol, args.input_dir)
    try:
        verifier.run()
    except Exception as exc:
        write_json((args.output_dir or args.input_dir) / "member_a_verification.json", {
            "status": "failed", "generated_at_local": datetime.now(TZ).isoformat(timespec="seconds"),
            "run_directory": relative(verifier.run_dir), "error_type": type(exc).__name__, "error": str(exc),
            "checks_passed_before_failure": verifier.checks,
            "elapsed_seconds": round(time.perf_counter() - verifier.started, 3),
        })
        raise


if __name__ == "__main__":
    main()
