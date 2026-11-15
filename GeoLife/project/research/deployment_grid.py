"""Reproducible paired GeoLife deployment grid, separate from historical scripts.

Prepare freezes the original 120-second windows and measures coverage only.
Evaluate trains once per clean LOUO fold, saves both models, then reuses them
for every test-only condition. Large caches and individual predictions stay in
project/artifacts; only aggregate CSV files belong in the research output.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd

from . import features
from .deployment_protocol import (
    config_digest, degrade_window, iter_conditions, load_config,
    noise_seeds_for_condition, window_quality,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "project" / "artifacts" / "member_a"
OUTPUTS = ROOT / "现实关联" / "outputs"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.name.endswith(".gz") else None
    frame.to_csv(temp, index=False, compression=compression)
    temp.replace(path)


def _resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _implementation_digests() -> dict:
    base = Path(__file__).resolve().parent
    return {name: file_digest(base / name) for name in
            ["deployment_grid.py", "deployment_protocol.py", "features.py", "model_seq.py",
             "degradation.py", "segmentation.py", "config.py", "model.py"]}


def _prepare_environment() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "numpy": np.__version__, "pandas": pd.__version__, "cpu_count": os.cpu_count()}
    for package in ["torch", "scikit-learn", "PyYAML"]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _fast_reference(points: pd.DataFrame, cfg: dict):
    """Vectorized equivalent of segmentation.segment_windows, retaining its IDs.

    Each trajectory uses the same pandas datetime sort as the historical code
    (including its tie order). Run boundaries are formed BEFORE removing
    unlabelled points, so separated same-label runs are never accidentally joined.
    """
    points = points.reset_index(drop=True)
    timestamps = points["datetime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if points["datetime"].isna().any():
        raise ValueError("points.csv contains missing timestamps")
    order = np.concatenate([
        points.iloc[index].sort_values("datetime").index.to_numpy()
        for index in points.groupby(["user_id", "traj_id"], sort=True).indices.values()
    ])
    p = points.iloc[order].reset_index(drop=True)
    stamp = timestamps[order]
    user = p["user_id"].astype(str).to_numpy()
    traj = p["traj_id"].astype(str).to_numpy()
    labels = p["label"].fillna("__unlabelled__").astype(str).to_numpy()
    boundary = np.r_[True, (user[1:] != user[:-1]) | (traj[1:] != traj[:-1]) |
                     (labels[1:] != labels[:-1])]
    run = np.cumsum(boundary) - 1
    first = np.maximum.accumulate(np.where(boundary, np.arange(len(p)), 0))
    run_start = stamp[first]
    labelled = labels != "__unlabelled__"
    p = p.loc[labelled].reset_index(drop=True)
    stamp, user, traj, labels, run, run_start = [
        a[labelled] for a in (stamp, user, traj, labels, run, run_start)]
    span_ns = int(cfg["segmentation"]["window_sec"] * 1_000_000_000)
    bucket = (stamp - run_start) // span_ns
    boundary = np.r_[True, (run[1:] != run[:-1]) | (bucket[1:] != bucket[:-1])]
    starts = np.flatnonzero(boundary)
    counts = np.diff(np.r_[starts, len(p)])
    gap = np.r_[0, np.diff(stamp)] / 1_000_000_000.0
    gap[boundary] = 0.0
    max_gap = np.maximum.reduceat(gap, starts)
    start_time = (run_start[starts] + bucket[starts] * span_ns).astype("datetime64[ns]")
    meta = pd.DataFrame({
        "window_id": np.arange(len(starts)), "user_id": user[starts],
        "traj_id": traj[starts], "label": labels[starts],
        "start_time": start_time,
        "end_time": start_time + np.timedelta64(int(cfg["segmentation"]["window_sec"]), "s"),
        "n_points": counts, "max_gap": max_gap,
        "usable": (counts >= cfg["segmentation"]["min_points"]) &
                  (max_gap <= cfg["segmentation"]["max_gap_sec"]),
    })
    return p, meta, starts, counts


def _validate_reference(meta: pd.DataFrame, path: Path) -> bool:
    """Refuse silent window-ID changes against the historical local manifest."""
    if not path.exists():
        return False
    expected = pd.read_csv(path, dtype={"user_id": str, "traj_id": str},
                           parse_dates=["start_time", "end_time"])
    columns = ["window_id", "user_id", "traj_id", "label", "start_time", "end_time", "n_points", "usable"]
    pd.testing.assert_frame_equal(meta[columns], expected[columns], check_dtype=False)
    np.testing.assert_allclose(meta["max_gap"], expected["max_gap"], rtol=0, atol=1e-9)
    return True


def _coverage_for_step(stamp: np.ndarray, offsets: np.ndarray, step: int,
                       min_points: int, max_gap: float) -> tuple[np.ndarray, np.ndarray]:
    survives = np.empty(len(offsets) - 1, dtype=bool)
    counts = np.empty(len(offsets) - 1, dtype=np.int32)
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        t = stamp[start:end]
        if step:
            buckets = (t - t[0]) // (step * 1_000_000_000)
            t = t[np.r_[True, buckets[1:] != buckets[:-1]]]
        counts[i] = len(t)
        gap = np.diff(t).max() / 1_000_000_000 if len(t) > 1 else 0.0
        survives[i] = len(t) >= min_points and gap <= max_gap
    return survives, counts


def prepare(cfg: dict, output_dir: Path = OUTPUTS, force: bool = False) -> dict:
    """Freeze all clean-usable windows and run an exact full-data coverage pilot."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    source = _resolve(cfg["data"]["points_csv"])
    print(f"Hashing {source} ...", flush=True)
    identity = {"config_sha256": config_digest(cfg), "points_sha256": file_digest(source),
                "implementation_sha256": _implementation_digests()}
    fingerprint = _digest(identity)
    manifest_path = _resolve(cfg["data"]["reference_manifest"])
    prepare_json = ARTIFACTS / "prepare_metadata.json"
    if prepare_json.exists() and not force:
        cached = _read_json(prepare_json)
        if cached["fingerprint"] != fingerprint:
            raise ValueError("Preparation cache differs from source/config/code; rerun prepare with --force.")
        for key in ("manifest_sha256", "windows_sha256", "support_sha256"):
            target = {"manifest_sha256": manifest_path, "windows_sha256": ARTIFACTS / "windows.npz",
                      "support_sha256": ARTIFACTS / "support.npz"}[key]
            if file_digest(target) != cached[key]:
                raise ValueError(f"Preparation cache corrupted: {target}")
        print(f"Validated preparation cache: {cached['n_reference']:,} windows", flush=True)
        _write_coverage(cfg, output_dir, cached)
        return cached

    t0 = time.perf_counter()
    points = pd.read_csv(source, dtype={"user_id": str, "traj_id": str},
                         usecols=["user_id", "traj_id", "datetime", "x", "y", "label"],
                         parse_dates=["datetime"])
    p, all_meta, starts, counts = _fast_reference(points, cfg)
    verified = _validate_reference(all_meta, source.with_name("windows.csv"))
    valid = all_meta["usable"].to_numpy()
    meta = all_meta.loc[valid].copy().reset_index(drop=True)
    # A single packed array avoids caching 90,000 separate pandas objects.
    point_keep = np.repeat(valid, counts)
    packed = p.loc[point_keep]
    stamp = packed["datetime"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    xy = packed[["x", "y"]].to_numpy(dtype=np.float64)
    offsets = np.r_[0, np.cumsum(counts[valid])]
    _csv(meta, manifest_path)
    np.savez(ARTIFACTS / "windows.npz", time_ns=stamp, xy=xy, offsets=offsets)
    supports = {}
    for step in sorted(set(cfg["degradation"]["main_downsample_sec"] +
                           cfg["degradation"]["stress_downsample_sec"])):
        good, retained = _coverage_for_step(stamp, offsets, int(step),
                                           cfg["segmentation"]["min_points"],
                                           cfg["segmentation"]["max_gap_sec"])
        supports[f"ds{step}"] = good
        supports[f"counts_ds{step}"] = retained
        print(f"ds{step}: {good.sum():,}/{len(good):,} = {good.mean():.4%}", flush=True)
    supports["common"] = np.logical_and.reduce([
        supports[f"ds{s}"] for s in cfg["degradation"]["main_downsample_sec"]])
    np.savez(ARTIFACTS / "support.npz", **supports)
    users = sorted(meta["user_id"].unique())
    metadata = dict(identity, fingerprint=fingerprint, n_points=len(points),
                    n_all_windows=len(all_meta), n_reference=len(meta), n_users=len(users),
                    environment=_prepare_environment(),
                    historical_manifest_verified=verified, prepare_seconds=time.perf_counter() - t0,
                    manifest_sha256=file_digest(manifest_path),
                    windows_sha256=file_digest(ARTIFACTS / "windows.npz"),
                    support_sha256=file_digest(ARTIFACTS / "support.npz"))
    _json(ARTIFACTS / "fold_manifest.json", {
        "method": "leave_one_user_out", "reference_fingerprint": fingerprint,
        "folds": [{"test_user": user, "training_users": [u for u in users if u != user],
                   "n_test": int((meta["user_id"] == user).sum())} for user in users]})
    _json(prepare_json, metadata)
    _write_coverage(cfg, output_dir, metadata)
    return metadata


def _write_coverage(cfg, output_dir, metadata):
    meta = pd.read_csv(_resolve(cfg["data"]["reference_manifest"]), dtype={"user_id": str, "traj_id": str})
    with np.load(ARTIFACTS / "support.npz") as packed:
        supports = {key: packed[key] for key in packed.files}
    overall, by_user, by_class = [], [], []
    for c in iter_conditions(cfg):
        good = supports[f"ds{c.downsample_sec}"]
        overall.append({"condition": c.name, "is_stress": c.is_stress,
                        "downsample_sec": c.downsample_sec, "noise_sigma_m": c.noise_sigma_m,
                        "n_reference": len(meta), "n_usable": int(good.sum()),
                        "retention": float(good.mean()), "n_common": int(supports["common"].sum()),
                        "status": "no_usable_windows" if not good.any() else "coverage_only",
                        "config_sha256": metadata["config_sha256"]})
        for column, rows in [("user_id", by_user), ("label", by_class)]:
            for value, group in meta.groupby(column, sort=True):
                selected = good[group.index.to_numpy()]
                rows.append({"condition": c.name, column: value, "n_reference": len(group),
                             "n_usable": int(selected.sum()), "retention": float(selected.mean()),
                             "n_common": int(supports["common"][group.index.to_numpy()].sum())})
    _csv(pd.DataFrame(overall), output_dir / "retention_pilot.csv")
    _csv(pd.DataFrame(by_user), output_dir / "retention_by_user.csv")
    _csv(pd.DataFrame(by_class), output_dir / "retention_by_class.csv")
    _json(output_dir / "prepare_summary.json", metadata)


class Windows:
    def __init__(self, cfg):
        self.meta = pd.read_csv(_resolve(cfg["data"]["reference_manifest"]),
                                dtype={"user_id": str, "traj_id": str})
        with np.load(ARTIFACTS / "windows.npz") as packed:
            self.stamp, self.xy, self.offsets = packed["time_ns"], packed["xy"], packed["offsets"]
        with np.load(ARTIFACTS / "support.npz") as packed:
            self.support = {key: packed[key] for key in packed.files}

    def frame(self, index):
        start, end = self.offsets[index:index + 2]
        return pd.DataFrame({"datetime": self.stamp[start:end].astype("datetime64[ns]"),
                             "x": self.xy[start:end, 0], "y": self.xy[start:end, 1]})


def _extract(windows, seq_len):
    xa = features.build_core_a_matrix(windows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    xb, mask = features.build_core_b_sequences(windows, seq_len=seq_len)
    if not np.isfinite(xb).all():
        raise ValueError("Nonfinite sequence features; fix the source data before evaluating")
    return xa.to_numpy(dtype=np.float32), xb, mask


def _clean_features(data, cfg, fingerprint):
    folder = ARTIFACTS / f"features_{fingerprint[:16]}"
    paths = [folder / name for name in ["core_a.npy", "core_b.npy", "mask.npy"]]
    metadata_path = folder / "metadata.json"
    if metadata_path.exists():
        saved = _read_json(metadata_path)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Feature fingerprint mismatch")
        for path in paths:
            if file_digest(path) != saved["files"][path.name]:
                raise ValueError(f"Corrupted feature cache: {path}")
        return tuple(np.load(path, mmap_mode="r") for path in paths)
    folder.mkdir(parents=True, exist_ok=True)
    n, seq_len = len(data.meta), cfg["models"]["gru"]["seq_len"]
    arrays = [np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
              for path, shape in zip(paths, [(n, len(features.CORE_A_FEATURES)), (n, seq_len, 5), (n, seq_len)])]
    for start in range(0, n, 2000):
        end = min(start + 2000, n)
        batch = _extract([data.frame(i) for i in range(start, end)], seq_len)
        for target, values in zip(arrays, batch):
            target[start:end] = values
        print(f"Clean features {end:,}/{n:,}", flush=True)
    for array in arrays:
        array.flush()
    _json(metadata_path, {"fingerprint": fingerprint,
                          "files": {path.name: file_digest(path) for path in paths}})
    return tuple(arrays)


def _seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fold_models(data, cfg, clean, test_user, run_dir, identity, device,
                 inference_only=False):
    import joblib
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from .model_seq import GRUNet, _class_weight, _train_one_fold

    fold_dir = run_dir / "folds" / test_user
    fold_dir.mkdir(parents=True, exist_ok=True)
    info_path, rf_path, gru_path = [fold_dir / name for name in ["models.json", "rf.joblib", "gru.pt"]]
    train = data.meta["user_id"].to_numpy() != test_user
    y = data.meta["label"].map({c: i for i, c in enumerate(cfg["classes"])}).to_numpy(dtype=np.int64)
    expected = {"run_identity": identity, "test_user": test_user,
                "training_users": sorted(data.meta.loc[train, "user_id"].unique()),
                "train_window_ids_sha256": _digest(data.meta.loc[train, "window_id"].tolist()),
                "test_window_ids_sha256": _digest(data.meta.loc[~train, "window_id"].tolist())}
    if info_path.exists():
        saved = _read_json(info_path)
        if saved["identity"] != expected:
            raise ValueError(f"Model cache identity mismatch for {test_user}")
        if file_digest(rf_path) != saved["rf_sha256"] or file_digest(gru_path) != saved["gru_sha256"]:
            raise ValueError(f"Model cache corrupted for {test_user}")
        rf = joblib.load(rf_path)
        gru = GRUNet(hidden=cfg["models"]["gru"]["hidden"], n_class=len(cfg["classes"])).to(device)
        gru.load_state_dict(torch.load(gru_path, map_location=device, weights_only=True))
        return rf, gru, fold_dir, saved
    if inference_only:
        raise FileNotFoundError(f"No saved model for held-out user {test_user}; run evaluate without --inference-only first")
    if not train.any() or train.all():
        raise ValueError("LOUO requires at least two users and a nonempty held-out user")
    xa, xb, mask = clean
    seed = cfg["seeds"]["training"]
    _seed(seed, torch)  # MUST precede both model constructions.
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=cfg["models"]["rf"]["n_estimators"],
                                 class_weight=cfg["models"]["rf"]["class_weight"],
                                 random_state=seed, n_jobs=-1)
    rf.fit(xa[train], y[train])
    joblib.dump(rf, rf_path, compress=3)
    rf_seconds = time.perf_counter() - t0
    _seed(seed, torch)
    gru_cfg = cfg["models"]["gru"]
    gru = GRUNet(hidden=gru_cfg["hidden"], n_class=len(cfg["classes"])).to(device)
    t0 = time.perf_counter()
    gru = _train_one_fold(
        gru, torch.tensor(xb[train], dtype=torch.float32, device=device),
        torch.tensor(mask[train], dtype=torch.float32, device=device),
        torch.tensor(y[train], dtype=torch.int64, device=device),
        _class_weight(y[train], len(cfg["classes"])), epochs=gru_cfg["epochs"],
        batch_size=gru_cfg["batch_size"], lr=gru_cfg["lr"], seed=seed, device=device)
    torch.save(gru.state_dict(), gru_path)
    saved = {"identity": expected, "rf_sha256": file_digest(rf_path), "gru_sha256": file_digest(gru_path),
             "rf_training_seconds": rf_seconds, "gru_training_seconds": time.perf_counter() - t0}
    _json(info_path, saved)
    return rf, gru, fold_dir, saved


def _predict_condition(data, cfg, c, seed, test_index, rf, gru, clean, device):
    import torch
    n = len(test_index)
    survive = data.support[f"ds{c.downsample_sec}"][test_index]
    pred_rf, pred_gru = np.full(n, -1, dtype=int), np.full(n, -1, dtype=int)
    positions = np.flatnonzero(survive)
    batch_size = cfg["models"]["gru"]["batch_size"]
    gru.eval()
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            pos = positions[start:start + batch_size]
            idx = test_index[pos]
            if c.downsample_sec == 0 and c.noise_sigma_m == 0:
                xa, xb, mask = [array[idx] for array in clean]
            else:
                windows = [degrade_window(data.frame(int(i)), c, noise_seed=seed,
                                          window_id=int(data.meta.iloc[i]["window_id"])) for i in idx]
                # Coverage depends only on timestamps; assert the optimized pilot
                # and the shared protocol select exactly the same windows.
                if not all(window_quality(w, cfg)["usable"] for w in windows):
                    raise ValueError("Pilot/protocol coverage mismatch")
                xa, xb, mask = _extract(windows, cfg["models"]["gru"]["seq_len"])
            pred_rf[pos] = rf.predict(xa)
            pred_gru[pos] = gru(torch.tensor(xb, dtype=torch.float32, device=device),
                                 torch.tensor(mask, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    frame = data.meta.iloc[test_index][["window_id", "user_id", "label"]].reset_index(drop=True)
    frame["survive"] = survive
    frame["common"] = data.support["common"][test_index]
    frame["condition"], frame["noise_seed"] = c.name, seed
    frame["pred_rf"], frame["pred_gru"] = pred_rf, pred_gru
    return frame


def _score_predictions(frame, cfg, c, seed, cohort, n_total, is_full, identity):
    from sklearn.metrics import accuracy_score, f1_score
    usable = frame["survive"].astype(bool)
    selected = usable & frame["common"].astype(bool) if cohort == "common" else usable
    selected_frame = frame.loc[selected]
    y = selected_frame["label"].map({label: i for i, label in enumerate(cfg["classes"])}).to_numpy(dtype=int)
    labels = list(range(len(cfg["classes"])))
    result = {"condition": c.name, "downsample_sec": c.downsample_sec,
              "noise_sigma_m": c.noise_sigma_m, "noise_seed": seed, "cohort": cohort,
              "is_stress": c.is_stress, "scope": "full_louo" if is_full else "partial_louo",
              "n_reference": n_total, "n_usable": int(usable.sum()),
              "retention": float(usable.mean()) if len(frame) else 0.0, "n_scored": len(y),
              "status": "evaluated" if len(y) else "no_usable_windows", "run_sha256": identity}
    for name, column in [("rf", "pred_rf"), ("gru", "pred_gru")]:
        pred = selected_frame[column].to_numpy(dtype=int)
        if len(y):
            if (pred < 0).any():
                raise ValueError("Missing predictions on an eligible window")
            result[f"{name}_macro_f1"] = float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))
            result[f"{name}_accuracy"] = float(accuracy_score(y, pred))
            pc = f1_score(y, pred, labels=labels, average=None, zero_division=0)
        else:
            result[f"{name}_macro_f1"], result[f"{name}_accuracy"] = np.nan, np.nan
            pc = np.full(len(labels), np.nan)
        for label, value in zip(cfg["classes"], pc):
            result[f"{name}_f1_{label}"] = value
    result["gru_minus_rf"] = result["gru_macro_f1"] - result["rf_macro_f1"]
    result["gru_minus_rf_percentage_points"] = result["gru_minus_rf"] * 100
    return result


def _aggregate(run_dir, cfg, users, all_users, output_dir, identity, smoke):
    rows = []
    n_total = 0
    for c in iter_conditions(cfg):
        for seed in noise_seeds_for_condition(cfg, c):
            paths = [run_dir / "folds" / u / f"{c.name}_seed{seed}.csv.gz" for u in users]
            if not all(p.exists() for p in paths):
                continue
            frame = pd.concat([pd.read_csv(p, dtype={"user_id": str}) for p in paths], ignore_index=True)
            if frame["window_id"].duplicated().any():
                raise ValueError("Repeated OOF window IDs")
            n_total = len(frame)
            for cohort in ["condition", "common"]:
                rows.append(_score_predictions(frame, cfg, c, seed, cohort, n_total,
                                                set(users) == set(all_users) and not smoke, identity))
    results = pd.DataFrame(rows)
    if results.empty:
        return
    is_full = set(users) == set(all_users) and not smoke
    prefix = "" if is_full else "partial_"
    _csv(results, run_dir / "grid_results.csv")
    _csv(results, output_dir / f"{prefix}grid_results.csv")
    numeric = [c for c in results.columns if c.endswith("macro_f1") or c.startswith(("rf_f1_", "gru_f1_")) or c == "gru_minus_rf"]
    summary = results.groupby(["condition", "cohort", "scope"], sort=False)[numeric].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    _csv(summary.reset_index(), output_dir / f"{prefix}grid_seed_summary.csv")
    # A seed standard deviation is conditional on these fitted models and users;
    # it is not a confidence interval for unseen-user generalization.
    _json(output_dir / f"{prefix}evaluation_summary.json", {
        "run_sha256": identity, "scope": "full_louo" if is_full else "partial_louo",
        "smoke_overrides": smoke, "completed_users": users, "all_users": all_users,
        "n_reference": n_total, "seed_std_scope": "noise_realizations_only_fixed_models_and_users",
        "all_conditions_complete": len(results) == 2 * sum(len(noise_seeds_for_condition(cfg, c)) for c in iter_conditions(cfg))})


def evaluate(cfg: dict, output_dir: Path = OUTPUTS, users=None, threads=8,
             inference_only=False, smoke=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import sklearn
    import torch

    preparation = prepare(cfg, output_dir)
    data = Windows(cfg)
    all_users = sorted(data.meta["user_id"].unique())
    users = all_users if users is None else sorted(set(str(u).zfill(3) for u in users))
    if not set(users).issubset(all_users) or not users:
        raise ValueError(f"Unknown/empty held-out users: {users}; available: {all_users}")
    torch.set_num_threads(max(1, threads))
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "numpy": np.__version__, "pandas": pd.__version__,
                   "sklearn": sklearn.__version__, "torch": torch.__version__, "device": str(device),
                   "processor": platform.processor(), "torch_threads": torch.get_num_threads(),
                   "torch_interop_threads": torch.get_num_interop_threads(),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    identity_payload = {"preparation_fingerprint": preparation["fingerprint"],
                        "config_sha256": config_digest(cfg), "environment": environment}
    identity = _digest(identity_payload)
    run_dir = ARTIFACTS / f"run_{identity[:16]}"
    if inference_only:
        # Check BEFORE extracting all 90k clean feature windows. Missing historical
        # checkpoints are a known prerequisite failure, not a reason to spend time
        # rebuilding training features that this invocation cannot use.
        missing = [user for user in users if not all(
            (run_dir / "folds" / user / name).exists() for name in ["models.json", "rf.joblib", "gru.pt"])]
        if missing:
            raise FileNotFoundError(f"No complete saved model for held-out users {missing}; "
                                    "run evaluate without --inference-only first")
    run_dir.mkdir(parents=True, exist_ok=True)
    _json(run_dir / "run_manifest.json", dict(identity_payload, run_sha256=identity, config=cfg))
    clean = _clean_features(data, cfg, preparation["fingerprint"])
    completed = []
    for user in users:
        print(f"LOUO held-out user {user}; device={device}", flush=True)
        rf, gru, fold_dir, model_metadata = _fold_models(data, cfg, clean, user, run_dir,
                                                       identity, device, inference_only)
        test_index = np.flatnonzero(data.meta["user_id"].to_numpy() == user)
        for c in iter_conditions(cfg):
            for seed in noise_seeds_for_condition(cfg, c):
                target = fold_dir / f"{c.name}_seed{seed}.csv.gz"
                receipt = target.with_name(target.name + ".json")
                if target.exists() and receipt.exists():
                    saved = _read_json(receipt)
                    if (saved.get("run_sha256") != identity or saved.get("prediction_sha256") != file_digest(target)
                            or saved.get("rf_sha256") != model_metadata["rf_sha256"]
                            or saved.get("gru_sha256") != model_metadata["gru_sha256"]):
                        raise ValueError(f"Prediction cache corrupted: {target}")
                    continue
                frame = _predict_condition(data, cfg, c, seed, test_index, rf, gru, clean, device)
                _csv(frame, target)
                _json(receipt, {"run_sha256": identity, "prediction_sha256": file_digest(target),
                                "rf_sha256": model_metadata["rf_sha256"], "gru_sha256": model_metadata["gru_sha256"]})
                print(f"  {c.name} seed={seed} n={int(frame['survive'].sum())}", flush=True)
        completed.append(user)
        _aggregate(run_dir, cfg, completed, all_users, output_dir, identity, smoke)
    return run_dir


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "现实关联" / "degradation_config.yaml")
    parser.add_argument("--mode", choices=["prepare", "evaluate"], default="prepare")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--force", action="store_true", help="Rebuild preparation cache after data/config/code changes")
    parser.add_argument("--users", nargs="+", help="Subset of held-out users; output is explicitly partial LOUO")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--inference-only", action="store_true", help="Fail if the required fold checkpoints are absent")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.mode == "prepare":
        summary = prepare(cfg, args.output_dir, args.force)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    else:
        if args.force:
            prepare(cfg, args.output_dir, True)
        folder = evaluate(cfg, args.output_dir, args.users, args.threads, args.inference_only)
        print(f"Saved run: {folder}", flush=True)
