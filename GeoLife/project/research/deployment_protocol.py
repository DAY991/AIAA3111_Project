"""Frozen, reproducible degradation protocol shared by members A and B.

Input windows must be the complete, time-ordered clean reference windows.
``window_id`` is the persistent reference-manifest ID, never a fold-local index.
The Gaussian realization is generated for *all original points* before selecting
retained point indices. Thus the same original point gets the same standardized
noise across sampling rates, models and noise magnitudes. Coordinates are changed
only after selecting the bucket-first sample, matching the legacy operation order.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .config import CLASSES
from .degradation import downsample_window

FINAL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = FINAL_ROOT / "现实关联" / "degradation_config.yaml"
_NOISE_DOMAIN = "geolife-deployment-paired-noise-v1"


@dataclass(frozen=True)
class GridCondition:
    name: str
    downsample_sec: int
    noise_sigma_m: float
    is_stress: bool = False


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_config(cfg: dict) -> None:
    """Reject ambiguous or incompatible changes to the version-1 protocol."""
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    try:
        if cfg["protocol_version"] != "1.0.0":
            raise ValueError("Unsupported protocol_version; expected 1.0.0")
        if cfg["classes"] != CLASSES:
            raise ValueError("classes must preserve the fixed five-class order")
        for key in ("points_csv", "reference_manifest"):
            if not isinstance(cfg["data"][key], str) or not cfg["data"][key]:
                raise ValueError(f"data.{key} must be a nonempty path")
        seg = cfg["segmentation"]
        if (seg["window_sec"], seg["min_points"], seg["max_gap_sec"]) != (120, 10, 30):
            raise ValueError("Protocol 1.0.0 freezes 120s / 10 points / 30s maximum gap")
        if cfg["split"] != {"method": "leave_one_user_out", "group_column": "user_id"}:
            raise ValueError("The split must remain leave_one_user_out by user_id")
        deg = cfg["degradation"]
        expected = {
            "main_downsample_sec": [0, 5, 10],
            "stress_downsample_sec": [30],
            "noise_sigma_m": [0, 5, 10, 20],
            "downsample_method": "first_point_per_time_bucket",
            "order": "downsample_then_noise",
            "noise_distribution": "independent_gaussian_xy",
            "noise_scale": "per_axis_sigma_m",
            "pairing": "noise_seed_window_id_original_point_index",
        }
        for key, value in expected.items():
            if deg[key] != value:
                raise ValueError(f"Protocol 1.0.0 requires degradation.{key} = {value!r}")
        seeds = cfg["seeds"]
        for key in ("training", "noise_primary"):
            if not _nonnegative_int(seeds[key]):
                raise ValueError(f"seeds.{key} must be a nonnegative integer")
        repeats = seeds["noise_repeats"]
        if not isinstance(repeats, list) or not repeats or not all(map(_nonnegative_int, repeats)):
            raise ValueError("seeds.noise_repeats must contain nonnegative integers")
        if len(repeats) != len(set(repeats)) or seeds["noise_primary"] not in repeats:
            raise ValueError("noise_repeats must be unique and include noise_primary")
        repeat_conditions = seeds["repeat_conditions"]
        valid_repeats = {c.name for c in iter_conditions(cfg, include_stress=False) if c.noise_sigma_m > 0}
        if not isinstance(repeat_conditions, list) or not set(repeat_conditions) <= valid_repeats:
            raise ValueError("Only main-grid conditions with positive noise can repeat")
        rf, gru = cfg["models"]["rf"], cfg["models"]["gru"]
        if rf != {"n_estimators": 300, "class_weight": "balanced"}:
            raise ValueError("Protocol 1.0.0 freezes the RF training parameters")
        if gru != {"seq_len": 128, "hidden": 64, "epochs": 15, "batch_size": 512, "lr": 0.001}:
            raise ValueError("Protocol 1.0.0 freezes the GRU training parameters")
        ev = cfg["evaluation"]
        if (ev["primary_metric"] != "macro_f1" or ev["fixed_class_order"] is not True
                or ev["zero_division"] != 0 or ev["report_common_support"] is not True
                or ev["report_class_coverage"] is not True or ev["no_usable_windows_metric"] is not None):
            raise ValueError("Evaluation must report fixed-label macro F1, coverage and common support; unavailable F1 is null")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed configuration: {exc}") from exc


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    return cfg


def config_digest(cfg: dict) -> str:
    """Hash parsed content, independent of YAML comments and mapping key order."""
    raw = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a configured path relative to the final workspace root."""
    path = Path(path)
    return path if path.is_absolute() else FINAL_ROOT / path


def iter_conditions(cfg: dict, include_stress: bool = True) -> list[GridCondition]:
    deg = cfg["degradation"]
    rows = [(s, False) for s in deg["main_downsample_sec"]]
    if include_stress:
        rows += [(s, True) for s in deg["stress_downsample_sec"]]
    return [GridCondition(f"ds{s}_n{sigma:g}", int(s), float(sigma), stress)
            for s, stress in rows for sigma in deg["noise_sigma_m"]]


def noise_seeds_for_condition(cfg: dict, condition: GridCondition) -> list[int]:
    seeds = cfg["seeds"]
    if condition.noise_sigma_m > 0 and not condition.is_stress and condition.name in seeds["repeat_conditions"]:
        return list(seeds["noise_repeats"])
    return [int(seeds["noise_primary"])]


def standardized_noise(window_id: str | int, n_points: int, noise_seed: int = 0) -> np.ndarray:
    """Return stable (original point, xy-axis) N(0,1) draws, never Python hash().

    NumPy PCG64 is explicit; seed entropy is the complete SHA-256 digest of a
    JSON tuple (domain, integer noise seed, string manifest window ID).
    """
    if not _nonnegative_int(noise_seed):
        raise ValueError("noise_seed must be a nonnegative integer")
    if not _nonnegative_int(n_points):
        raise ValueError("n_points must be a nonnegative integer")
    payload = json.dumps([_NOISE_DOMAIN, noise_seed, str(window_id)], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    entropy = np.frombuffer(digest, dtype="<u4").tolist()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    return rng.standard_normal((n_points, 2))


def degrade_window(w: pd.DataFrame, condition: GridCondition, noise_seed: int = 0,
                   *, window_id: str | int) -> pd.DataFrame:
    """Degrade a clean reference-window copy with paired pointwise noise.

    Output retains the original columns and has a fresh RangeIndex. Reordered
    window lists are safe when their persistent window IDs travel with them.
    """
    if condition.downsample_sec < 0 or condition.noise_sigma_m < 0:
        raise ValueError("Downsampling and Gaussian sigma must be nonnegative")
    if not {"datetime", "x", "y"} <= set(w.columns):
        raise ValueError("Reference windows require datetime, x and y columns")
    if not w["datetime"].is_monotonic_increasing or w["datetime"].isna().any():
        raise ValueError("Reference windows must be chronologically ordered with valid timestamps")
    # Position, rather than DataFrame index labels, identifies original points.
    # Selecting a temporary positional index through the legacy function keeps
    # the historical bucket-first rule exact, even for unusual input indices.
    indexed = w.copy()
    indexed.index = pd.RangeIndex(len(w))
    if condition.downsample_sec and len(indexed) > 1:
        offsets = (indexed["datetime"] - indexed["datetime"].iloc[0]).dt.total_seconds()
        buckets = (offsets // condition.downsample_sec).astype(int)
        selected = np.flatnonzero(~buckets.duplicated(keep="first").to_numpy())
    else:
        selected = np.arange(len(indexed))
    degraded = downsample_window(indexed, condition.downsample_sec)
    if condition.noise_sigma_m > 0:
        noise = standardized_noise(window_id, len(indexed), noise_seed)[selected]
        degraded.loc[:, "x"] = degraded["x"].to_numpy(dtype=float) + condition.noise_sigma_m * noise[:, 0]
        degraded.loc[:, "y"] = degraded["y"].to_numpy(dtype=float) + condition.noise_sigma_m * noise[:, 1]
    return degraded.reset_index(drop=True)


def window_quality(w: pd.DataFrame, cfg: dict) -> dict:
    """Reapply the clean eligibility rule; coordinates do not affect coverage."""
    n = len(w)
    gaps = w["datetime"].diff().dt.total_seconds().dropna()
    max_gap = float(gaps.max()) if len(gaps) else 0.0
    enough = n >= cfg["segmentation"]["min_points"]
    continuous = max_gap <= cfg["segmentation"]["max_gap_sec"]
    return {"usable": bool(enough and continuous), "n_points": n,
            "max_gap_sec": max_gap,
            "reason": "ok" if enough and continuous else ("too_few_points" if not enough else "gap_too_large")}


def degrade_windows(windows: Iterable[pd.DataFrame], window_ids: Iterable[str | int],
                    condition: GridCondition, noise_seed: int, cfg: dict) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    windows, window_ids = list(windows), list(window_ids)
    if len(windows) != len(window_ids):
        raise ValueError("Windows and persistent window IDs must have identical lengths")
    if len(set(map(str, window_ids))) != len(window_ids):
        raise ValueError("Reference window IDs must be unique")
    degraded, quality = [], []
    for w, wid in zip(windows, window_ids):
        d = degrade_window(w, condition, noise_seed, window_id=wid)
        degraded.append(d)
        quality.append({"window_id": wid, **window_quality(d, cfg)})
    return degraded, pd.DataFrame(quality, columns=["window_id", "usable", "n_points", "max_gap_sec", "reason"])


def coverage_summary(quality: pd.DataFrame) -> dict:
    n = len(quality)
    usable = int(quality["usable"].sum())
    return {"n_reference": n, "n_usable": usable, "retention": usable / n if n else float("nan")}
