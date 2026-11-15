"""Compute four-condition paired Core A feature drift from A's verified core handoff."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".member_b_libs"))
sys.path.insert(0, str(ROOT / "tools"))
from member_b_bootstrap import bootstrap_research

bootstrap_research(ROOT)
import numpy as np
import pandas as pd
from research import features
from research.deployment_grid import Windows
from research.deployment_protocol import degrade_window, iter_conditions, load_config, window_quality

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = {
    "speed": ["mean_speed", "std_speed", "max_speed", "median_speed", "q25_speed", "q75_speed"],
    "acceleration": ["mean_acc", "std_acc", "max_acc"],
    "heading_change": ["mean_heading", "sum_heading", "std_heading"],
    "distance": ["path_length"],
    "sampling_time": ["duration", "n_points", "mean_gap"],
    "stop_ratio": ["stop_ratio"],
}
NAMES = ("ds0_n0", "ds10_n0", "ds0_n20", "ds10_n20")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    cfg = load_config(ROOT / "现实关联" / "degradation_config.yaml")
    data = Windows(cfg)
    columns = features.CORE_A_FEATURES
    if len(columns) != 17 or sorted(sum(GROUPS.values(), [])) != sorted(columns):
        raise ValueError("Frozen groups do not match CORE_A_FEATURES")
    indices = np.flatnonzero(data.support["common"])
    if len(indices) != 71941:
        raise ValueError(f"Unexpected common-support size: {len(indices)}")
    conditions = {c.name: c for c in iter_conditions(cfg) if c.name in NAMES}
    values = {name: np.empty((len(indices), len(columns)), dtype=np.float64) for name in NAMES}
    for position, index in enumerate(indices):
        window_id = int(data.meta.iloc[index]["window_id"])
        clean = data.frame(int(index))
        for name in NAMES:
            window = degrade_window(clean, conditions[name], noise_seed=0, window_id=window_id)
            if not window_quality(window, cfg)["usable"]:
                raise ValueError(f"Common window became unusable: {window_id}, {name}")
            row = features.window_statistical_features(window)
            values[name][position] = [row[col] for col in columns]
        if (position + 1) % 10000 == 0:
            print(f"Processed {position + 1}/{len(indices)} common windows", flush=True)
    # A's _extract replaces nonfinite values with zero before RF prediction.
    for name in NAMES:
        values[name] = np.nan_to_num(values[name], nan=0.0, posinf=0.0, neginf=0.0)

    feature_to_group = {column: group for group, members in GROUPS.items() for column in members}
    summary = []
    for name in NAMES:
        difference = values[name] - values["ds0_n0"]
        for j, column in enumerate(columns):
            delta = difference[:, j]
            absolute = np.abs(delta)
            summary.append({
                "condition": name, "feature_group": feature_to_group[column], "feature": column,
                "n_windows": len(indices), "median_clean": np.median(values["ds0_n0"][:, j]),
                "median_degraded": np.median(values[name][:, j]),
                "median_paired_difference": np.median(delta),
                "median_absolute_paired_difference": np.median(absolute),
                "q25_absolute_paired_difference": np.quantile(absolute, .25),
                "q75_absolute_paired_difference": np.quantile(absolute, .75),
            })
    pd.DataFrame(summary).to_csv(OUT / "feature_drift_by_feature.csv", index=False)

    group_rows = []
    for (condition, group), block in pd.DataFrame(summary).groupby(["condition", "feature_group"], sort=False):
        group_rows.append({"condition": condition, "feature_group": group,
                           "n_windows": len(indices),
                           "feature_median_absolute_changes": "; ".join(
                               f"{row.feature}={row.median_absolute_paired_difference}"
                               for row in block.itertuples()),
                           "aggregation_note": "Original feature units retained; no cross-unit sum"})
    pd.DataFrame(group_rows).to_csv(OUT / "feature_drift_by_group.csv", index=False)

    meta = data.meta.iloc[indices].reset_index(drop=True)
    breakdown = []
    for name in NAMES[1:]:
        absdiff = np.abs(values[name] - values["ds0_n0"])
        for dimension in ("label", "user_id"):
            for value, loc in meta.groupby(dimension, sort=True).indices.items():
                for j, column in enumerate(columns):
                    breakdown.append({"condition": name, "dimension": dimension,
                                      "value": value, "feature": column, "n_windows": len(loc),
                                      "median_absolute_paired_difference": np.median(absdiff[loc, j])})
    pd.DataFrame(breakdown).to_csv(OUT / "feature_drift_by_class_user.csv", index=False)

    handoff = json.loads((ROOT / "现实关联" / "outputs" / "member_a_handoff.json").read_text(encoding="utf-8"))
    files = [ROOT / "现实关联" / "artifacts" / "reference_manifest.csv",
             ROOT / "project" / "artifacts" / "member_a" / "support.npz",
             ROOT / "project" / "artifacts" / "member_a" / "windows.npz",
             ROOT / "现实关联" / "degradation_config.yaml",
             ROOT / "project" / "research" / "deployment_protocol.py",
             ROOT / "project" / "research" / "features.py"]
    record = {"status": "feature_drift_complete_permutation_pending_rf_models",
              "run_hash": handoff["run_sha256"], "cohort": "common", "noise_seed": 0,
              "n_windows": len(indices), "n_users": int(meta["user_id"].nunique()),
              "participating_users": sorted(meta["user_id"].astype(str).unique().tolist()),
              "feature_groups_version": "v1.0.0", "feature_groups": GROUPS,
              "feature_order": columns, "conditions": list(NAMES),
              "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
              "nonfinite_rule": "replace NaN and +/-Infinity with 0.0 as in A's _extract",
              "permutation_repeats_planned": 10, "permutation_seed_planned": 20261112,
              "permutation_executed": False}
    (OUT / "member_b_analysis_manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote feature drift outputs", flush=True)


if __name__ == "__main__":
    main()
