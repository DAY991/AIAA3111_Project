"""Export the completed member-A experiment without training or loading models.

Run from final/: python project/scripts/summarize_member_a.py
Only measured full-LOUO results are accepted. All paths in the handoff are
relative to final/ when possible; unavailable metrics remain missing values.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "现实关联" / "outputs"
DEFAULT_ARTIFACTS = ROOT / "project" / "artifacts" / "member_a"
COHORTS = ["common", "condition"]
METRICS = ["rf_macro_f1", "gru_macro_f1", "gru_minus_rf"]
EPSILON = 1e-12


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parsed_config_sha256(cfg: dict) -> str:
    raw = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, value: dict) -> None:
    # No NaN JSON literals; unavailable values are explicitly null.
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def expect_same(actual, expected, message: str) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=EPSILON, equal_nan=True):
        raise ValueError(message)


def condition_specs(cfg: dict) -> list[dict]:
    degradation = cfg["degradation"]
    return [{"condition": f"ds{step}_n{sigma:g}", "downsample_sec": int(step),
             "noise_sigma_m": float(sigma), "is_stress": stress}
            for steps, stress in [(degradation["main_downsample_sec"], False),
                                  (degradation["stress_downsample_sec"], True)]
            for step in steps for sigma in degradation["noise_sigma_m"]]


def condition_seeds(cfg: dict, spec: dict) -> list[int]:
    seeds = cfg["seeds"]
    if spec["condition"] in seeds["repeat_conditions"] and spec["noise_sigma_m"] > 0 and not spec["is_stress"]:
        return seeds["noise_repeats"]
    return [seeds["noise_primary"]]


def validate_results(grid: pd.DataFrame, seed_summary: pd.DataFrame, summary: dict,
                     preparation: dict, run: dict, cfg: dict, meta: pd.DataFrame,
                     support: dict) -> None:
    if summary.get("scope") != "full_louo" or not summary.get("all_conditions_complete") or summary.get("smoke_overrides"):
        raise ValueError("A completed full_louo run without smoke overrides is required")
    expected_users = sorted(meta["user_id"].unique())
    if sorted(summary["completed_users"]) != expected_users or sorted(summary["all_users"]) != expected_users:
        raise ValueError("Completed folds do not match the full reference user set")
    if (len(expected_users) != preparation["n_users"] or len(meta) != preparation["n_reference"]
            or summary.get("n_reference") != len(meta)):
        raise ValueError("Preparation/reference counts differ")
    run_id = summary["run_sha256"]
    if run["run_sha256"] != run_id or run["preparation_fingerprint"] != preparation["fingerprint"]:
        raise ValueError("Evaluation/run/preparation identities differ")
    if run["config"] != cfg or parsed_config_sha256(cfg) != run["config_sha256"] or run["config_sha256"] != preparation["config_sha256"]:
        raise ValueError("Current/frozen/preparation configurations differ")
    if not grid["scope"].eq("full_louo").all() or not grid["run_sha256"].eq(run_id).all():
        raise ValueError("grid_results.csv contains a partial or different run")
    expected_keys = {(spec["condition"], seed, cohort) for spec in condition_specs(cfg)
                     for seed in condition_seeds(cfg, spec) for cohort in COHORTS}
    keys = list(grid[["condition", "noise_seed", "cohort"]].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("Missing, duplicated, or unexpected condition/seed/cohort rows")
    if meta["window_id"].duplicated().any() or len(support["common"]) != len(meta):
        raise ValueError("Invalid reference manifest/support alignment")
    for spec in condition_specs(cfg):
        subset = grid.loc[grid["condition"].eq(spec["condition"])]
        if (not subset["downsample_sec"].eq(spec["downsample_sec"]).all()
                or not subset["noise_sigma_m"].eq(spec["noise_sigma_m"]).all()
                or not subset["is_stress"].eq(spec["is_stress"]).all()):
            raise ValueError(f"Condition parameter mismatch at {spec['condition']}")
        good = support[f"ds{spec['downsample_sec']}"]
        if not subset["n_reference"].eq(len(meta)).all() or not subset["n_usable"].eq(int(good.sum())).all():
            raise ValueError(f"Coverage mismatch at {spec['condition']}")
        expect_same(subset["retention"], float(good.mean()), f"Retention mismatch at {spec['condition']}")
        for cohort in COHORTS:
            mask = good & support["common"] if cohort == "common" else good
            rows = subset.loc[subset["cohort"].eq(cohort)]
            if not rows["n_scored"].eq(int(mask.sum())).all():
                raise ValueError(f"Scored-support mismatch at {spec['condition']} / {cohort}")
            if mask.any() and not rows["status"].eq("evaluated").all():
                raise ValueError("Eligible rows must have evaluated status")
            if not mask.any():
                if not rows["status"].eq("no_usable_windows").all() or not rows[METRICS].isna().all().all():
                    raise ValueError("Unusable conditions must retain missing F1 values")
    for model in ["rf", "gru"]:
        expect_same(grid[f"{model}_macro_f1"], grid[[f"{model}_f1_{label}" for label in cfg["classes"]]].mean(axis=1),
                    f"{model} macro-F1 differs from the fixed five-class mean")
    expect_same(grid["gru_minus_rf"], grid["gru_macro_f1"] - grid["rf_macro_f1"], "Stored model differences are inconsistent")
    expect_same(grid["gru_minus_rf_percentage_points"], 100 * grid["gru_minus_rf"], "Percentage-point conversion mismatch")
    # Recalculate from the individual seed rows; never copy summary statistics.
    groups = grid.groupby(["condition", "cohort", "scope"], sort=False)
    stored = seed_summary.set_index(["condition", "cohort", "scope"])
    if stored.index.duplicated().any() or set(stored.index) != set(groups.groups):
        raise ValueError("Seed-summary rows do not match the measured grid")
    for key, group in groups:
        for metric in METRICS:
            series = group[metric]
            for suffix, value in [("mean", series.mean()), ("std", series.std(ddof=1)), ("count", series.count())]:
                expect_same(stored.loc[key, f"{metric}_{suffix}"], value,
                            f"Seed-summary mismatch at {key}: {metric}_{suffix}")


def support_mask(spec: dict, cohort: str, support: dict) -> np.ndarray:
    usable = support[f"ds{spec['downsample_sec']}"]
    return usable & support["common"] if cohort == "common" else usable


def primary_table(grid: pd.DataFrame, cfg: dict, cohort: str,
                  meta: pd.DataFrame, support: dict) -> pd.DataFrame:
    specs = [spec for spec in condition_specs(cfg) if not spec["is_stress"]]
    selected = grid.loc[grid["cohort"].eq(cohort) & grid["noise_seed"].eq(cfg["seeds"]["noise_primary"])]
    table = selected.set_index("condition").loc[[spec["condition"] for spec in specs]].reset_index()
    clean = table.loc[table["condition"].eq("ds0_n0")].iloc[0]
    clean_spec = next(spec for spec in specs if spec["condition"] == "ds0_n0")
    clean_mask = support_mask(clean_spec, cohort, support)
    table["clean_baseline_same_support"] = [
        bool(np.array_equal(support_mask(spec, cohort, support), clean_mask)) for spec in specs]
    table["clean_baseline_n_scored"] = int(clean_mask.sum())
    table["n_scored_users"] = [int(meta.loc[support_mask(spec, cohort, support), "user_id"].nunique()) for spec in specs]
    table["scored_reference_fraction"] = table["n_scored"] / table["n_reference"]
    for model in ["rf", "gru"]:
        table[f"clean_{model}_macro_f1"] = clean[f"{model}_macro_f1"]
        table[f"{model}_f1_drop_from_clean"] = (
            clean[f"{model}_macro_f1"] - table[f"{model}_macro_f1"]
        ).where(table["clean_baseline_same_support"])
    table["gru_minus_rf"] = table["gru_macro_f1"] - table["rf_macro_f1"]
    table["gru_minus_rf_percentage_points"] = 100 * table["gru_minus_rf"]
    table["clean_gap_percentage_points"] = 100 * (clean["gru_macro_f1"] - clean["rf_macro_f1"])
    table["gap_change_from_clean_percentage_points"] = (
        table["gru_minus_rf_percentage_points"] - table["clean_gap_percentage_points"]
    ).where(table["clean_baseline_same_support"])
    table["baseline_comparison_scope"] = ("identical_common_windows" if cohort == "common"
                                           else "condition_specific_supports_selection_may_differ")
    return table


def key_seed_table(grid: pd.DataFrame, cfg: dict, meta: pd.DataFrame, support: dict) -> pd.DataFrame:
    rows = []
    for name in cfg["seeds"]["repeat_conditions"]:
        spec = next(spec for spec in condition_specs(cfg) if spec["condition"] == name)
        for cohort in COHORTS:
            group = grid.loc[grid["condition"].eq(name) & grid["cohort"].eq(cohort)]
            clean = grid.loc[grid["condition"].eq("ds0_n0") & grid["cohort"].eq(cohort)].iloc[0]
            clean_spec = next(spec for spec in condition_specs(cfg) if spec["condition"] == "ds0_n0")
            clean_mask = support_mask(clean_spec, cohort, support)
            same_support = bool(np.array_equal(support_mask(spec, cohort, support), clean_mask))
            for metric in METRICS:
                values = group[metric]
                baseline = float(clean[metric])
                difference = float(values.mean() - baseline) if same_support else np.nan
                rows.append({"condition": name, "cohort": cohort, "scope": "full_louo", "metric": metric,
                             "mean": float(values.mean()), "std_ddof1": float(values.std(ddof=1)),
                             "n_seeds": int(values.count()),
                             "noise_seeds": ";".join(map(str, sorted(group["noise_seed"]))),
                             "clean_baseline": baseline, "clean_baseline_same_support": same_support,
                             "clean_baseline_n_scored": int(clean_mask.sum()),
                             "mean_minus_clean": difference,
                             "mean_minus_clean_percentage_points": 100 * difference,
                             "n_reference": int(group["n_reference"].iloc[0]),
                             "n_scored": int(group["n_scored"].iloc[0]),
                             "n_scored_users": int(meta.loc[support_mask(spec, cohort, support), "user_id"].nunique()),
                             "uncertainty_scope": "noise_realizations_only_fixed_models_and_users",
                             "run_sha256": str(group["run_sha256"].iloc[0])})
    return pd.DataFrame(rows)


def per_class_table(primary: pd.DataFrame, cfg: dict, meta: pd.DataFrame, support: dict) -> pd.DataFrame:
    rows = []
    clean = primary.loc[primary["condition"].eq("ds0_n0")].iloc[0]
    common_meta = meta.loc[support["common"]]
    for record in primary.to_dict("records"):
        for label in cfg["classes"]:
            rf, gru = record[f"rf_f1_{label}"], record[f"gru_f1_{label}"]
            rows.append({"condition": record["condition"], "downsample_sec": record["downsample_sec"],
                         "noise_sigma_m": record["noise_sigma_m"], "noise_seed": record["noise_seed"],
                         "cohort": "common", "scope": "full_louo", "label": label,
                         "rf_f1": rf, "gru_f1": gru, "gru_minus_rf": gru - rf,
                         "gru_minus_rf_percentage_points": 100 * (gru - rf),
                         "clean_rf_f1": clean[f"rf_f1_{label}"], "clean_gru_f1": clean[f"gru_f1_{label}"],
                         "rf_f1_drop_from_clean": clean[f"rf_f1_{label}"] - rf,
                         "gru_f1_drop_from_clean": clean[f"gru_f1_{label}"] - gru,
                         "reference_class_count": int(meta["label"].eq(label).sum()),
                         "common_class_count": int(common_meta["label"].eq(label).sum()),
                         "n_scored": record["n_scored"], "n_scored_users": record["n_scored_users"],
                         "run_sha256": record["run_sha256"]})
    return pd.DataFrame(rows)


def qc_table(grid: pd.DataFrame, primary: pd.DataFrame, keys: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    selected = grid.loc[grid["cohort"].eq("common") & grid["noise_seed"].eq(cfg["seeds"]["noise_primary"])].set_index("condition")
    clean_gap = float(selected.loc["ds0_n0", "gru_macro_f1"] - selected.loc["ds0_n0", "rf_macro_f1"])
    sampling = cfg["degradation"]["main_downsample_sec"]
    rows = []
    for spec in condition_specs(cfg):
        source = selected.loc[spec["condition"]]
        row = {**spec, "cohort": "common", "noise_seed": cfg["seeds"]["noise_primary"],
               "n_reference": int(source["n_reference"]), "n_usable": int(source["n_usable"]),
               "retention": source["retention"], "n_scored": int(source["n_scored"]), "status": source["status"],
               "rf_macro_f1": source["rf_macro_f1"], "gru_macro_f1": source["gru_macro_f1"],
               "gru_minus_rf": source["gru_macro_f1"] - source["rf_macro_f1"],
               "winner": None, "winner_reversed_from_clean": False, "winner_reversed_from_previous_ds": False,
               "previous_ds": None, "rf_increases_vs_previous_ds": False, "gru_increases_vs_previous_ds": False,
               "rf_same_sigma_monotonic_nonincreasing": None, "gru_same_sigma_monotonic_nonincreasing": None,
               "rf_additive_reference": np.nan, "gru_additive_reference": np.nan,
               "rf_additive_reference_below_zero": None, "gru_additive_reference_below_zero": None,
               "rf_additive_reference_above_one": None, "gru_additive_reference_above_one": None,
               "rf_interaction_f1": np.nan, "gru_interaction_f1": np.nan,
               "gap_change_from_clean_percentage_points": np.nan,
               "repeated_mean_gap_change_from_clean_percentage_points": np.nan,
               "repeated_mean_gap_not_larger_than_clean": None, "run_sha256": source["run_sha256"]}
        notes = []
        if spec["is_stress"]:
            notes.append("30s压力条件无可用窗口；120s窗口不满足至少10点；F1保留NA，不作为0分")
        else:
            gap = float(row["gru_minus_rf"])
            winner = "GRU" if gap > EPSILON else ("RF" if gap < -EPSILON else "tie")
            row["winner"] = winner
            row["winner_reversed_from_clean"] = gap * clean_gap < -EPSILON
            gap_change = 100 * (gap - clean_gap)
            row["gap_change_from_clean_percentage_points"] = gap_change
            notes.append(f"{winner} Macro-F1较高；相对clean模型差值变化{gap_change:+.4f}个百分点")
            position = sampling.index(spec["downsample_sec"])
            for model in ["rf", "gru"]:
                values = [selected.loc[f"ds{step}_n{spec['noise_sigma_m']:g}", f"{model}_macro_f1"] for step in sampling]
                row[f"{model}_same_sigma_monotonic_nonincreasing"] = bool(np.all(np.diff(values) <= EPSILON))
                additive = float(selected.loc[f"ds{spec['downsample_sec']}_n0", f"{model}_macro_f1"]
                                 + selected.loc[f"ds0_n{spec['noise_sigma_m']:g}", f"{model}_macro_f1"]
                                 - selected.loc["ds0_n0", f"{model}_macro_f1"])
                row[f"{model}_additive_reference"] = additive
                row[f"{model}_additive_reference_below_zero"] = additive < 0
                row[f"{model}_additive_reference_above_one"] = additive > 1
                row[f"{model}_interaction_f1"] = float(row[f"{model}_macro_f1"] - additive)
                if additive < 0 or additive > 1:
                    notes.append(f"{model.upper()}加性参照{additive:.6f}超出F1的[0,1]范围；交互差受有界指标限制，不直接解释为机理或缓和强度")
            if position:
                previous_step = sampling[position - 1]
                previous = selected.loc[f"ds{previous_step}_n{spec['noise_sigma_m']:g}"]
                row["previous_ds"] = previous_step
                previous_gap = float(previous["gru_macro_f1"] - previous["rf_macro_f1"])
                row["winner_reversed_from_previous_ds"] = gap * previous_gap < -EPSILON
                for model in ["rf", "gru"]:
                    delta = float(row[f"{model}_macro_f1"] - previous[f"{model}_macro_f1"])
                    row[f"{model}_increases_vs_previous_ds"] = delta > EPSILON
                    if delta > EPSILON:
                        notes.append(f"同σ下{model.upper()}在ds{spec['downsample_sec']}较ds{previous_step}提高{100 * delta:.4f}个百分点；描述性非单调记录")
            repeated = keys.loc[keys["condition"].eq(spec["condition"]) & keys["cohort"].eq("common") & keys["metric"].eq("gru_minus_rf")]
            if len(repeated):
                delta = float(repeated["mean"].iloc[0] - clean_gap)
                row["repeated_mean_gap_change_from_clean_percentage_points"] = 100 * delta
                row["repeated_mean_gap_not_larger_than_clean"] = delta <= EPSILON
                notes.append(f"三噪声种子平均模型差值相对clean变化{100 * delta:+.4f}个百分点；绝对占优不自动证明新增鲁棒性优势")
        row["qc_note"] = "；".join(notes)
        rows.append(row)
    return pd.DataFrame(rows)


def runtime_summary(job: dict, models: list[dict], run_id: str, run_dir: Path) -> dict:
    if job.get("status") != "completed" or job.get("run_sha256") != run_id:
        raise ValueError("Job record must be completed and match the formal run")
    started, finished = [datetime.fromisoformat(job[key]) for key in ["started_at_utc", "finished_at_utc"]]
    wall = (finished - started).total_seconds()
    if wall < 0:
        raise ValueError("Negative job duration")
    result = {"run_sha256": run_id, "job_status": job["status"], "job_pid": job["pid"],
              "started_at_utc": job["started_at_utc"], "finished_at_utc": job["finished_at_utc"],
              "wall_seconds": wall, "wall_minutes": wall / 60, "wall_hours": wall / 3600,
              "n_folds": len(models), "n_saved_models": 2 * len(models),
              "run_directory": relative(run_dir), "environment": job.get("runtime", {}),
              "timer_semantics": {
                  "rf_training_seconds": "Recorded fold timer includes RF fitting and joblib serialization.",
                  "gru_training_seconds": "Recorded fold timer includes GRU fitting, torch save and adjacent model-file hashing.",
                  "wall_seconds": "Launcher start to successful figure finalization; not a pure GPU-kernel benchmark."}}
    total_bytes, recorded = 0, 0.0
    for model in ["rf", "gru"]:
        seconds = [float(item[f"{model}_training_seconds"]) for item in models]
        sizes = [(run_dir / "folds" / item["identity"]["test_user"] / ("rf.joblib" if model == "rf" else "gru.pt")).stat().st_size for item in models]
        if not all(np.isfinite(seconds)) or min(seconds) < 0:
            raise ValueError("Invalid recorded fold training durations")
        result[model] = {"n_models": len(models), "training_seconds_sum": sum(seconds),
                         "training_seconds_min": min(seconds), "training_seconds_max": max(seconds),
                         "training_seconds_mean": float(np.mean(seconds)), "training_seconds_median": float(np.median(seconds)),
                         "cache_bytes_total": sum(sizes), "cache_bytes_min": min(sizes), "cache_bytes_max": max(sizes)}
        total_bytes += sum(sizes)
        recorded += sum(seconds)
    result.update(model_cache_total_bytes=total_bytes, model_cache_total_GiB=total_bytes / 2 ** 30,
                  recorded_training_seconds_sum=recorded,
                  wall_minus_recorded_training_seconds=wall - recorded)
    return result


def export(input_dir: Path, output_dir: Path, artifacts: Path, config_path: Path,
           inventory_path: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: input_dir / filename for name, filename in {
        "grid_results": "grid_results.csv", "grid_seed_summary": "grid_seed_summary.csv",
        "evaluation_summary": "evaluation_summary.json", "prepare_summary": "prepare_summary.json"}.items()}
    summary, preparation = read_json(paths["evaluation_summary"]), read_json(paths["prepare_summary"])
    run_id = summary["run_sha256"]
    run_dir = artifacts / f"run_{run_id[:16]}"
    run_path = run_dir / "run_manifest.json"
    run = read_json(run_path)
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    manifest_path = resolve(cfg["data"]["reference_manifest"])
    source_paths = {"points_sha256": resolve(cfg["data"]["points_csv"]), "manifest_sha256": manifest_path,
                    "windows_sha256": artifacts / "windows.npz", "support_sha256": artifacts / "support.npz"}
    for name, path in source_paths.items():
        if sha256_file(path) != preparation[name]:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    meta = pd.read_csv(manifest_path, dtype={"user_id": str, "traj_id": str})
    with np.load(source_paths["support_sha256"]) as arrays:
        support = {key: arrays[key] for key in arrays.files}
    grid, seeds = pd.read_csv(paths["grid_results"]), pd.read_csv(paths["grid_seed_summary"])
    validate_results(grid, seeds, summary, preparation, run, cfg, meta, support)
    users = sorted(meta["user_id"].unique())
    models = []
    for user in users:
        item = read_json(run_dir / "folds" / user / "models.json")
        if (item["identity"]["run_identity"] != run_id or item["identity"]["test_user"] != user
                or sorted(item["identity"]["training_users"]) != [u for u in users if u != user]):
            raise ValueError(f"Fold model metadata mismatch: {user}")
        models.append(item)
    primary = {cohort: primary_table(grid, cfg, cohort, meta, support) for cohort in COHORTS}
    keys = key_seed_table(grid, cfg, meta, support)
    tables = {
        "grid_primary_common.csv": primary["common"], "grid_primary_condition.csv": primary["condition"],
        "grid_key_seed_summary.csv": keys,
        "per_class_robustness.csv": per_class_table(primary["common"], cfg, meta, support),
        "grid_qc_notes.csv": qc_table(grid, primary["common"], keys, cfg),
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    runtime_path = output_dir / "runtime_summary.json"
    job_path = artifacts / "job_result.json"
    write_json(runtime_path, runtime_summary(read_json(job_path), models, run_id, run_dir))
    inventory_path = inventory_path or input_dir / "member_a_fold_inventory.csv"
    inventory_sha = None
    if inventory_path.exists():
        inventory = pd.read_csv(inventory_path, dtype={"test_user": str})
        if sorted(inventory["test_user"]) != users or not inventory["run_sha256"].eq(run_id).all():
            raise ValueError("Model inventory does not match the formal run")
        inventory_sha = sha256_file(inventory_path)
    common_meta = meta.loc[support["common"]]
    representative = ["ds0_n0", "ds10_n0", "ds0_n20", "ds10_n20"]
    output_paths = {path.stem: relative(output_dir / path) for path in map(Path, tables)}
    output_paths["runtime_summary"] = relative(runtime_path)
    output_paths["member_a_handoff"] = relative(output_dir / "member_a_handoff.json")
    handoff = {
        "scope": "full_louo", "all_conditions_complete": True, "run_sha256": run_id,
        "protocol_version": cfg["protocol_version"], "preparation_fingerprint": preparation["fingerprint"],
        "frozen_implementation_sha256": preparation["implementation_sha256"],
        "config": {"path": relative(config_path), "parsed_sha256": parsed_config_sha256(cfg),
                   "file_sha256": sha256_file(config_path)},
        "inputs": {key.removesuffix("_sha256"): {"path": relative(path), "sha256": preparation[key]}
                   for key, path in source_paths.items()},
        "source_artifacts": {name: {"path": relative(path), "sha256": sha256_file(path)}
                             for name, path in dict(paths, run_manifest=run_path, job_result=job_path).items()},
        "evaluation": {"method": "leave_one_user_out", "n_folds": len(users), "users": users,
                       "n_reference": len(meta), "n_common": len(common_meta),
                       "n_common_users": int(common_meta["user_id"].nunique()),
                       "common_users": sorted(common_meta["user_id"].unique()),
                       "reference_users_without_common_windows": sorted(set(users) - set(common_meta["user_id"])),
                       "classes": cfg["classes"], "primary_cohort": "common",
                       "condition_specific_cohort": "condition", "fixed_five_class_macro_f1": True,
                       "unusable_condition_metric": None},
        "grid": {"main_downsample_sec": cfg["degradation"]["main_downsample_sec"],
                 "stress_downsample_sec": cfg["degradation"]["stress_downsample_sec"],
                 "noise_sigma_m_per_axis": cfg["degradation"]["noise_sigma_m"],
                 "main_conditions": primary["common"]["condition"].tolist(),
                 "stress_conditions": [s["condition"] for s in condition_specs(cfg) if s["is_stress"]],
                 "seeds": cfg["seeds"], "degradation": cfg["degradation"],
                 "noise_repeat_scope": "noise_realizations_only_fixed_models_and_users"},
        "member_b_interface": {
            "representative_conditions": representative, "recommended_comparison_cohort": "common",
            "noise_seed": cfg["seeds"]["noise_primary"], "n_common": len(common_meta),
            "protocol_module": "research.deployment_protocol",
            "degradation_call": "degrade_window(full_clean_window, condition, noise_seed=0, window_id=persistent_window_id)",
            "window_source": "research.deployment_grid.Windows(config); frame(index); meta.window_id is the persistent ID",
            "fold_selection": "For a window of user U, use the held-out-U model; its training users exclude U.",
            "model_inventory": {"path": relative(inventory_path), "sha256": inventory_sha,
                                "present_at_export": inventory_path.exists(), "fold_column": "test_user",
                                "rf_path_column": "rf_path", "rf_hash_column": "rf_sha256",
                                "gru_path_column": "gru_path", "gru_hash_column": "gru_sha256"},
            "model_paths": {
                "rf_template": relative(run_dir / "folds" / "{test_user}" / "rf.joblib"),
                "gru_template": relative(run_dir / "folds" / "{test_user}" / "gru.pt"),
                "metadata_template": relative(run_dir / "folds" / "{test_user}" / "models.json"),
                "rf_format": "joblib", "gru_format": "torch_state_dict",
                "gru_class": "research.model_seq.GRUNet", "model_parameters": cfg["models"]},
            "prediction_template": relative(run_dir / "folds" / "{test_user}" / "{condition}_seed{noise_seed}.csv.gz"),
            "coverage_independent_of_noise": True,
            "interpretation": "Absolute model advantage and improvement over the clean model gap are separate quantities; permutation importance is predictive, not causal."},
        "outputs": output_paths, "frozen_run_environment": run["environment"],
    }
    write_json(output_dir / "member_a_handoff.json", handoff)
    return {"run_sha256": run_id, "n_folds": len(users), "n_common_users": int(common_meta["user_id"].nunique()),
            "tables": {filename: len(table) for filename, table in tables.items()},
            "model_inventory_present": inventory_path.exists(), "output_dir": relative(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, help="Defaults to the input directory")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--config", type=Path, default=ROOT / "现实关联" / "degradation_config.yaml")
    parser.add_argument("--model-inventory", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.input_dir, args.output_dir or args.input_dir, args.artifact_root,
                            args.config, args.model_inventory), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
