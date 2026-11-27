"""Read-only progress snapshot for the frozen member-A experiment.

Usage: python tools/member_a_status.py [--json]
Counts distinguish finished model pairs from finished condition predictions.
The monitor never imports or changes the frozen scientific implementation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TZ = timezone(timedelta(hours=8))


def local_time(timestamp: float | None = None) -> str:
    value = datetime.now(LOCAL_TZ) if timestamp is None else datetime.fromtimestamp(timestamp, LOCAL_TZ)
    return value.isoformat(timespec="seconds")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def numeric(value) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def expected_conditions(cfg: dict) -> list[str]:
    """Derive expected receipts from the run's embedded, frozen configuration."""
    try:
        deg, seeds = cfg["degradation"], cfg["seeds"]
        keys = []
        for stress, steps in [(False, deg["main_downsample_sec"]), (True, deg["stress_downsample_sec"])]:
            for step in steps:
                for sigma in deg["noise_sigma_m"]:
                    condition = f"ds{step}_n{sigma:g}"
                    repeats = (seeds["noise_repeats"] if sigma > 0 and not stress
                               and condition in seeds["repeat_conditions"] else [seeds["noise_primary"]])
                    keys.extend(f"{condition}_seed{seed}" for seed in repeats)
        return keys
    except (KeyError, TypeError, ValueError):
        return []


def summarize_run(manifest_path: Path, expected_users: list[str]) -> dict:
    manifest = read_json(manifest_path)
    folder = manifest_path.parent
    if not isinstance(manifest, dict):
        return {"run": folder.name, "error": "run_manifest.json 暂不可读"}
    identity = manifest.get("run_sha256")
    environment = manifest.get("environment", {})
    keys = expected_conditions(manifest.get("config", {}))
    counts = {key: 0 for key in keys}
    finished_models, fully_predicted, rf_only = [], [], []
    invalid_receipts = 0
    rf_sum, gru_sum = 0.0, 0.0
    newest_model = None
    newest_stamp, newest_path = manifest_path.stat().st_mtime, manifest_path
    folds = folder / "folds"
    for fold in sorted(folds.iterdir()) if folds.exists() else []:
        if not fold.is_dir():
            continue
        info_path = fold / "models.json"
        info = read_json(info_path)
        valid_model = (isinstance(info, dict) and isinstance(info.get("identity"), dict)
                       and info["identity"].get("run_identity") == identity
                       and (fold / "rf.joblib").exists() and (fold / "gru.pt").exists())
        if valid_model:
            finished_models.append(fold.name)
            rf_seconds = numeric(info.get("rf_training_seconds"))
            gru_seconds = numeric(info.get("gru_training_seconds"))
            rf_sum += rf_seconds
            gru_sum += gru_seconds
            stamp = info_path.stat().st_mtime
            if newest_model is None or stamp > newest_model[0]:
                newest_model = (stamp, {"user": fold.name, "rf_training_seconds": rf_seconds,
                                       "gru_training_seconds": gru_seconds, "finished_at_local": local_time(stamp)})
        elif (fold / "rf.joblib").exists():
            rf_only.append(fold.name)
        fold_keys = set()
        for path in fold.iterdir():
            if not path.is_file():
                continue
            stamp = path.stat().st_mtime
            if stamp > newest_stamp:
                newest_stamp, newest_path = stamp, path
            if not path.name.endswith(".csv.gz.json"):
                continue
            key = path.name.removesuffix(".csv.gz.json")
            receipt = read_json(path)
            prediction_path = path.with_name(path.name.removesuffix(".json"))
            valid = (valid_model and key in counts and isinstance(receipt, dict)
                     and receipt.get("run_sha256") == identity and prediction_path.exists()
                     and receipt.get("rf_sha256") == info.get("rf_sha256")
                     and receipt.get("gru_sha256") == info.get("gru_sha256"))
            if valid:
                counts[key] += 1
                fold_keys.add(key)
            else:
                invalid_receipts += 1
        if keys and set(keys) <= fold_keys:
            fully_predicted.append(fold.name)
    summaries = []
    for path in (ROOT / "现实关联" / "outputs").glob("*evaluation_summary.json"):
        value = read_json(path)
        if isinstance(value, dict) and value.get("run_sha256") == identity:
            if path.stat().st_mtime > newest_stamp:
                newest_stamp, newest_path = path.stat().st_mtime, path
            summaries.append({"file": str(path.relative_to(ROOT)), "scope": value.get("scope"),
                              "completed_users": len(value.get("completed_users", [])),
                              "n_reference": value.get("n_reference"),
                              "all_conditions_complete": value.get("all_conditions_complete"),
                              "modified_at_local": local_time(path.stat().st_mtime)})
    aggregate = folder / "grid_results.csv"
    aggregate_info = None
    if aggregate.exists():
        if aggregate.stat().st_mtime > newest_stamp:
            newest_stamp, newest_path = aggregate.stat().st_mtime, aggregate
        try:
            with aggregate.open(encoding="utf-8-sig", newline="") as handle:
                rows = [r for r in csv.DictReader(handle) if r.get("run_sha256") == identity]
            aggregate_info = {"row_count": len(rows), "scope": sorted({r.get("scope", "") for r in rows}),
                              "n_reference": sorted({int(r["n_reference"]) for r in rows}),
                              "modified_at_local": local_time(aggregate.stat().st_mtime)}
        except (OSError, ValueError, KeyError):
            aggregate_info = {"error": "grid_results.csv 暂不可读"}
    preparation = manifest.get("preparation_fingerprint", "")
    feature_folder = folder.parent / f"features_{preparation[:16]}"
    feature_files = [feature_folder / name for name in ["core_a.npy", "core_b.npy", "mask.npy", "metadata.json"]]
    for path in feature_files:
        if path.exists() and path.stat().st_mtime > newest_stamp:
            newest_stamp, newest_path = path.stat().st_mtime, path
    n_users = len(expected_users)
    return {"run": folder.name, "run_sha256": identity,
            "torch": environment.get("torch"), "device": environment.get("device"),
            "device_name": environment.get("device_name"), "torch_threads": environment.get("torch_threads"),
            "expected_folds": n_users,
            "model_pairs_finished": len(finished_models), "model_pair_users": finished_models,
            "folds_with_all_prediction_receipts": len(fully_predicted), "prediction_complete_users": fully_predicted,
            "prediction_receipts_finished": sum(counts.values()), "prediction_receipts_expected": n_users * len(keys),
            "prediction_receipts_by_condition_seed": counts, "invalid_or_unmatched_receipts": invalid_receipts,
            "rf_written_without_model_receipt": rf_only,
            "rf_training_seconds_sum": round(rf_sum, 3), "gru_training_seconds_sum": round(gru_sum, 3),
            "last_model_pair": newest_model[1] if newest_model else None,
            "clean_features_metadata_present": (feature_folder / "metadata.json").exists(),
            "clean_feature_bytes_present": sum(path.stat().st_size for path in feature_files[:3] if path.exists()),
            "output_summaries": summaries, "run_grid": aggregate_info,
            "latest_artifact_at_local": local_time(newest_stamp),
            "latest_artifact": str(newest_path.relative_to(ROOT)),
            "count_validation": "收据身份和文件存在检查；未重新计算所有模型及预测文件哈希"}


def nvidia_snapshot() -> dict:
    def query(fields, kind):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        return subprocess.run(["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=5, creationflags=flags)
    result = {"available": False, "gpus": [], "python_compute_processes": []}
    try:
        devices = query("name,utilization.gpu,memory.used,memory.total", "gpu")
        if devices.returncode != 0:
            result["note"] = "nvidia-smi 暂不可用"
            return result
        result["available"] = True
        for row in csv.reader(devices.stdout.splitlines(), skipinitialspace=True):
            if len(row) == 4:
                result["gpus"].append(dict(zip(["name", "utilization_percent", "memory_used_mib", "memory_total_mib"], row)))
        apps = query("pid,process_name,used_gpu_memory", "compute-apps")
        if apps.returncode == 0:
            for row in csv.reader(apps.stdout.splitlines(), skipinitialspace=True):
                if len(row) != 3 or "python" not in Path(row[1]).name.lower():
                    continue
                try:
                    result["python_compute_processes"].append({"pid": int(row[0]), "name": Path(row[1]).name,
                                                                "gpu_memory_mib": row[2]})
                except ValueError:
                    continue
        else:
            result["compute_process_note"] = "驱动暂未提供计算进程列表"
    except (OSError, subprocess.TimeoutExpired):
        result["note"] = "nvidia-smi 未找到或查询超时"
    return result


def member_a_processes() -> dict:
    """Expose only the experiment's PIDs and counters, never command lines."""
    try:
        import psutil
    except ImportError:
        return {"available": False, "processes": [], "note": "未安装 psutil；仍可查看 GPU 进程"}
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time", "cpu_times", "memory_info"]):
        try:
            details = process.info
            if "python" not in (details.get("name") or "").lower():
                continue
            args = details.get("cmdline") or []
            if not any(any(entry in str(arg) for entry in ("run_deployment_grid.py", "run_member_a.py"))
                       for arg in args):
                continue
            cpu, memory = details.get("cpu_times"), details.get("memory_info")
            found.append({"pid": details["pid"], "name": details["name"],
                          "started_at_local": local_time(details["create_time"]),
                          "cpu_seconds": round(cpu.user + cpu.system, 2) if cpu else None,
                          "working_set_mib": round(memory.rss / 1024 ** 2, 1) if memory else None})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return {"available": True, "processes": found}


def snapshot() -> dict:
    artifacts = ROOT / "project" / "artifacts" / "member_a"
    folds = read_json(artifacts / "fold_manifest.json") or {}
    expected_users = [str(f["test_user"]) for f in folds.get("folds", []) if "test_user" in f]
    return {"generated_at_local": local_time(), "timezone": "UTC+08:00",
            "expected_users": len(expected_users),
            "runs": [summarize_run(path, expected_users) for path in sorted(artifacts.glob("run_*/run_manifest.json"))],
            "gpu": nvidia_snapshot(), "member_a_processes": member_a_processes()}


def print_text(status: dict) -> None:
    print(f"成员 A 状态 · {status['generated_at_local']}")
    if not status["runs"]:
        print("尚未发现正式模型运行目录。")
    for run in status["runs"]:
        print(f"\n{run['run']}")
        if "error" in run:
            print(run["error"])
            continue
        print(f"  PyTorch {run['torch']} · {run['device']} · {run['device_name']} · torch线程 {run['torch_threads']}")
        print(f"  模型对完成 {run['model_pairs_finished']}/{run['expected_folds']} 折；"
              f"所有条件预测完成 {run['folds_with_all_prediction_receipts']}/{run['expected_folds']} 折")
        print(f"  预测收据 {run['prediction_receipts_finished']}/{run['prediction_receipts_expected']}；"
              f"干净特征缓存{'已完成' if run['clean_features_metadata_present'] else '尚未完成'}")
        counts = run["prediction_receipts_by_condition_seed"]
        items = [f"{key}={value}/{run['expected_folds']}" for key, value in counts.items()]
        for start in range(0, len(items), 4):
            print("    " + "  ".join(items[start:start + 4]))
        print(f"  累计训练：RF {run['rf_training_seconds_sum']:.1f}s；GRU {run['gru_training_seconds_sum']:.1f}s")
        last = run["last_model_pair"]
        if last:
            print(f"  最近模型折 {last['user']}：RF {last['rf_training_seconds']:.1f}s，GRU {last['gru_training_seconds']:.1f}s")
        if run["rf_written_without_model_receipt"]:
            print("  已写 RF、模型对尚未完成：" + ", ".join(run["rf_written_without_model_receipt"]))
        for summary in run["output_summaries"]:
            print(f"  汇总 {summary['scope']}：{summary['completed_users']} 用户，"
                  f"条件齐全={summary['all_conditions_complete']}")
        if not run["output_summaries"]:
            print("  尚无本次运行对应的输出汇总。")
        if run["invalid_or_unmatched_receipts"]:
            print(f"  未匹配或未完成的预测收据：{run['invalid_or_unmatched_receipts']}")
        print(f"  最新写入 {run['latest_artifact_at_local']} · {run['latest_artifact']}")
    for device in status["gpu"]["gpus"]:
        print(f"\nGPU {device['name']}：利用率 {device['utilization_percent']}%，"
              f"显存 {device['memory_used_mib']}/{device['memory_total_mib']} MiB")
    gpu_apps = status["gpu"]["python_compute_processes"]
    print("GPU Python 进程：" + (", ".join(str(p["pid"]) for p in gpu_apps) if gpu_apps else "未发现或驱动未列出"))
    processes = status["member_a_processes"]["processes"]
    if processes:
        print("A 实验进程：" + "; ".join(f"PID {p['pid']}，CPU累计 {p['cpu_seconds']}s，内存 {p['working_set_mib']} MiB" for p in processes))
    elif status["member_a_processes"]["available"]:
        print("A 实验进程：未发现成员A实验入口进程。")
    elif not status["member_a_processes"]["available"]:
        print(status["member_a_processes"]["note"])


def main() -> None:
    # Windows redirected pipes may default to cp1252 even in a Chinese shell.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable UTF-8 JSON snapshot")
    args = parser.parse_args()
    status = snapshot()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print_text(status)


if __name__ == "__main__":
    main()
