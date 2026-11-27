"""Launch member A using the current Python environment.

Usage from GeoLife/:
    python tools/run_member_a.py --mode prepare
    python tools/run_member_a.py --require-cuda --mode evaluate
    python tools/run_member_a.py --cpu --mode evaluate --users 010

Remaining arguments are forwarded unchanged to run_deployment_grid.py.
Runtime information and console output append to project/artifacts/member_a/launcher.log.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
JOB_RESULT = ROOT / "project" / "artifacts" / "member_a" / "job_result.json"


def write_job_result(value):
    JOB_RESULT.parent.mkdir(parents=True, exist_ok=True)
    temporary = JOB_RESULT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(JOB_RESULT)


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        count = self.terminal.write(value)
        self.logfile.write(value)
        return count

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return self.terminal.isatty()

    @property
    def encoding(self):
        return self.terminal.encoding


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--finalize-figures", action="store_true",
                        help="Generate measured plots after a complete full-LOUO evaluation")
    parser.add_argument("--log-file", type=Path,
                        default=ROOT / "project" / "artifacts" / "member_a" / "launcher.log")
    options, remaining = parser.parse_known_args(argv)
    if options.cpu and options.require_cuda:
        raise ValueError("--cpu and --require-cuda cannot be used together")
    # Must be set before importing CUDA PyTorch, including any indirect import.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if options.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT / "project"))
    options.log_file.parent.mkdir(parents=True, exist_ok=True)
    with options.log_file.open("a", encoding="utf-8", buffering=1) as logfile:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, logfile), Tee(stderr, logfile)
        state = {"status": "running", "pid": os.getpid(),
                 "started_at_utc": datetime.now(timezone.utc).isoformat(),
                 "python_executable": sys.executable, "arguments": remaining,
                 "finalize_figures_requested": options.finalize_figures,
                 "log_file": str(options.log_file)}
        write_job_result(state)
        try:
            import torch
            import numpy
            import pandas
            import sklearn

            available = torch.cuda.is_available()
            details = {
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "python_executable": sys.executable,
                "torch": torch.__version__, "torch_path": torch.__file__,
                "numpy": numpy.__version__, "pandas": pandas.__version__,
                "sklearn": sklearn.__version__, "dependency_paths": [],
                "cuda_available": available, "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if available else None,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "forwarded_arguments": remaining,
            }
            state["runtime"] = details
            write_job_result(state)
            print(json.dumps({"member_a_runtime": details}, ensure_ascii=False), flush=True)
            if options.require_cuda and not available:
                raise RuntimeError("CUDA was required but is unavailable; inspect the runtime details above")
            from research.deployment_grid import main as run_grid

            run_grid(remaining)
            state["figures_generated"] = False
            if options.finalize_figures:
                selection = argparse.ArgumentParser(add_help=False)
                selection.add_argument("--output-dir", type=Path, default=ROOT / "现实关联" / "outputs")
                selection.add_argument("--mode", default="prepare")
                selection.add_argument("--users", nargs="+")
                requested, _ = selection.parse_known_args(remaining)
                summary_path = requested.output_dir / "evaluation_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
                if (requested.mode == "evaluate" and not requested.users
                        and summary.get("scope") == "full_louo" and summary.get("all_conditions_complete") is True):
                    script = ROOT / "project" / "scripts" / "make_deployment_figures.py"
                    saved_argv = sys.argv
                    try:
                        sys.argv = [str(script), "--input-dir", str(requested.output_dir)]
                        runpy.run_path(str(script), run_name="__main__")
                    finally:
                        sys.argv = saved_argv
                    state["figures_generated"] = True
                    state["run_sha256"] = summary.get("run_sha256")
                else:
                    print("Figure finalization skipped: this invocation did not complete full LOUO.", flush=True)
            state.update(status="completed", finished_at_utc=datetime.now(timezone.utc).isoformat())
            write_job_result(state)
        except BaseException as exc:
            # argparse --help exits successfully without a completed experiment.
            success = isinstance(exc, SystemExit) and exc.code in (None, 0)
            state.update(status="completed" if success else "failed",
                         finished_at_utc=datetime.now(timezone.utc).isoformat())
            if not success:
                state["error"] = {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()}
            write_job_result(state)
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
