# Data preparation and reproduction

Run the initial `cd GeoLife` once from the repository root; run all subsequent commands from `GeoLife/`. The published `现实关联/outputs/` and `member_b/outputs/` directories are archived results. Fresh A runs below write to `work/reproduced_a/` so they do not replace those records.

## 1. Environment

Python 3.12 is recommended. The archived A run used NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, SciPy 1.17.1, joblib 1.5.3 and PyTorch 2.11.0+cu128. Keep scikit-learn 1.8.0 when loading the saved RF models; earlier bundled environments used different RF bootstrap semantics.

```powershell
cd GeoLife
py -3.12 -m venv .venv
$projectPython = '.\.venv\Scripts\python.exe'
& $projectPython -m pip install --upgrade pip

# Choose ONE torch build. CPU works, but complete training is slower.
& $projectPython -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
# Alternatively, for a compatible NVIDIA device/driver:
# & $projectPython -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

& $projectPython -m pip install -r requirements-member-a.txt
& $projectPython -m unittest discover -s project/tests -v
```

On another shell use the virtual environment's Python executable in place of `$projectPython`. The tests use small synthetic fixtures to check parsing, split/coverage rules, paired noise, model persistence and resume behavior. They do not reproduce the reported full-data scores.

`tools/run_member_a.py` uses the active Python environment. It does not depend on the original author's absolute interpreter path or copied `.libs` directories.

## 2. Obtain the original data and build points.csv

Obtain **GeoLife GPS Trajectories 1.3** from the [Microsoft Research dataset/user-guide page](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) and follow the source's terms. After extraction, identify the `Data` directory containing numbered users:

```text
Geolife Trajectories 1.3/
  Data/
    000/Trajectory/*.plt
    ...
    010/labels.txt
    010/Trajectory/*.plt
```

Point `--raw-root` at that `Data` directory, not at its parent:

```powershell
& $projectPython project/scripts/prepare_geolife.py --raw-root 'D:\datasets\Geolife Trajectories 1.3\Data'
```

This creates `project/data/processed/points.csv` and `data_preparation.json`, including counts and file hashes. It reuses the frozen loader's time-interval label matching, local metre coordinates and label normalization. Unsupported labels remain missing at this stage and are excluded by the reference-window segmentation; removing them early could change same-label boundaries.

The script creates its output directory and refuses to overwrite an existing points file unless `--overwrite` is supplied. An optional `--output-dir` is useful for inspection, but the frozen A configuration expects the default `project/data/processed/points.csv` location.

Expected original preparation: 5,434,096 loaded points, 115,150 candidate windows, 90,120 clean-usable reference windows from 54 users. The original data source has 182 users, only a subset has transportation labels, and alignment/window rules further reduce that set. A fresh environment can produce a different CSV byte hash or run identity; do not change archived hashes to make it appear identical.

## 3. Freeze windows and measure coverage

```powershell
& $projectPython tools/run_member_a.py --mode prepare --output-dir work/reproduced_a
```

This reconstructs persistent reference IDs and writes packed windows, support masks and the LOUO fold manifest under `project/artifacts/member_a/`. Expected coverage is native 90,120, 5s 84,195, 10s 71,941, and 30s zero. The main-grid common support contains 71,941 windows from 48 users, while the training framework still has 54 folds. `prepare` produces coverage only, not model scores.

If data, configuration or frozen source files change, the preparation cache refuses reuse. Preserve any archive you need before deliberately using `--force`; preparation uses a fixed cache location, while models/features are stored under fingerprinted directories.

## 4. Train and evaluate A

For one real held-out user, retaining all other users for training:

```powershell
& $projectPython tools/run_member_a.py --mode evaluate --users 010 --threads 8 --output-dir work/reproduced_a
```

This is **partial LOUO**, including full training for that fold, not a miniature smoke test. Then run all folds; an existing compatible first fold is reused:

```powershell
& $projectPython tools/run_member_a.py --mode evaluate --threads 8 --output-dir work/reproduced_a --finalize-figures
# Add --require-cuda to fail instead of falling back to CPU.
# Add --cpu to explicitly disable CUDA.
```

Every fold trains once on clean reference windows. Both models reuse paired degraded test windows for the 12 main conditions; `ds10_n10` has noise seeds 0/1/2. Four 30s stress conditions retain missing scores. Training seed 0 is set before model initialization. Predictions, model hashes and receipts allow resuming completed folds/conditions; an interrupted unfinished training fold is retrained.

The historical run completed in about 2h16m on an RTX 5060 Laptop GPU, excluding the earlier clean-feature preparation. This is an observed runtime, not a prediction for another machine. No fresh full 54-fold training was run just to publish these entry points.

## 5. Export and independently verify a completed A run

```powershell
# Produce interaction_contrasts.csv before the verifier checks that table.
& $projectPython project/scripts/make_deployment_figures.py --input-dir work/reproduced_a
& $projectPython tools/verify_member_a.py --input-dir work/reproduced_a
& $projectPython project/scripts/summarize_member_a.py --input-dir work/reproduced_a
```

The verifier discovers the completed run ID from that directory's `evaluation_summary.json`; `--run run_<prefix>` can select one explicitly. It checks the full-data protocol counts, file integrity, user separation, per-window labels/support and independently pooled five-class metrics without retraining. It expects the complete frozen GeoLife experiment, not a sample-data exception.

The archived result is `04d2f3dee1b64ac6e6d359b0aae75128a001a8cd25ac2f4df9afce4a389a1d50`. A new environment/source-byte identity can produce another run directory. Keep new outputs separate; do not claim a new training run is the original archive merely because the protocol parameters match. Tiny fixtures check functionality; only full-data results support comparisons to the reported grid.

## 6. Restore inputs for the archived B analysis

B can inspect the published CSVs immediately. Recomputing the existing B analysis requires the original A handoff artifacts, available in the team's previously prepared packages:

- `GeoLife_MemberB_Core_20261115.zip` (about 72.2 MiB);
- `GeoLife_MemberB_RF_20261115.zip` (about 11.10 GiB).

These large packages and individual movement records are not Git files. Obtain them from the group through the same course-approved transfer channel used for handoff; there is no public download URL in this repository. The package provenance is recorded in `provenance/member_b_package_summary.json`. Obtaining the original GeoLife data and running A is the alternative for rebuilding the underlying inputs, but creates a new experiment identity.

Extract both ZIPs into the same parent directory to obtain one `GeoLife_MemberB_Handoff/` tree. Validate in that extracted tree first:

```powershell
# Only the B CPU requirements are needed for these commands.
& $projectPython tools/member_b_package_check.py --root 'D:\handoff\GeoLife_MemberB_Handoff' --mode all --smoke
```

Restore the **data/model artifacts**, preserving their relative paths under this repository's `GeoLife/`:

| Required path | Purpose |
|---|---|
| `现实关联/artifacts/reference_manifest.csv` | Persistent IDs, users and labels |
| `project/artifacts/member_a/windows.npz` | Original-window timestamps and local coordinates |
| `project/artifacts/member_a/support.npz` | Common and per-condition eligibility |
| `project/artifacts/member_a/fold_manifest.json` | Training/test users |
| `project/artifacts/member_a/run_04d2f3dee1b64ac6/run_manifest.json` | Archived run identity |
| `.../folds/<user>/rf.joblib` and `models.json` | The matching held-out-user RF |
| `.../folds/<user>/<condition>_seed0.csv.gz` and its receipt | Four original baseline predictions |
| `project/artifacts/member_a/features_be719a48cdedcb9e/core_a.npy` | Optional reuse of clean statistics |

Use the handoff inventory and checksums rather than guessing user/model associations. There are 54 original folds but only 48 RF models are needed for the common-support B analysis; six users have no common windows. B does not need GRU weights or the clean sequence/mask cache. Do not overwrite the repository's source code with copies from the handoff ZIP merely to restore data; Git newline normalization can alter raw source hashes even when the code is equivalent.

The supplied B scripts intentionally refer to the archived A run and the published A baseline tables. For recomputation against those exact inputs:

```powershell
# B outputs use fixed paths: back up member_b/outputs before deliberately rerunning.
& $projectPython member_b/run_member_b_drift.py
& $projectPython member_b/run_member_b_permutation.py
```

Their working caches are under ignored `work/`. Do not reuse them across run identities. Applying B to a fresh A run requires explicitly selecting the new models/inventory and new baseline/output paths in a separate analysis; do not silently relabel new models as the archived RUN or overwrite the published B evidence. The original B scripts and provenance remain unchanged in this upload.

## 7. Files intentionally excluded and report status

Git contains code, tests, locked requirements, aggregate results, figures and provenance. It excludes raw trajectories, `points.csv`, packed individual windows, installed dependencies, model weights, per-window predictions and large ZIPs. A model ZIP is over GitHub's ordinary file limit; upload of source code does not imply upload of those artifacts.

`docs/report_draft.pdf` is the supplied report snapshot, unchanged. See `docs/report_review_notes.md` for pending factual, presentation and length corrections. This branch supplies missing reproduction files; it does not claim that the PDF was revised or that new model results were generated.
