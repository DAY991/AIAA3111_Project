# GeoLife transportation mode classification under data degradation

AIAA3111 Data Mining course project, Group 16. We classify GeoLife GPS trajectory windows as **walk, bike, bus, car, or subway** and compare two fixed pipelines—statistical features with a Random Forest (RF) and sequence features with a GRU—when observations become sparse or coordinates receive synthetic noise. The study evaluates held-out users and reports both classification performance and the fraction of windows that remain usable.

## Research question and protocol

Under the same clean training data and leave-one-user-out (LOUO) split, how do the fixed RF and GRU pipelines behave across paired sampling and added-position-noise conditions? Which results are accompanied by loss of window coverage or changes in the RF's input features?

The frozen protocol is [v1.0.0](现实关联/degradation_config.yaml): five merged classes; non-overlapping 120-second windows within known same-label runs; at least 10 points and a maximum internal gap of 30 seconds. `ds0` retains GeoLife's native irregular sampling. `ds5` and `ds10` retain the first point per 5- or 10-second time bucket. Added x/y noise is independent Gaussian with per-axis standard deviation `n0`, `n5`, `n10`, or `n20` metres; `n0` means **no added noise**, not error-free positioning. The 30-second thinning conditions measure a coverage failure boundary. Degradation applies to test windows only. The shared `deployment_protocol.py` pairs noise by seed, persistent window ID, and original point position.

The primary model comparison uses the **same 71,941 windows from 48 users** that remain usable in all 12 main conditions. The clean reference has 90,120 usable windows from 54 users. Coverage and condition-specific performance are reported separately; common-support scores do not represent every original window. Each LOUO fold trains on the other users' clean reference windows. Metrics are pooled out-of-fold predictions with a fixed five-class Macro-F1 definition, plus accuracy and per-class F1. The main grid uses noise seed 0; `ds10_n10` also uses seeds 1 and 2.

## Results in brief

On the common support, the clean RF/GRU Macro-F1 values are **0.6344/0.6605**. GRU has the higher point estimate in all 12 main grid conditions, but its advantage varies non-monotonically. At `ds5_n10` the difference is 14.28 percentage points; at `ds10_n10` the seed-0 difference is 2.66 points, close to the clean gap of 2.61 points. At 10-second thinning, 71,941 of 90,120 clean reference windows remain usable (**79.83%**). At 30-second thinning, none satisfy the frozen usability rule, so classification F1 is undefined rather than zero.

Member B recomputed four paired conditions (`ds0_n0`, `ds10_n0`, `ds0_n20`, `ds10_n20`) on those same 71,941 windows. Added 20 m per-axis noise strongly changes dense-sampling speed estimates; thinning reduces that particular simulated noise response while also removing motion detail. In grouped RF permutation analysis, clean speed-group importance is **0.3572 F1**, compared with **0.0055** at `ds0_n20` and **0.0720** at `ds10_n20`. These are dependencies of fixed RF models, not causal effects or real-device deployment validation. See the [A results report](docs/member_a_report.md) and [B report section](member_b/outputs/member_b_report_section.md) for methods, numbers, and limits.

## Repository layout

```text
docs/
  Proposal.docx                    Submitted research proposal
  完整优化方案与分工.md              Frozen experiment plan and A/B responsibilities
  member_a_report.md               A's final grid report
  reproduction.md                  Data preparation, fresh A runs and archived B inputs
  historical/                     Early EDA, pilot, 3-user baseline, and progress notes
project/research/                 Frozen data, feature, model, and protocol code
project/scripts/                  Raw-data preparation, A evaluation, summary and figures
project/tests/                    Synthetic parser and paired-protocol regression tests
tools/run_member_a.py              Portable A launcher using the current Python environment
tools/verify_member_a.py           Independent full-data archive verification
tools/member_b_bootstrap.py       B's CPU-only import helper
现实关联/
  degradation_config.yaml         Frozen experiment settings
  outputs/                       A's small aggregate CSV/JSON tables and PNG figures
member_b/
  run_member_b_drift.py            Four-condition paired feature drift
  run_member_b_permutation.py      Within-user grouped RF permutation
  outputs/                       B's aggregate CSV tables, evidence, and report section
provenance/member_b_master_manifest.json
requirements.txt                 Locked CPU dependencies for B analysis
requirements-member-a.txt          A training and plotting dependencies
../.gitignore                    Excludes data, model weights, caches, and environments
```

The A output folder includes `grid_primary_common.csv` (primary paired comparison), `grid_primary_condition.csv` (each condition's surviving windows), `grid_results.csv` (full grid), `per_class_robustness.csv`, coverage tables, the key-seed summary, interaction contrasts, figures, and verification JSON. B outputs include 68 feature-drift rows, 24 group-drift rows, 240 permutation-repeat rows, 24 permutation summaries, 12 scenario mappings, and a three-paper evidence table. The repository manifest embeds B's original input and fold-model hashes and hashes the packaged small B outputs. A's handoff JSON records the original run identity and hashes of large local inputs, which are intentionally absent here.

## Reproduction and data availability

See the [reproduction guide](docs/reproduction.md) for a Python 3.12 environment, the original GeoLife source, exact artifact locations, and commands. Use [requirements.txt](requirements.txt) for CPU-only B work, or [requirements-member-a.txt](requirements-member-a.txt) for A training/plots after selecting a CPU or CUDA PyTorch wheel. **The published aggregate results are available immediately; full-data computation still requires the source data or compatible handoff artifacts, which are not Git files.** This update adds the missing A preparation, training, evaluation, verification, plotting and test entry points. It does not claim that a new full 54-fold experiment was run during publication.

From this `GeoLife/` directory, the full-data A route is:

```powershell
python project/scripts/prepare_geolife.py --raw-root 'D:\datasets\Geolife Trajectories 1.3\Data'
python tools/run_member_a.py --mode prepare --output-dir work/reproduced_a
python tools/run_member_a.py --mode evaluate --threads 8 --output-dir work/reproduced_a --finalize-figures
python project/scripts/make_deployment_figures.py --input-dir work/reproduced_a
python tools/verify_member_a.py --input-dir work/reproduced_a
python project/scripts/summarize_member_a.py --input-dir work/reproduced_a
```

The data script creates its output directory and refuses unrequested overwrites. Fresh A outputs are kept under ignored `work/reproduced_a/` instead of replacing archived tables. A fresh environment can create a new run identity. The preserved B permutation script deliberately binds to the original A run and its RF receipts; read the guide before attempting B recomputation against newly trained models. Original large handoff ZIPs and their checksums are described in `provenance/member_b_package_summary.json`.

The original GeoLife dataset should be obtained from its authorized source under its applicable terms. Do not commit raw `.plt` trajectories, the processed `points.csv`, `reference_manifest.csv`, `windows.npz`, `support.npz`, `rf.joblib`, `gru.pt`, compressed per-window prediction CSVs, the local Python environment, or the original handoff directories. These files are large and can contain sensitive location traces or derived individual movement records. `.gitignore` is a guard; **inspect `git status` before every push**. Keep large private artifacts in a controlled local or institution-approved location. If access to them is required for assessment, arrange it through the course's approved channel rather than GitHub.

## Academic and interpretation notes

The [proposal](docs/Proposal.docx), experiment plan, source code, aggregate data, provenance, and [literature evidence](member_b/outputs/literature_evidence.csv) are included so that methods and claims can be checked. The three cited studies are used within the scope documented in that evidence table; the 12 [scenario mappings](member_b/outputs/scenario_mapping.csv) are illustrative stress cases, not calibrated battery-saving or urban-canyon measurements. The comparison is between these particular RF and GRU pipelines. Windows were built from known same-label segments, so this is an offline classification evaluation, not an end-to-end live mode-transition detector.

**Disclosure for the final course submission:** AI assistance was used to help develop, inspect, organize, and draft parts of the code and written analysis. Group members must verify all reported numbers, source attributions, and claims, and adapt this disclosure to the actual work performed under the course requirements. Do not present generated text as independently authored experimental evidence.

## Before publishing

1. Review `docs/Proposal.docx` and the reports for participant names or other details the group does not wish to publish.
2. Run `git status --short` and inspect staged files; verify that no raw data, model weights, per-window caches, or local environment folders appear.
3. Add the final report and poster only after they are complete and checked against the aggregate tables.
