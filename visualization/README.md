# Remaining visualization generator

This folder contains the remaining poster visualizations for the GeoLife project. It reads aggregate CSV files under `GeoLife/` but does not modify or write to the `GeoLife/` folder.

## Generate

```bash
python -m pip install -r visualization/requirements.txt
python visualization/generate_remaining_visualizations.py
```

Generated PNG files are written to `visualization/outputs/`:

- `project_method_flowchart.png`: project method pipeline
- `class_distribution.png`: clean reference window counts by transportation mode
- `rf_speed_group_importance.png`: fixed-RF speed-group permutation importance
- `mean_speed_feature_drift.png`: paired mean-speed drift under 20 m noise
- `clean_per_class_performance.png`: clean-condition RF and GRU F1 by class
- `noise_seed_uncertainty.png`: 10 s + 10 m Macro-F1 across three noise seeds

The script uses 300 dpi output and reads only the following aggregate records:

- `GeoLife/现实关联/outputs/per_class_robustness.csv`
- `GeoLife/现实关联/outputs/grid_key_seed_summary.csv`
- `GeoLife/member_b/outputs/group_permutation_summary.csv`
- `GeoLife/member_b/outputs/feature_drift_by_feature.csv`
