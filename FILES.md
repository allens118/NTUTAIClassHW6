# File Guide

## Core Files

- `train_diffusion.py`
  Main training entry point for the DDPM implementation.

- `build_report_figures.py`
  Rebuilds report figures from saved experiment summaries.

- `requirements.txt`
  Dependency list for the Python environment.

- `報告.md`
  Final assignment report with methodology, experiments, and analysis.

- `README.md`
  Repository overview and quick project summary.

## Result Directory

- `outputs_compare_1000/`
  Main long-horizon experiment outputs used for the final report.

### Important files inside `outputs_compare_1000`

- `training_summary.json`
  Numerical summary of training and validation results.

- `loss_curve_curated_v2.png`
  Curated loss figure with milestone epochs and special-point annotations.

- `milestone_metrics.png`
  Milestone comparison chart for training loss, validation loss, and gap.

- `epoch_comparison_v2.png`
  Combined comparison figure of generated sample grids across selected epochs.

## Not Pushed

- `.venv/`
  Local Python virtual environment.

- `crop_2020_img/`
  Local dataset folder.

- `outputs_bench*`
  Temporary benchmark folders.

- `outputs_compare_1000/checkpoints/`
  Large model checkpoint files excluded to keep the repository lightweight.

- `outputs_compare_1000/samples/`
  Intermediate sample images excluded because the main comparison figures are already included.
