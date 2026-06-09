# NTUT AI Class HW6

## Project Overview
This project implements a diffusion-model pipeline for facial illustration generation using the Pixiv face dataset.

The repository includes:

- PyTorch-based DDPM training code
- GPU-compatible training configuration
- Bilingual report documentation
- Report-ready visualization figures

## Main Files

- `train_diffusion.py`
  Main DDPM training script. Handles preprocessing, model construction, training, validation, sampling, checkpointing, and loss visualization.

- `build_report_figures.py`
  Rebuilds curated report figures from saved training summaries, including milestone comparisons and annotated loss curves.

- `requirements.txt`
  Python dependency list for the project.

- `報告.md`
  Final assignment report, including methodology, experiments, results, and analysis.

## Included Result Files

- `outputs_compare_1000/loss_curve_curated_v2.png`
  Curated long-horizon loss curve with milestone ticks and annotated special points.

- `outputs_compare_1000/milestone_metrics.png`
  Comparison figure for milestone training loss, validation loss, and generalization gap.

- `outputs_compare_1000/epoch_comparison_v2.png`
  Combined visualization of generated sample grids at selected epochs.

- `outputs_compare_1000/training_summary.json`
  Numerical summary of the long-horizon compact experiment.

## Not Included

- Dataset folder `crop_2020_img`
- Virtual environment `.venv`
- Temporary benchmark outputs
- Large checkpoint files

## Recommended Command

```powershell
python train_diffusion.py --epochs 20 --batch-size 32 --image-size 64 --base-channels 64 --diffusion-steps 300 --output-dir outputs_full --device cuda
```
