# Paired supervised baseline

`paired_supervised_baseline.py` audits paired validation images and trains a
direct LGE-to-image U-Net separately for T2 and C0 (static bSSFP). The baseline
uses 96 × 96 normalized 2D slices, patient-level train/validation separation,
Huber loss, gradient clipping, early stopping, and validation-loss checkpointing.

For an interactive workflow with separate setup, data review, training, metrics,
and prediction cells, open
[`Myocardial Paired Supervised Baseline.ipynb`](Myocardial%20Paired%20Supervised%20Baseline.ipynb).

From the repository root, run:

```bash
python paired_supervised_baseline.py \
  --data-root .. \
  --output-dir experiments/02_supervised/results \
  --epochs 40 \
  --image-size 96 \
  --batch-size 8 \
  --modalities T2 C0
```

The data root is the directory containing `Myo_train/MyoPS_train` and
`Myo_val/MyoPS_val`. To generate only the geometry tables and multi-case
alignment panels without training:

```bash
python paired_supervised_baseline.py \
  --data-root .. \
  --output-dir experiments/02_supervised/results \
  --audit-only
```

The output directory contains:

- `alignment_T2.png` and `alignment_C0.png`: central-slice image pairs and
  red/green overlays for a deterministic sample of validation subjects.
- `alignment_metadata.csv`: source/target shapes, voxel sizes, affine
  differences, and whether the target was resampled to the LGE grid.
- `lge_to_<modality>_best.keras`, `lge_to_<modality>_history.csv`, and
  `training_<modality>.png`: best checkpoint and training history for each
  independently trained target.
- `predictions_<modality>.png`: validation LGE, reference, and predicted
  central slices for several subjects.
- `validation_metrics_by_case.csv`: subject-level MAE, SSIM, and PSNR for the
  U-Net, an LGE-copy baseline, and a constant training-target-mean baseline.
- `validation_metrics_summary.csv`: equal-subject-weight averages of those
  metrics by target and method.

These image metrics and visual panels are research diagnostics, not clinical
validation or quantitative T2 measurements. Matching voxel geometry is not
proof of anatomical registration; review the overlays and source images before
interpreting voxelwise prediction scores.

The initial supervised run is retained in `results/`. The separate run
archives under
[`../01_diffusion/runs/`](../01_diffusion/runs/) document the earlier diffusion
experiment and are intentionally preserved as the preceding experiment.
