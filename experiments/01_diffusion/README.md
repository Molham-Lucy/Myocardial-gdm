# Experiment 01 — LGE-conditioned diffusion

This is the original and first modeling experiment in the project. It remains
available for reproducibility and comparison. The notebook now also provides
an isolated Run 3 using velocity (`v`) prediction to address instability in
the earlier epsilon-prediction sampler.

## Run history

1. **Run 1** — training completed, but the initial generation showed severe
   sampling instability and poor image quality. The written report is
   [`runs/Run1.txt`](runs/Run1.txt); the rendered notebook snapshot is
   [`runs/Run%201.html`](runs/Run%201.html).
2. **Run 2** — best-checkpoint sampling and loss behavior were investigated.
   The generated T2 remained blurry/mottled and not anatomically convincing.
   The written report is [`runs/Run%202.txt`](runs/Run%202.txt); the rendered
   notebook snapshot is [`runs/Run%202.html`](runs/Run%202.html).

Both HTML files and both text reports are retained as historical records; they
are not replaced when the notebook is rerun.

**Historical-record note:** `runs/Run 2.txt` reports a validation-loss spike
to 63.6 at epoch 98, but the currently stored legacy
`results/history/myocardial_training_history.csv` is smooth and reaches its
best validation loss at epoch 100. Treat that report and CSV as separate
snapshots until their exact checkpoint/run provenance is reconciled.

## Run 3 — velocity prediction

Run 3 keeps the previous data, schedule, U-Net, optimizer, and resolution, and
changes the prediction target from raw noise (`epsilon`) to velocity (`v`).
The velocity parameterization gives a more stable clean-image estimate at the
highest-noise timestep, where epsilon prediction required division by a very
small value. This is a controlled test of that numerical issue, not a
guarantee of better anatomy or clinical performance.

Run 3 writes to distinct files so the earlier run remains available:

- `results/checkpoints/myocardial_conditional_diffusion_v_prediction.keras`
- `results/history/myocardial_training_history_v_prediction.csv`
- `results/figures/ddpm_training_history_v_prediction.png`
- `results/figures/run_03_validation_comparison_C0.png` and
  `run_03_validation_comparison_T2.png`
- `results/figures/run_03_ddim_steps_C0.png` and
  `run_03_ddim_steps_T2.png`
- `results/predictions/run_03_v_prediction/`
- `results/metrics/run_03_velocity_comparison_by_case.csv` and
  `run_03_velocity_comparison_summary.csv`
- `results/metrics/run_03_ddim_steps_by_case.csv` and
  `run_03_ddim_steps_summary.csv`

The notebook restores its best validation-loss checkpoint before generating
predictions. Its comparison cell evaluates one central slice per held-out
validation case for both the prior epsilon model and Run 3 using the same
starting noise. Raw epsilon- and velocity-prediction losses are not directly
comparable. The image-domain comparison is a limited research diagnostic;
it measures one central resized slice per subject. On the current validation
set, 100 DDIM steps performed slightly better than 250, 500, or 1,000 steps
for both targets; extra steps did not improve the images. Run 3 has completed
training and these held-out comparisons, but the generated anatomy remains
blurry and is not clinically usable. At 100 steps, mean per-case MAE improved
from 0.861 to 0.292 for C0 and from 0.948 to 0.258 for T2 versus the previous
epsilon sampler; mean SSIM was 0.228 for C0 and 0.250 for T2. These numbers
are specific to one central 128 × 128 resized slice per validation subject,
and do not establish diagnostic fidelity.

## Current artifacts

- [`Myocardial Diffusion Model.ipynb`](Myocardial%20Diffusion%20Model.ipynb):
  experiment workflow. Rerun from setup to use the organized output paths.
- `results/checkpoints/myocardial_conditional_diffusion.keras`: saved
  diffusion checkpoint from the previous run.
- `results/history/myocardial_training_history.csv`: per-epoch previous run
  metrics.
- `results/figures/`: variance schedule and training history plots.
- `results/predictions/Case1001/`: generated C0 and T2 NIfTI outputs.

The existing run results are development evidence, not evidence of clinical
validity. In particular, low noise-prediction loss did not translate to
convincing validation synthesis.
