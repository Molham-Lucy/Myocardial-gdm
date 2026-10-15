# MyoPS missing-modality synthesis study

This project investigates whether paired LGE cardiac MRI can support
prediction of missing T2-weighted and C0/static bSSFP images. It is an
experimental research workflow; current model outputs are not clinically
validated and synthesized T2-weighted images are not quantitative T2 maps.

## Experiment progression

| Order | Experiment | Purpose and current status |
|---|---|---|
| 01 | [LGE-conditioned diffusion](experiments/01_diffusion/Myocardial%20Diffusion%20Model.ipynb) | Retained as the first experiment. Run 3 velocity prediction improved held-out image metrics over the original epsilon sampler, but outputs remain blurry and are not clinically usable. Run 1/Run 2 artifacts and the separate Run 3 outputs are preserved under [01_diffusion](experiments/01_diffusion/). |
| 02 | [Direct supervised U-Net](experiments/02_supervised/Myocardial%20Paired%20Supervised%20Baseline.ipynb) | Paired direct image-to-image baseline with alignment review, checkpointing, per-subject metrics, and prediction panels. Initial scores improve over simple baselines, but predictions remain blurred and validation is small. See [02_supervised](experiments/02_supervised/). |
| 03 | [External/clinical-information evaluation](experiments/03_external_clinical_evaluation/README.md) | Planned, not implemented. Requires external-center data, expert review/labels, and a pre-specified clinical endpoint. |

The current evidence and claims boundary are in
[`docs/study_design.md`](docs/study_design.md).

## Repository layout

```text
myocardial-gdm/
├── experiments/
│   ├── 01_diffusion/
│   │   ├── Myocardial Diffusion Model.ipynb
│   │   ├── runs/                  # Preserved Run 1/Run 2 HTML and text records
│   │   └── results/               # Checkpoint, histories, figures, NIfTI predictions
│   ├── 02_supervised/
│   │   ├── Myocardial Paired Supervised Baseline.ipynb
│   │   ├── README.md
│   │   └── results/               # Checkpoints, histories, metrics, review panels
│   └── 03_external_clinical_evaluation/
├── docs/
│   ├── study_design.md
│   └── upstream/                  # Preserved inherited repository documentation
├── myocardial_data.py             # Shared NIfTI pairing and data pipeline
└── paired_supervised_baseline.py  # Supervised experiment implementation/CLI
```

The MyoPS data are external to this project tree and are not copied or
modified. Keep `Myo_train/MyoPS_train` and `Myo_val/MyoPS_val` beneath a common
data-root directory, or set `MYO_DATA_ROOT` to that directory. Both notebooks
search the current working directory and its parents for the data.

## Running the experiments

Use the project Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate myocardial-gdm
jupyter lab
```

Open Experiment 01 or 02 from its numbered folder. Run cells in order. The
notebooks locate the shared project code and dataset from the current working
directory and its parents. Training is intentionally a notebook action; the
third experiment is not yet available.

The supervised experiment can also be run from the repository root:

```bash
python paired_supervised_baseline.py \
  --data-root .. \
  --output-dir experiments/02_supervised/results \
  --epochs 40 \
  --image-size 96 \
  --batch-size 8 \
  --modalities T2 C0
```

## Interpretation and limitations

- The current split has 80 paired T2 and 104 paired C0 training subjects, and
  15 validation subjects per target. Validation so far does not establish
  performance on an unseen center.
- Matching image geometry does not guarantee anatomical registration.
- Image similarity scores do not establish preservation of infarct, scar, or
  edema findings. No downstream classifier experiment is implemented yet.
- Do not claim clinical reliability or state-of-the-art performance from the
  current data or results. See the [study design and evaluation plan](docs/study_design.md).
