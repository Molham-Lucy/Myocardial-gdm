# Study design and evidence plan

## Research question

Can an LGE cardiac MRI support prediction of paired T2-weighted and static
C0/bSSFP images, and do the predicted images preserve clinically relevant
information on patients and centers not used for model development?

Treat this as a feasibility question until the data, external validation, and
expert review support stronger conclusions. Synthesized weighted images are
not quantitative T2 maps and are not substitutes for acquired images.

## Experiments and progression

1. **Diffusion feasibility experiment** — retained as the first experiment in
   [`experiments/01_diffusion/`](../experiments/01_diffusion/). The initial
   epsilon-prediction runs completed optimization but did not produce visually
   credible T2/bSSFP outputs. A controlled velocity-prediction rerun reduced
   mean MAE versus the prior epsilon sampler on one central resized slice from
   0.861 to 0.292 for C0 and 0.948 to 0.258 for T2 (15 validation cases per
   modality), but predictions remain blurry and SSIM remains low. A subsequent
   aspect-preserving preprocessing ablation made only small changes in a
   same-noise comparison with Run 3: C0 MAE/SSIM/PSNR were 0.287/0.320/13.35
   versus 0.289/0.302/13.18, and T2 values were 0.258/0.345/13.89 versus
   0.257/0.336/13.88. Visual quality remains insufficient for clinical use.
   The 214-pair geometry audit found no shape or affine mismatches, but matching
   grids do not prove anatomical registration; expert review remains necessary.
   The notebooks, reports, prediction volumes, checkpoints, histories, and
   comparison artifacts are retained as development evidence.
2. **Direct supervised baseline** — [`experiments/02_supervised/`](../experiments/02_supervised/)
   trains independent LGE-to-T2 and LGE-to-C0 U-Nets and compares against
   simple image baselines. Initial predictions are blurred; metrics are
   preliminary and are not clinical validation.
3. **External-center and clinical-information evaluation (planned)** —
   [`experiments/03_external_clinical_evaluation/`](../experiments/03_external_clinical_evaluation/)
   requires additional independent sites and expert annotations before
   implementation or clinical claims.

## Current dataset and split limitations

The current MyoPS pairing pipeline reports 80 paired T2 and 104 paired C0
training subjects, with 15 validation subjects per target. The reviewed
validation sample is from `AnonymousCenter`; this does not establish
generalization to unseen centers. A research-ready analysis must tabulate
patient-level counts by site, acquisition/protocol, modality availability, and
quality before assigning splits.

Keep every slice from a patient in the same split. For a cross-center claim,
reserve one or more complete centers for testing. If the available data do not
contain enough distinct centers or cases for a locked test set, report that
limitation and collect appropriate external data rather than tuning on the
validation set and relabeling it as a test set.

## Evaluation plan

- Review pair identity, image quality, orientation, registration, and
  acquisition differences with cardiac MRI domain experts. Matching NIfTI
  shapes and affines—as found in the current 214-pair audit—is not evidence of
  accurate anatomical registration or sequence-phase correspondence.
- Report patient-level MAE, SSIM, and PSNR for weighted-image fidelity,
  alongside multi-case and multi-slice visual review. Use the same patients,
  preprocessing, and splits for every method. Do not compare diffusion
  noise-prediction loss directly with image-domain metrics.
- Compare with explicitly defined simple baselines and relevant published
  methods reproduced under the same evaluation protocol. Report all outcomes
  and uncertainty; do not claim superiority unless supported by a fair,
  pre-specified comparison.
- If the intended endpoint is quantitative T2, require acquired quantitative
  T2 reference measurements and agreement/error analyses suitable for that
  endpoint. Similarity to a T2-weighted image cannot validate quantitative
  T2.
- Add infarct/scar/edema classification only after suitable expert labels and
  patient/site-level sample sizes exist. Compare acquired modalities alone
  against acquired-plus-synthesized inputs, with sensitivity, specificity,
  calibration, subgroup and failure analyses. Audit for findings invented or
  removed by synthesis.
- Define intended use, clinically acceptable errors, uncertainty handling,
  abstention rules, and prospective study requirements with clinical,
  statistical, privacy, and regulatory collaborators before clinical claims.

## Current claims boundary

The current supervised results support only a small-sample feasibility
baseline: its mean validation MAE was lower than the training-mean baseline in
the current split. Blurred example predictions, 15 validation subjects per
modality, and no held-out-center evaluation mean the project does not yet
establish reliable modality prediction, preserved diagnostic findings,
clinical utility, or state-of-the-art performance. Run 3's lower MAE than the
previous epsilon sampler demonstrates improvement within this diffusion
pipeline, but its low SSIM and visibly mottled predictions do not support a
claim of credible synthesized anatomy. Run 4's aspect-preserving preprocessing
had mixed, small image-metric effects and did not resolve the visual-quality
problem.
