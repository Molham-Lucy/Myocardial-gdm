# Experiment 03 — external and clinical-information evaluation

**Status: planned; not yet implemented.**

This experiment should begin only after an external patient cohort, suitable
expert labels/reference measurements, and a pre-specified evaluation protocol
are available.

Prerequisites:

- External-center test data held out from model selection.
- Expert assessment of pairing, acquisition, image quality, and registration.
- A defined intended use: weighted-image synthesis, quantitative T2 estimation,
  or a downstream diagnostic task.
- For diagnostic classification, expert infarct/scar/edema labels and
  patient/site-separated evaluation.
- Predefined acceptance criteria, uncertainty/abstention behavior, subgroup
  analyses, and appropriate clinical and regulatory oversight.

Do not interpret generated images as acquired scans or quantitative T2 maps.
See [`docs/study_design.md`](../../docs/study_design.md) for the current claims
boundary and proposed analysis.
