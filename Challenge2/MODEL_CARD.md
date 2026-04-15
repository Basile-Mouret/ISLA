# Model Card

## Final Submission Strategy

The final submission is a subject-specific top3 weighted vote.

- One model is trained per candidate row in `subject_models.md`.
- For each subject, predictions from top-3 candidates are combined with weighted majority vote.
- Weights are the CV mean accuracies from the corresponding search stage.

## Per-Subject Setup

- `A`: `fbcsp_broad_c6_k16_lda` + `mne_fbcsp_lda` around late windows.
- `B`: `riemann_fgmdm_8_30_lwf` on late windows.
- `C`: base (not fine-tuned), CSP/FBCSP mix.
- `D`: base (not fine-tuned), Riemann TS LR windows.
- `E`: fine-tuned FBCSP dense/broad late windows.
- `F`: fine-tuned FBCSP broad plus base FBCSP.

## Why C and D are not fine-tuned

During earlier iterations, `C` and `D` were already very strong and stable in internal CV, so tuning effort focused on `A`, `B`, `E`, and `F`.

## Scripts

- `produce_submission.py`: minimal final submission generator from `subject_models.md`.
- `tune_subjects.py`: lightweight fine-tuning utility for selected subjects.

## Output

Running `produce_submission.py` produces:

- folder: `submissions/final_top3_submission/`
- zip: `submissions/final_top3_submission.zip`
