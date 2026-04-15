# Hybrid Submission: Top3 Vote With Fine-Tuned E

## Goal

Keep the previously strong `mne_pyriemann_finetune_subject_top3_vote` predictions for subjects `A`, `B`, `C`, `D`, and `F`, but replace subject `E` with the newer fine-tuned prediction that came from the `E`-only fine-tune run.

## Source Mapping

- `A`: `submissions/mne_pyriemann_finetune_subject_top3_vote/subject_A_y_pred.csv`
- `B`: `submissions/mne_pyriemann_finetune_subject_top3_vote/subject_B_y_pred.csv`
- `C`: `submissions/mne_pyriemann_finetune_subject_top3_vote/subject_C_y_pred.csv`
- `D`: `submissions/mne_pyriemann_finetune_subject_top3_vote/subject_D_y_pred.csv`
- `E`: `submissions/mne_pyriemann_finetune_subject_best/subject_E_y_pred.csv`
- `F`: `submissions/mne_pyriemann_finetune_subject_top3_vote/subject_F_y_pred.csv`

## Output

- Directory: `submissions/mne_pyriemann_finetune_top3_with_E_finetuned/`
- Zip: `submissions/mne_pyriemann_finetune_top3_with_E_finetuned.zip`

## Why This Hybrid Exists

- The earlier `mne_pyriemann_finetune_subject_top3_vote.zip` performed better on the public leaderboard than later ensemble submissions.
- The later `E` fine-tuning showed a much stronger internal CV result for subject `E`.
- This hybrid tries to preserve the previously successful predictions for `A/B/C/D/F` while only swapping in the improved `E` prediction.

## Fine-Tune Preservation Fix

The fine-tune runner was patched so future partial runs do not wipe untuned-subject results from `submissions/mne_pyriemann_finetune_cv_results.csv`.

Before the patch:

- Running `--subjects E` replaced the shared fine-tune CSV with only `E` rows.

After the patch:

- Existing untuned-subject rows are preserved.
- Only the subjects targeted by the current run are replaced in the aggregate fine-tune CSV.
