# Final Subject Models (0.87 strategy)

This file defines the exact per-subject top3 weighted-vote strategy used by `produce_submission.py`.

- Subjects `A`, `B`, `E`, and `F` use fine-tuned candidates.
- Subjects `C` and `D` keep base (not fine-tuned) candidates.

| subject | rank | model | start | stop | weight | note |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1 | fbcsp_broad_c6_k16_lda | 640 | 1472 | 0.7929386370644466 | fine-tuned |
| A | 2 | mne_fbcsp_lda | 768 | 1536 | 0.7505396238051186 | base |
| A | 3 | fbcsp_broad_c6_k16_lda | 896 | 1472 | 0.7497687326549491 | fine-tuned |
| B | 1 | riemann_fgmdm_8_30_lwf | 704 | 1536 | 0.7711995066296639 | fine-tuned |
| B | 2 | riemann_fgmdm_8_30_lwf | 768 | 1537 | 0.7641073080481036 | fine-tuned |
| B | 3 | riemann_fgmdm_8_30_lwf | 768 | 1536 | 0.7641073080481036 | fine-tuned |
| C | 1 | mne_csp_8_30_lda | 512 | 1280 | 0.9643848288621648 | base |
| C | 2 | mne_csp_8_30_lda | 512 | 1536 | 0.9498920752389762 | base |
| C | 3 | mne_fbcsp_lda | 256 | 1280 | 0.9497378970089424 | base |
| D | 1 | riemann_ts_lr_6_35 | 512 | 1536 | 0.9569842738205366 | base |
| D | 2 | riemann_ts_lr_6_35 | 512 | 1280 | 0.942645698427382 | base |
| D | 3 | riemann_ts_lr_6_35 | 768 | 1536 | 0.9355534998458216 | base |
| E | 1 | fbcsp_dense_c6_k20_lda | 320 | 1344 | 0.8572309589885908 | fine-tuned |
| E | 2 | fbcsp_broad_c6_k16_lda | 320 | 1216 | 0.857076780758557 | fine-tuned |
| E | 3 | fbcsp_dense_c6_k20_lda | 320 | 1280 | 0.8428923835954363 | fine-tuned |
| F | 1 | fbcsp_broad_c6_k16_lda | 512 | 1536 | 0.8068146777674992 | fine-tuned |
| F | 2 | mne_fbcsp_lda | 512 | 1536 | 0.799722479185939 | base |
| F | 3 | fbcsp_broad_c6_k16_lda | 512 | 1537 | 0.799568300955905 | fine-tuned |
