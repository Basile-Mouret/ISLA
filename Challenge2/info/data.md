# Data

## Files

The challenge data is provided as NumPy (`.npy`) files, one set per subject. Each subject `S` has three files:

| File | Shape | Description |
| --- | --- | --- |
| `subject_S_X_train.npy` | `(140, 64, 1537)` | Training EEG trials |
| `subject_S_y_train.npy` | `(140,)` | Training labels |
| `subject_S_X_test.npy` | `(60, 64, 1537)` | Test EEG trials |

Files can be loaded, for instance, with:

```python
import numpy as np

data = {}
for subject in ['A', 'B', 'C', 'D', 'E', 'F']:
    data[f'subject_{subject}'] = {}
    data[f'subject_{subject}']['X_train'] = np.load(f'subject_{subject}_X_train.npy')  # shape: (140, 64, 1537)
    data[f'subject_{subject}']['y_train'] = np.load(f'subject_{subject}_y_train.npy')  # shape: (140,)
    data[f'subject_{subject}']['X_test'] = np.load(f'subject_{subject}_X_test.npy')    # shape: (60, 64, 1537)
```

## EEG Trials (`X_train`, `X_test`)

Each array has three dimensions: `(trials, channels, time points)`.

- Trials: 140 training trials and 60 test trials per subject.
- Channels: 64 EEG electrodes placed on the scalp.
- Time points: 1537 samples per trial (`dtype: float64`).

## Labels (`y_train`)

Each entry is a string indicating the motor imagery class performed during that trial:

- `'left_hand'` - subject imagined a left-hand movement
- `'right_hand'` - subject imagined a right-hand movement

The two classes are balanced (70 trials each per subject).

## Key Characteristics

- Per-subject data: each subject has their own train/test split. Models should be trained and evaluated independently per subject.
- 3D input: unlike tabular data, each trial is a matrix of shape `(64, 1537)`. Most classifiers require feature extraction (e.g. band power, covariance) before fitting.
- No missing values: all arrays are fully populated.
