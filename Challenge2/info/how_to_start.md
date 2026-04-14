# How to Start

## 1. Download the Data

Once registered, download the competition data from the challenge link. You should download and store these files into a `data` folder.

```text
data/
    subject_A_X_train.npy
    subject_A_y_train.npy
    subject_A_X_test.npy
    subject_B_X_train.npy
    ...
    subject_F_X_test.npy
```

## 2. Load the Data

```python
import numpy as np

X_train = np.load("data/subject_A_X_train.npy")  # shape: (140, 64, 1537)
y_train = np.load("data/subject_A_y_train.npy")  # shape: (140,)
X_test = np.load("data/subject_A_X_test.npy")    # shape: (60, 64, 1537)
```

Each trial is a matrix of shape `(64 channels, 1537 time points)`. Labels are strings: `'left_hand'` or `'right_hand'`.

## 3. Extract Features and Build a Model

Raw EEG trials cannot be fed directly to most classifiers - you first need to extract features. A simple and effective baseline uses the log-variance of each channel across time, followed by Linear Discriminant Analysis (LDA):

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def temporal_variance_log(X):
    # X shape: (n_epochs, 64, T) -> output: (n_epochs, 64)
    var = np.nan_to_num(np.var(X, axis=-1), nan=1e-10, posinf=1e-10, neginf=1e-10)
    return np.log(np.clip(var, 1e-10, None))


pipeline = Pipeline([
    ('log_var', FunctionTransformer(temporal_variance_log)),
    ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
])

pipeline.fit(X_train, y_train)
```

## 4. Generate Predictions

```python
import pandas as pd

y_pred = pipeline.predict(X_test)
pd.DataFrame({'y_pred': y_pred}).to_csv("subject_A_y_pred.csv", index=False)
```

The output file `subject_A_y_pred.csv` should look like:

```text
y_pred
left_hand
right_hand
left_hand
...
```

Each file must contain exactly 60 rows (excluding the header).

## 5. Submit

Create one prediction file per subject (`subject_A_y_pred.csv` through `subject_F_y_pred.csv`), put them all in a single zip file, and upload it on the competition platform.

## Summary

The minimal working pipeline (for all subjects) is:

```python
import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def temporal_variance_log(X):
    var = np.nan_to_num(np.var(X, axis=-1), nan=1e-10, posinf=1e-10, neginf=1e-10)
    return np.log(np.clip(var, 1e-10, None))


pipeline = Pipeline([
    ('log_var', FunctionTransformer(temporal_variance_log)),
    ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
])

for subject in ['A', 'B', 'C', 'D', 'E', 'F']:
    X_train = np.load(f'data/subject_{subject}_X_train.npy')
    y_train = np.load(f'data/subject_{subject}_y_train.npy')
    X_test = np.load(f'data/subject_{subject}_X_test.npy')

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    pd.DataFrame({'y_pred': y_pred}).to_csv(f'subject_{subject}_y_pred.csv', index=False)
```
