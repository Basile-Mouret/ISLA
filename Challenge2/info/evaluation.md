# Evaluation

## Metric

Submissions are evaluated using classification accuracy - the proportion of test trials correctly classified:

```text
Accuracy = (1 / n) * sum_{i=1..n} [y_hat_i = y_i]
```

where `y_hat_i` is the predicted class and `y_i` is the true class for trial `i`.

The final score is the average accuracy across all six subjects:

```text
Score = (1 / 6) * sum_{s in {A,B,C,D,E,F}} Accuracy_s
```

Higher score is better. The leaderboard is ranked in descending order of average accuracy.

## Submission Format

You must submit one CSV file per subject, named `subject_A_y_pred.csv` through `subject_F_y_pred.csv`. Each file must contain a single column `y_pred` with the predicted class for each trial in `X_test`, in the same order.

Expected format (example for subject A):

```text
y_pred
left_hand
right_hand
left_hand
...
```

- Each file must contain exactly 60 rows (excluding the header), matching the 60 test trials.
- Each value must be either `left_hand` or `right_hand`.
- The column must be named `y_pred`.

All prediction files `subject_A_y_pred.csv` through `subject_F_y_pred.csv` should be put in a single zip file that will be submitted to the Codabench platform.

## How Scoring Works

For each subject, accuracy is computed and then averaged across subjects:

```python
import numpy as np
import pandas as pd

subjects = ['A', 'B', 'C', 'D', 'E', 'F']
accuracies = []

for subject in subjects:
    y_pred = pd.read_csv(f'subject_{subject}_y_pred.csv')['y_pred'].values
    y_test = np.load(f'subject_{subject}_y_test.npy')
    accuracies.append(np.mean(y_pred == y_test))

score = np.mean(accuracies)
```
