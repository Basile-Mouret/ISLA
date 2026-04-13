The Epigenetic Clock Challenge
What is an Epigenetic Clock?

DNA methylation is a biochemical process in which a methyl group is added to cytosine bases in DNA, predominantly at CpG dinucleotide sites. These modifications play a crucial role in gene regulation without altering the underlying DNA sequence. Remarkably, DNA methylation patterns change in a predictable manner as we age, acting as a biological clock — the so-called epigenetic clock.

Pioneering work by Steve Horvath (2013) and Gregory Hannum et al. demonstrated that a relatively small subset of CpG sites can be used to predict an individual's chronological age with striking accuracy. The difference between predicted and actual age — known as epigenetic age acceleration — has since been linked to various health outcomes, mortality risk, and disease susceptibility.
Biological Context

This challenge is inspired by real-world data from the study GSE42861 (Liu & Feinberg, 2013), which profiled genome-wide DNA methylation in peripheral blood leukocytes using the Illumina HumanMethylation450 BeadChip. This array measures methylation levels at over 450,000 CpG sites across the human genome, providing a rich molecular portrait of each individual.

In the original study, DNA from blood samples was subjected to bisulfite conversion — a chemical treatment that converts unmethylated cytosines to uracil while leaving methylated cytosines unchanged — followed by hybridization to the array. The resulting methylation values (beta values, ranging from 0 to 1) quantify the proportion of methylated molecules at each CpG site.
The Challenge

You are provided with two datasets ([X_train, y_train] and X_test) containing:

    DNA methylation profiles measured across a subset of CpG sites from blood samples
    Chronological age of each individual (available only in the training set)

Your objective: build a predictive model in Python that learns the relationship between CpG methylation patterns and age from the training data, and accurately predicts the age of individuals in the test set (i.e. y_test).
Why Does This Matter?

Epigenetic clocks have become a cornerstone of aging research. Accurate age prediction from methylation data enables:

    Forensic applications — estimating the age of individuals from biological samples
    Clinical research — identifying patients who are aging faster or slower than expected (epigenetic age acceleration), which correlates with disease risk
    Drug development — evaluating the impact of therapeutic interventions on biological aging
    Public health — understanding the epigenetic effects of lifestyle, environment, and socioeconomic factors on aging trajectories

Suggested Approaches

This is fundamentally a regression problem. You are encouraged to explore various statistical and machine learning methods, such as:

    Penalized regression (Ridge, Lasso, Elastic Net)
    Support Vector Regression
    Random Forests / Gradient Boosted Trees
    Neural Networks
    Dimensionality reduction combined with regression (PCA + regression)

A key difficulty is that the number of features (CpG sites) is much larger than the number of samples — a classic high-dimensional setting (p >> n) — making regularization and feature selection critical.
Acknowledgements

The Epiclock challenge was originally created by Florent Chuffart and is available at https://github.com/fchuffar/starting_kit_epiclock1.0. His work was supported by the RIS (Réseau Inter-disciplinaire autour de la Statistique).
Chasuite
Competitions v1.6
Chahub
Chagrade
About
About
Github
Privacy and Terms
API Docs


Evaluation
Metric

Submissions are evaluated using the Root Mean Squared Error (RMSE) between the predicted ages and the true chronological ages:

RMSE=1n∑i=1n(y^i−yi)2
RMSE=n1​i=1∑n​(y^​i​−yi​)2
​

where:

    y^iy^​i​ is the predicted age for individual ii
    yiyi​ is the true chronological age for individual ii
    nn is the number of individuals in the test set

Lower RMSE is better. The leaderboard is ranked in ascending order of RMSE.
Public and Private Leaderboards

The test set is split into two parts:
Leaderboard 	Description
Public (Test RMSE) 	Computed on the first 100 samples of the test set. This score is visible during the competition.
Private (Private Test RMSE) 	Computed on the remaining samples of the test set. This score is hidden during the competition and revealed at the end.

The final ranking is determined by the Private Test RMSE.
Submission Format

Your submission must be a CSV file named y_pred.csv containing a single column age with the predicted ages for all individuals in X_test, in the same order.

Expected format:

age
45.2
62.7
38.1
...

    The file must contain exactly as many rows (excluding the header) as there are samples in X_test.
    Each value should be a numeric age prediction (float or integer).
    The column must be named age.

How Scoring Works

The scoring program reads your y_pred.csv and compares it against the ground truth y_test.csv. In Python, the evaluation is equivalent to:

import numpy as np

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))


Files

The challenge data is split into three CSV files:
File 	Shape 	Description
X_train.csv 	489 rows x 10,001 columns 	Training features (methylation + gender)
y_train.csv 	489 rows x 1 column 	Training target (chronological age)
X_test.csv 	200 rows x 10,001 columns 	Test features (same columns as X_train)
Features (X_train, X_test)

Each row corresponds to an individual. The columns are:

    gender — categorical variable ("m" or "f")
    10,000 CpG probes (e.g. cg05918312, cg15523060, ...) — DNA methylation beta values ranging from 0 (unmethylated) to 1 (fully methylated)

>>> import pandas as pd
>>> X_train = pd.read_csv("X_train.csv")
>>> X_train.shape
(489, 10001)

>>> X_train.iloc[:5, :7]
  gender  cg05918312  cg15523060  cg11014124  cg18481596  cg02700891  cg06124711
0      m    0.812625    0.830062    0.375180    0.890127    0.046729    0.106498
1      f    0.634586    0.751566    0.173774    0.822238    0.074177    0.085449
2      m    0.725237    0.832268    0.210390    0.798711    0.065560    0.088541
3      f    0.752556    0.752197    0.191539    0.784993    0.065843    0.090826
4      f    0.709544    0.770333    0.184543    0.840292    0.078066    0.087805

>>> X_train["gender"].value_counts()
gender
f    348
m    141

Target (y_train)

The target variable is the chronological age (in years) of each individual:

>>> y_train = pd.read_csv("y_train.csv")
>>> y_train.describe()
              age
count  489.000000
mean    51.670757
std     11.841161
min     18.000000
25%     45.000000
50%     54.000000
75%     61.000000
max     70.000000

Ages range from 18 to 70 years, with a mean of approximately 52 years.
Key Characteristics

    High-dimensional setting: 10,001 features for only 489 training samples (p >> n). Regularization and/or feature selection will be essential.
    Mixed types: the gender column is categorical and will need encoding (e.g. pd.get_dummies or LabelEncoder) before use in most models.
    No missing values: all cells are populated — no imputation is required.
    Beta values: methylation values are bounded in [0, 1], with a global mean of ~0.49. No further normalization is strictly necessary, but standardization may help some models.


