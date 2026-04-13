import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import warnings

class Model:
    def __init__(self):
        warnings.filterwarnings("ignore")
        self.pipeline = Pipeline([
            ('variance_filter', VarianceThreshold(threshold=1e-5)),
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNetCV(
                l1_ratio=[0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0],
                n_alphas=25,
                cv=5,
                n_jobs=-1,
                max_iter=2500,
                tol=1e-3,
                random_state=42
            ))
        ])

    def preprocess_df(self, X):
        X_num = X.copy()
        if 'gender' in X_num.columns:
            X_num['gender'] = X_num['gender'].map({'m': 1, 'f': 0, 'M': 1, 'F': 0}).fillna(0.5)
        return X_num.astype(np.float32)

    def fit(self, X, y):
        X_num = self.preprocess_df(X)
        self.pipeline.fit(X_num, y.values.ravel())

    def predict(self, X):
        X_num = self.preprocess_df(X)
        return self.pipeline.predict(X_num)
