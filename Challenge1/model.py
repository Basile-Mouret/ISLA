from sklearn.dummy import DummyRegressor


class Model:
    def __init__(self):
        self.dummy = DummyRegressor(strategy="mean")

    def fit(self, X, y):
        self.dummy.fit(X=X, y=y)

    def predict(self, X):
        return self.dummy.predict(X)
