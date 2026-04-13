from gender_models import build_best_gender_blend_model


class Model:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        target = y.iloc[:, 0] if hasattr(y, "iloc") else y
        self.model = build_best_gender_blend_model()
        self.model.fit(X=X, y=target)

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model must be fit before prediction.")
        return self.model.predict(X)
