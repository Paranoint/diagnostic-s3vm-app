from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.kernel_ridge import KernelRidge
import numpy as np

class HelpTrainingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, threshold=0.95, max_iter=10):
        self.base_estimator = clone(base_estimator)
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, X, y):
        y = np.array(y)
        labeled_mask = y != -1
        X_labeled, y_labeled = X[labeled_mask], y[labeled_mask]
        X_unlabeled = X[~labeled_mask]

        for _ in range(self.max_iter):
            self.base_estimator.fit(X_labeled, y_labeled)

            if len(X_unlabeled) == 0:
                break

            probs = self.base_estimator.predict_proba(X_unlabeled)
            confidence = np.max(probs, axis=1)
            predicted = np.argmax(probs, axis=1)

            confident_idx = confidence >= self.threshold
            if not np.any(confident_idx):
                break

            X_conf = X_unlabeled[confident_idx]
            y_conf = predicted[confident_idx]

            X_labeled = np.vstack((X_labeled, X_conf))
            y_labeled = np.concatenate((y_labeled, y_conf))
            X_unlabeled = X_unlabeled[~confident_idx]

        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

class SurrogateLSSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=1.0, sigma=1.0):
        self.gamma = gamma
        self.sigma = sigma
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.model_ = KernelRidge(
            alpha=1.0 / self.gamma,
            kernel='rbf',
            gamma=1.0 / (2 * self.sigma**2)
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return np.where(self.model_.predict(X) >= 0.5, 1, 0)

    def predict_proba(self, X):
        scores = self.model_.predict(X)
        prob_pos = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - prob_pos, prob_pos]).T
    
def predict_ls_help(X_labeled, y_labeled, X_unlabeled):
    base_model = SurrogateLSSVM(gamma=30.0, sigma=13.5)
    model = HelpTrainingClassifier(base_model, threshold=0.9, max_iter=10)

    model.fit(X_labeled, y_labeled)
    return model.predict(X_unlabeled)