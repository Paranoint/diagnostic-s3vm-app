import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.linalg import solve

class LapSVM:
    def __init__(self, *, gamma: float = 0.5, lamA: float = 1.0, lamI: float = 0.1, k: int = 10, eps: float = 1e-8):
        self.gamma = gamma
        self.lamA = lamA
        self.lamI = lamI
        self.k = k
        self.eps = eps

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return rbf_kernel(X1, X2, gamma=self.gamma)

    def _affinity(self, X: np.ndarray) -> np.ndarray:
        A = kneighbors_graph(X, self.k, mode="connectivity", include_self=False)
        return 0.5 * (A + A.T)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if not np.all(np.isin(np.unique(y), [-1.0, 0.0, 1.0])):
            raise ValueError("Метки должны быть -1 (неразмечено), 0 или 1")

        labeled_mask = y != -1
        if np.sum(labeled_mask) == 0:
            raise ValueError("Нет размеченных точек. Обучение невозможно.")

        n = len(y)
        K = self._rbf_kernel(X, X)
        W = self._affinity(X)
        L = csgraph.laplacian(W, normed=True).toarray()

        y_full = np.zeros(n)
        y_full[labeled_mask] = y[labeled_mask] * 2 - 1

        A = K + self.lamA * np.eye(n) + self.lamI * L
        self.alpha = solve(A, y_full)
        self.X_train_ = X
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        K_test = self._rbf_kernel(X, self.X_train_)
        return K_test @ self.alpha

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= self.eps, 1, 0)

def predict_lapsvm(X_labeled, y_labeled, X_unlabeled):
    y_all = np.concatenate([y_labeled, [-1] * len(X_unlabeled)])
    X_all = np.vstack([X_labeled, X_unlabeled])

    model = LapSVM(gamma=0.5, lamA=10.0, lamI=10.0, k=10)
    model.fit(X_all, y_all)
    return model.predict(X_unlabeled)
