"""BOOSTING - dai dien: ADABOOST (huan luyen tuan tu, tang trong so mau bi doan sai)."""
import numpy as np
from base_models import Stump


class AdaBoost:
    """AdaBoost roi rac (SAMME hai lop) voi Stump."""

    def __init__(self, T=50):
        self.T = T

    def fit(self, X, y):
        n = len(y)
        w = np.full(n, 1 / n)
        self.stumps, self.alphas = [], []
        for _ in range(self.T):
            st = Stump().fit(X, y, w)
            pred = st.predict(X)
            eps = np.clip(w[pred != y].sum(), 1e-10, 1 - 1e-10)  # sai so co trong so
            alpha = 0.5 * np.log((1 - eps) / eps)                # trong so mo hinh
            w = w * np.exp(-alpha * y * pred)                    # tang mau sai, giam mau dung
            w /= w.sum()                                         # chuan hoa
            self.stumps.append(st)
            self.alphas.append(alpha)
        return self

    def score(self, X):
        return sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))

    def staged_predict(self, X):
        f = np.zeros(len(X))
        for a, s in zip(self.alphas, self.stumps):
            f += a * s.predict(X)
            yield np.where(f >= 0, 1, -1)

    def predict(self, X):
        return np.where(self.score(X) >= 0, 1, -1)
