"""BAGGING: bootstrap (lay mau co hoan lai) + bo phieu da so."""
import numpy as np
from base_models import Tree


class Bagging:
    """Bootstrap + bo phieu da so."""

    def __init__(self, T=25, max_depth=12, seed=0):
        self.T, self.max_depth, self.seed = T, max_depth, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.trees = []
        for _ in range(self.T):
            idx = rng.integers(0, n, n)                          # lay mau CO HOAN LAI
            self.trees.append(Tree(self.max_depth).fit(X[idx], y[idx]))
        return self

    def staged_predict(self, X):
        v = np.zeros(len(X))
        for tr in self.trees:
            v += tr.predict(X)
            yield np.where(v >= 0, 1, -1)

    def predict(self, X):
        for p in self.staged_predict(X):
            pass
        return p
