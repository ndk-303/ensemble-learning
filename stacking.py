"""STACKING: tang 1 = base learners (du doan out-of-fold), tang 2 = meta learner."""
import numpy as np


class Stacking:
    """Tang 1: base learners (du doan OUT-OF-FOLD). Tang 2: meta learner hoc tu cac du doan do."""

    def __init__(self, bases, meta, k=5, seed=0):
        self.bases, self.meta_f, self.k, self.seed = bases, meta, k, seed

    def fit(self, X, y):
        n = len(y)
        idx = np.random.default_rng(self.seed).permutation(n)
        folds = np.array_split(idx, self.k)
        Z = np.zeros((n, len(self.bases)))
        for f in folds:
            tr = np.setdiff1d(idx, f)
            for j, mk in enumerate(self.bases):
                Z[f, j] = mk().fit(X[tr], y[tr]).predict_proba(X[f])   # chi du doan phan DA GIU LAI
        self.Z = Z                                                    # bang meta-feature
        self.meta = self.meta_f().fit(Z, y)
        self.models = [mk().fit(X, y) for mk in self.bases]          # huan luyen lai tren toan bo train
        return self

    def meta_input(self, X):
        return np.column_stack([m.predict_proba(X) for m in self.models])

    def predict(self, X):
        return self.meta.predict(self.meta_input(X))
