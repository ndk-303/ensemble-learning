"""
Cac mo hinh co so (viet tay bang NumPy), dung lam "weak/base learner" cho ensemble.
Nhan: +1 = ac tinh (malignant), -1 = lanh tinh (benign).
"""
import numpy as np


class Stump:
    """Cay quyet dinh 1 tang. Chon (dac trung, nguong, chieu) cuc tieu SAI SO CO TRONG SO."""

    def fit(self, X, y, w=None):
        n, d = X.shape
        w = np.full(n, 1 / n) if w is None else w
        best = (np.inf, None, None, None)
        for f in range(d):
            xs = np.unique(X[:, f])
            if len(xs) < 2:
                continue
            thr = (xs[:-1] + xs[1:]) / 2                       # nguong = trung diem
            pred = np.where(X[:, f][None, :] < thr[:, None], 1, -1)  # chieu +: trai=+1
            err = ((pred != y[None, :]) * w).sum(1)
            for pol, e in ((1, err), (-1, w.sum() - err)):      # chieu - : dao nhan
                e = np.round(e, 12)
                i = int(np.argmin(e))
                if e[i] < best[0]:
                    best = (e[i], f, thr[i], pol)
        self.err, self.f, self.t, self.pol = best
        return self

    def predict(self, X):
        return self.pol * np.where(X[:, self.f] < self.t, 1, -1)

    def predict_proba(self, X):
        return (self.predict(X) == 1).astype(float)

    def __str__(self):
        s = "+1" if self.pol == 1 else "-1"
        return f"x[{self.f}] < {self.t:.2f} -> {s}, nguoc lai -> {'-1' if self.pol == 1 else '+1'}"


def ent(q):
    """Entropy nhi phan H(q) = -q*log2(q) - (1-q)*log2(1-q)."""
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


class Tree:
    """Cay quyet dinh chon diem chia theo Entropy (cuc dai hoa Information Gain)."""

    def __init__(self, max_depth=3, min_samples=2):
        self.max_depth, self.min_samples = max_depth, min_samples

    def fit(self, X, y):
        self.root = self._grow(X, (y == 1).astype(float), 0)
        return self

    def _grow(self, X, p, depth):
        n = len(p)
        node = {"p": p.mean()}
        if depth >= self.max_depth or n < self.min_samples or p.min() == p.max():
            return node
        d = X.shape[1]
        best, tot = (np.inf, None, None), p.sum()
        for f in range(d):
            o = np.argsort(X[:, f], kind="stable")
            xs, ps = X[o, f], p[o]
            cl = np.cumsum(ps)[:-1]
            nl = np.arange(1, n)
            nr, cr = n - nl, tot - cl
            g = (nl * ent(cl / nl) + nr * ent(cr / nr)) / n        # Entropy sau khi chia (nho nhat = IG lon nhat)
            g[xs[1:] <= xs[:-1]] = np.inf
            i = int(np.argmin(g))
            if g[i] < best[0]:
                best = (g[i], f, (xs[i] + xs[i + 1]) / 2)
        if best[1] is None or not np.isfinite(best[0]):
            return node
        _, f, t = best
        m = X[:, f] < t
        node.update(f=f, t=t, l=self._grow(X[m], p[m], depth + 1), r=self._grow(X[~m], p[~m], depth + 1))
        return node

    def _walk(self, node, X, idx, out):
        if "f" not in node:
            out[idx] = node["p"]
            return
        m = X[idx, node["f"]] < node["t"]
        self._walk(node["l"], X, idx[m], out)
        self._walk(node["r"], X, idx[~m], out)

    def predict_proba(self, X):
        out = np.empty(len(X))
        self._walk(self.root, X, np.arange(len(X)), out)
        return out

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, -1)


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.X, self.p = (X - self.mu) / self.sd, (y == 1).astype(float)
        return self

    def predict_proba(self, X):
        Z = (X - self.mu) / self.sd
        D = ((Z[:, None, :] - self.X[None]) ** 2).sum(2)
        return self.p[np.argsort(D, 1)[:, :self.k]].mean(1)

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, -1)


class LogReg:
    """Hoi quy logistic, gradient descent."""

    def __init__(self, lr=0.1, it=2000, l2=1e-3):
        self.lr, self.it, self.l2 = lr, it, l2

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z, t = (X - self.mu) / self.sd, (y == 1).astype(float)
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for _ in range(self.it):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            self.w -= self.lr * (Z.T @ (p - t) / len(t) + self.l2 * self.w)
            self.b -= self.lr * (p - t).mean()
        return self

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.w + self.b)))

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, -1)
