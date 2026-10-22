"""
BAGGING (Bootstrap Aggregating):
- Lấy mẫu Bootstrap có hoàn lại từ tập dữ liệu huấn luyện.
- Huấn luyện T mô hình cơ sở độc lập trên từng tập con mẫu.
- Kết hợp các dự đoán độc lập bằng biểu quyết đa số (Majority Voting) cho cả nhị phân và đa lớp.
"""
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier


class Bagging:
    """
    Thuật toán Bagging tự cài đặt (From Scratch logic) hỗ trợ base_estimator từ Scikit-Learn.
    Hỗ trợ cả bài toán phân lớp nhị phân và đa lớp (Multi-class).
    Mặc định sử dụng DecisionTreeClassifier (cây sâu để giảm phương sai / variance).
    """

    def __init__(self, base_estimator=None, T=25, seed=0):
        self.base_estimator = base_estimator or DecisionTreeClassifier(
            criterion="entropy", max_depth=12, random_state=seed
        )
        self.T = T
        self.seed = seed
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        """
        Huấn luyện T mô hình cơ sở trên T mẫu Bootstrap khác nhau.
        X: np.ndarray, kích thước (n, d)
        y: np.ndarray, nhãn phân lớp (0..K-1 hoặc {-1, 1})
        """
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.estimators_ = []

        for i in range(self.T):
            # 1. Bootstrap Sampling: Lấy ngẫu nhiên n phần tử CÓ HOÀN LẠI
            boot_idx = rng.integers(0, n, size=n)
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            # 2. Huấn luyện mô hình cơ sở độc lập trên tập mẫu con
            est = clone(self.base_estimator)
            if hasattr(est, "random_state"):
                est.random_state = self.seed + i
            est.fit(X_boot, y_boot)
            self.estimators_.append(est)

        return self

    def staged_predict(self, X):
        """Dự đoán tích lũy qua từng mô hình t = 1..T để theo dõi đường cong hội tụ."""
        n_samples = len(X)
        n_classes = len(self.classes_)
        votes = np.zeros((n_samples, n_classes), dtype=int)

        for est in self.estimators_:
            preds = est.predict(X)
            for c_idx, c in enumerate(self.classes_):
                votes[preds == c, c_idx] += 1
            # Biểu quyết đa số (Majority Voting)
            best_idx = np.argmax(votes, axis=1)
            yield self.classes_[best_idx]

    def predict(self, X):
        """Dự đoán nhãn cuối cùng bằng Majority Voting từ toàn bộ T mô hình."""
        pred = None
        for p in self.staged_predict(X):
            pred = p
        return pred
