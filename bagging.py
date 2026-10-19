"""
BAGGING (Bootstrap Aggregating):
- Lấy mẫu Bootstrap có hoàn lại từ tập dữ liệu huấn luyện.
- Huấn luyện T mô hình cơ sở độc lập trên từng tập con mẫu.
- Kết hợp các dự đoán độc lập bằng biểu quyết đa số (Majority Voting).
"""
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier


class Bagging:
    """
    Thuật toán Bagging tự cài đặt (From Scratch logic) hỗ trợ base_estimator từ Scikit-Learn.
    Mặc định sử dụng DecisionTreeClassifier (cây sâu để giảm phương sai / variance).
    """

    def __init__(self, base_estimator=None, T=25, seed=0):
        self.base_estimator = base_estimator or DecisionTreeClassifier(
            criterion="entropy", max_depth=12, random_state=seed
        )
        self.T = T
        self.seed = seed
        self.estimators_ = []

    def fit(self, X, y):
        """
        Huấn luyện T mô hình cơ sở trên T mẫu Bootstrap khác nhau.
        X: np.ndarray, kích thước (n, d)
        y: np.ndarray, nhãn {-1, 1}
        """
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
        votes = np.zeros(len(X))
        for est in self.estimators_:
            votes += est.predict(X)
            # Biểu quyết đa số trên nhãn {-1, 1}
            yield np.where(votes >= 0, 1, -1)

    def predict(self, X):
        """Dự đoán nhãn cuối cùng bằng Majority Voting từ toàn bộ T mô hình."""
        pred = None
        for p in self.staged_predict(X):
            pred = p
        return pred
