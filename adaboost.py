"""
ADABOOST (Adaptive Boosting - Giải thuật SAMME nhị phân):
- Huấn luyện tuần tự T mô hình học yếu (Weak Learners, mặc định là Decision Stump).
- Tăng trọng số cho các mẫu bị phân lớp sai, giảm trọng số cho các mẫu đúng.
- Kết hợp tuyến tính các mô hình theo hệ số tin cậy alpha.
"""
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    """
    Thuật toán AdaBoost tự cài đặt (From Scratch logic) với nhãn {-1, +1}.
    Mặc định sử dụng Decision Stump (DecisionTreeClassifier với max_depth=1).
    """

    def __init__(self, base_estimator=None, T=50, random_state=0):
        self.base_estimator = base_estimator or DecisionTreeClassifier(
            max_depth=1, random_state=random_state
        )
        self.T = T
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []

    def fit(self, X, y):
        """
        Huấn luyện tuần tự T mô hình weak learners với phân phối trọng số cập nhật liên tục.
        X: np.ndarray, kích thước (n, d)
        y: np.ndarray, nhãn {-1, +1}
        """
        n = len(y)
        # 1. Khởi tạo phân phối trọng số mẫu đều nhau: w_i = 1 / n
        w = np.full(n, 1.0 / n)
        self.estimators_ = []
        self.estimator_weights_ = []

        for t in range(self.T):
            # 2. Huấn luyện weak learner với trọng số mẫu w hiện tại
            est = clone(self.base_estimator)
            if hasattr(est, "random_state"):
                est.random_state = self.random_state + t
            est.fit(X, y, sample_weight=w)
            pred = est.predict(X)

            # 3. Tính sai số có trọng số: epsilon_t = sum_{y != pred} w_i
            incorrect = (pred != y)
            eps = np.sum(w[incorrect])
            eps = np.clip(eps, 1e-10, 1.0 - 1e-10)  # Tránh lỗi chia cho 0 hoặc log(0)

            # 4. Tính trọng số mô hình: alpha_t = 0.5 * ln((1 - epsilon_t) / epsilon_t)
            alpha = 0.5 * np.log((1.0 - eps) / eps)

            # 5. Cập nhật trọng số mẫu theo cấp số nhân:
            # Mẫu đoán sai: y * pred = -1 => exp(alpha) > 1 (tăng trọng số)
            # Mẫu đoán đúng: y * pred = +1 => exp(-alpha) < 1 (giảm trọng số)
            w *= np.exp(-alpha * y * pred)
            w /= np.sum(w)  # Chuẩn hóa để sum(w) = 1

            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)

        return self

    def decision_function(self, X):
        """Tính tổng kết hợp có trọng số: F(x) = sum_{t=1}^T alpha_t * h_t(x)."""
        scores = np.zeros(len(X))
        for alpha, est in zip(self.estimator_weights_, self.estimators_):
            scores += alpha * est.predict(X)
        return scores

    def staged_predict(self, X):
        """Dự đoán nhãn theo từng giai đoạn t = 1..T để theo dõi đường cong học."""
        scores = np.zeros(len(X))
        for alpha, est in zip(self.estimator_weights_, self.estimators_):
            scores += alpha * est.predict(X)
            yield np.where(scores >= 0, 1, -1)

    def predict(self, X):
        """Dự đoán nhãn cuối cùng: H(x) = sign(F(x))."""
        return np.where(self.decision_function(X) >= 0, 1, -1)
