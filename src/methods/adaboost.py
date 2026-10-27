"""
ADABOOST (Adaptive Boosting - Giải thuật SAMME Đa lớp & Nhị phân):
- Huấn luyện tuần tự T mô hình học yếu (Weak Learners, mặc định là Decision Stump).
- Áp dụng giải thuật SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss function).
- Tăng trọng số cho các mẫu bị phân lớp sai, giảm trọng số cho các mẫu đúng.
- Kết hợp tuyến tính các mô hình theo hệ số tin cậy alpha đa lớp.
"""
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    """
    Thuật toán AdaBoost tự cài đặt (From Scratch logic) hỗ trợ đa lớp (Multi-class SAMME).
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
        self.classes_ = None

    def fit(self, X, y):
        """
        Huấn luyện tuần tự T mô hình weak learners với phân phối trọng số cập nhật liên tục (SAMME).
        X: np.ndarray, kích thước (n, d)
        y: np.ndarray, nhãn phân lớp (0..K-1 hoặc {-1, +1})
        """
        self.classes_ = np.unique(y)
        K = len(self.classes_)
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

            # 3. Tính sai số có trọng số: epsilon_t = sum_{y != pred} w_i / sum(w)
            incorrect = (pred != y)
            eps = np.sum(w[incorrect]) / np.sum(w)
            eps = np.clip(eps, 1e-10, 1.0 - 1e-10)  # Tránh lỗi chia cho 0 hoặc log(0)

            # 4. Tính trọng số mô hình theo công thức SAMME:
            # alpha_t = ln((1 - epsilon_t) / epsilon_t) + ln(K - 1)
            if K > 1:
                alpha = np.log((1.0 - eps) / eps) + np.log(K - 1)
            else:
                alpha = np.log((1.0 - eps) / eps)

            # Nếu mô hình không tốt hơn đoán ngẫu nhiên (alpha <= 0), dừng sớm
            if alpha <= 0:
                if t == 0:
                    self.estimators_.append(est)
                    self.estimator_weights_.append(1.0)
                break

            # 5. Cập nhật trọng số mẫu theo cấp số nhân (SAMME):
            # w_i <- w_i * exp(alpha * I(y_i != h_t(x_i)))
            w *= np.exp(alpha * incorrect)
            w /= np.sum(w)  # Chuẩn hóa để sum(w) = 1

            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)

        return self

    def decision_function(self, X):
        """Tính ma trận điểm số tích lũy: S(x, k) = sum_{t=1}^T alpha_t * I(h_t(x) == k)."""
        n_samples = len(X)
        K = len(self.classes_)
        scores = np.zeros((n_samples, K))

        for alpha, est in zip(self.estimator_weights_, self.estimators_):
            pred = est.predict(X)
            for k_idx, c in enumerate(self.classes_):
                scores[pred == c, k_idx] += alpha

        return scores

    def staged_predict(self, X):
        """Dự đoán nhãn theo từng giai đoạn t = 1..T để theo dõi đường cong học."""
        n_samples = len(X)
        K = len(self.classes_)
        scores = np.zeros((n_samples, K))

        for alpha, est in zip(self.estimator_weights_, self.estimators_):
            pred = est.predict(X)
            for k_idx, c in enumerate(self.classes_):
                scores[pred == c, k_idx] += alpha
            yield self.classes_[np.argmax(scores, axis=1)]

    def predict(self, X):
        """Dự đoán nhãn cuối cùng: C(x) = argmax_k S(x, k)."""
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]
