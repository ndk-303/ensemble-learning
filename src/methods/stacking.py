"""
STACKING (Stacked Generalization):
- Tầng 1 (Base Learners): Dự đoán Out-Of-Fold (OOF) qua K-Fold CV để tạo meta-features.
- Tránh rò rỉ dữ liệu (Data Leakage) bằng cách chỉ dự đoán trên fold giữ lại.
- Tầng 2 (Meta Learner): Huấn luyện trên ma trận meta-features để đưa ra dự đoán tổng hợp.
- Hỗ trợ cả bài toán nhị phân và đa lớp (Multi-class).
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold


class Stacking:
    """
    Thuật toán Stacking tự cài đặt (From Scratch logic) hỗ trợ các estimator từ Scikit-Learn.
    Hỗ trợ cả bài toán phân lớp nhị phân và đa lớp.
    """

    def __init__(self, base_estimators, meta_estimator, k=5, seed=0):
        """
        base_estimators: danh sách các mô hình tầng 1 (e.g. Tree, KNN, LogisticRegression).
        meta_estimator: mô hình tầng 2 (e.g. LogisticRegression).
        k: số fold cho Out-Of-Fold Cross-Validation.
        """
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.k = k
        self.seed = seed
        self.fitted_bases_ = []
        self.fitted_meta_ = None
        self.meta_features_ = None

    def _get_pred(self, model, X):
        """Trích xuất xác suất dự đoán (hoặc decision function) từ mô hình cơ sở."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X)
            if df.ndim == 1:
                df = df[:, np.newaxis]
            return df
        else:
            p = model.predict(X)
            return p[:, np.newaxis]

    def fit(self, X, y, callback=None):
        """
        1. Tạo meta-features Z bằng K-Fold Out-of-fold.
        2. Huấn luyện Meta-Learner trên Z.
        3. Huấn luyện lại toàn bộ Base Learners trên toàn bộ tập train.
        """
        n = len(y)
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        Z = None

        if callback:
            callback("phase_1")
        # 1. Cơ chế Out-Of-Fold (OOF) chống Overfitting và Data Leakage
        for train_idx, val_idx in kf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            fold_preds = []
            for base in self.base_estimators:
                model_fold = clone(base)
                model_fold.fit(X_tr, y_tr)
                fold_preds.append(self._get_pred(model_fold, X_val))

            Z_val = np.column_stack(fold_preds)
            if Z is None:
                Z = np.zeros((n, Z_val.shape[1]))
            Z[val_idx] = Z_val

        self.meta_features_ = Z

        if callback:
            callback("phase_2")
        # 2. Huấn luyện Meta-Learner ở tầng 2 trên ma trận meta-features Z
        self.fitted_meta_ = clone(self.meta_estimator)
        self.fitted_meta_.fit(Z, y)

        if callback:
            callback("phase_3")
        # 3. Huấn luyện lại toàn bộ Base Learners trên toàn bộ tập train để phục vụ inference
        self.fitted_bases_ = []
        for base in self.base_estimators:
            full_model = clone(base)
            full_model.fit(X, y)
            self.fitted_bases_.append(full_model)

        return self

    def construct_meta_features(self, X):
        """Tạo đặc trưng meta cho tập kiểm thử bằng cách cho các Base Learners dự đoán."""
        cols = [self._get_pred(m, X) for m in self.fitted_bases_]
        return np.column_stack(cols)

    def predict(self, X):
        """Dự đoán nhãn cuối cùng từ Meta-Learner."""
        Z_test = self.construct_meta_features(X)
        return self.fitted_meta_.predict(Z_test)
