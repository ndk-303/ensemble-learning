"""
Module xu ly du lieu va khoi tao cac mo hinh co so.
- Tap du lieu: Scikit-Learn Digits (1797 mau anh 8x8, 64 dac trung, 10 lop so 0-9).
- Mo hinh co so: dt (Decision Tree), knn (K-Nearest Neighbors), svm (Support Vector Machine), lr (Logistic Regression).
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


SUPPORTED_MODELS = ["dt", "knn", "svm", "lr"]


def load_data(test_size=0.3, seed=42):
    """Tai va chia du lieu Digits theo ty le Train/Test phan tang (Stratified)."""
    dataset = load_digits()
    X = dataset.data  # shape (1797, 64)
    y = dataset.target  # Nhan so nguyen tu 0 den 9

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    feature_names = (
        dataset.feature_names
        if hasattr(dataset, "feature_names")
        else [f"pixel_{i}" for i in range(64)]
    )
    return X_train, y_train, X_test, y_test, feature_names


def build_model(model_name, mode="bagging", seed=0):
    """
    Khoi tao mo hinh co so theo ten viet tat.
    - dt: DecisionTreeClassifier (stump neu mode='boosting', depth=12 neu 'bagging', depth=6 neu 'stacking')
    - knn: KNeighborsClassifier (k=5 kem StandardScaler)
    - svm: SVC (probability=True kem StandardScaler)
    - lr: LogisticRegression (max_iter=2000 kem StandardScaler)
    """
    m = model_name.strip().lower()
    if m == "dt":
        if mode == "boosting":
            return DecisionTreeClassifier(max_depth=1, random_state=seed)
        elif mode == "bagging":
            return DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=seed)
        else:  # stacking or default
            return DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=seed)
    elif m == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    elif m == "svm":
        return make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed))
    elif m == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    else:
        raise ValueError(f"Model '{model_name}' khong hop le. Cac model ho tro: {', '.join(SUPPORTED_MODELS)}")


def get_base_models(seed=0):
    """Khoi tao danh sach cac mo hinh co so mac dinh de danh gia."""
    return {
        "Stump (1 tang)": build_model("dt", mode="boosting", seed=seed),
        "Cay sau (depth=12)": build_model("dt", mode="bagging", seed=seed),
        "KNN (k=5)": build_model("knn", seed=seed),
        "Logistic Regression": build_model("lr", seed=seed),
    }
