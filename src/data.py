"""
Module xu ly du lieu va khoi tao cac mo hinh co so.
- Tap du lieu: Scikit-Learn Digits (1797 mau anh 8x8, 64 dac trung, 10 lop so 0-9).
- Mo hinh co so: DecisionTree, KNN, LogisticRegression.
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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


def get_base_models(seed=0):
    """Khoi tao cac mo hinh co so tu thu vien Scikit-Learn."""
    return {
        "Stump (1 tang)": DecisionTreeClassifier(
            criterion="entropy", max_depth=1, random_state=seed
        ),
        "Cay sau (depth=12)": DecisionTreeClassifier(
            criterion="entropy", max_depth=12, random_state=seed
        ),
        "KNN (k=5)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
    }
