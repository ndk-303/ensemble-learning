"""
PIPELINE THUC NGHIEM & DOI CHIEU ENSEMBLE LEARNING
- Tap du lieu: Scikit-Learn Digits (1797 mau anh 8x8, 64 dac trung, 10 lop so 0-9).
- Mo hinh co so: Truc tiep tu thu vien Scikit-Learn (DecisionTree, KNN, LogisticRegression).
- Mo hinh Ensemble: Tu cai dat thuan toan hoc (Bagging, AdaBoost SAMME da lop, Stacking OOF).
- Doi chieu: So sanh truc tiep voi cac module tuong ung cua Scikit-Learn.
"""
import os
import sys

# Dam bao tuong thich ma hoa console tren Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bagging import Bagging
from adaboost import AdaBoost
from stacking import Stacking

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def load_data(seed=42):
    """Tai va chia du lieu Digits theo ty le 70% Train / 30% Test co phan tang (Stratified)."""
    dataset = load_digits()
    X = dataset.data  # shape (1797, 64)
    y = dataset.target  # Nhan so nguyen tu 0 den 9

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    feature_names = dataset.feature_names if hasattr(dataset, "feature_names") else [f"pixel_{i}" for i in range(64)]
    return X_train, y_train, X_test, y_test, feature_names


def get_base_models(seed=0):
    """Khoi tao cac mo hinh co so tu thu vien Scikit-Learn."""
    return {
        "Stump (1 tang)": DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=seed),
        "Cay sau (depth=12)": DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=seed),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
    }


def experiments():
    Xtr, ytr, Xte, yte, feature_names = load_data(seed=42)
    classes = np.unique(ytr)
    n_classes = len(classes)
    print("\n" + "=" * 75)
    print(f"THUC NGHIEM: Scikit-Learn Digits (Train: {len(ytr)}, Test: {len(yte)}, d: {Xtr.shape[1]}, Classes: {n_classes})")
    print("=" * 75)

    # 1. Khoi tao va huan luyen cac mo hinh co so
    models = get_base_models(seed=0)
    for name, model in list(models.items()):
        model.fit(Xtr, ytr)

    # 2. Khoi tao va huan luyen cac mo hinh Ensemble tu cai dat
    bag = Bagging(
        base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=0),
        T=50,
        seed=0,
    ).fit(Xtr, ytr)
    models["Bagging (50 cay)"] = bag

    ada = AdaBoost(
        base_estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
        T=100,
        random_state=0,
    ).fit(Xtr, ytr)
    models["AdaBoost (100 stump)"] = ada

    stacking_bases = [
        DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=0),
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)),
    ]
    stk = Stacking(
        base_estimators=stacking_bases,
        meta_estimator=LogisticRegression(max_iter=2000, random_state=0),
        k=5,
        seed=0,
    ).fit(Xtr, ytr)
    models["Stacking (Tree+KNN+LR -> LR)"] = stk

    # 3. Danh gia Accuracy tren Train va Test
    res = {}
    print(f"\n{'Mo hinh':32s} {'Train Acc':>10s} {'Test Acc':>10s}")
    print("-" * 55)
    for name, model in models.items():
        tr_acc = accuracy_score(ytr, model.predict(Xtr))
        te_acc = accuracy_score(yte, model.predict(Xte))
        res[name] = (tr_acc, te_acc)
        print(f"{name:32s} {tr_acc:10.4f} {te_acc:10.4f}")

    if hasattr(stk.fitted_meta_, "coef_"):
        print(f"\nKich thuoc ma tran trong so Meta Logistic Regression (K x MetaFeatures): {stk.fitted_meta_.coef_.shape}")

    # 4. Doi chieu truc tiep voi cac mo hinh Scikit-Learn tuong ung
    sk_models = {
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)),
        "Bagging (50 cay)": BaggingClassifier(
            estimator=DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=0),
            n_estimators=50,
            random_state=0,
        ),
        "AdaBoost (100 stump)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
            n_estimators=100,
            random_state=0,
        ),
        "Stacking (Tree+KNN+LR -> LR)": StackingClassifier(
            estimators=[
                ("tree", DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=0)),
                ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
                ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))),
            ],
            final_estimator=LogisticRegression(max_iter=2000, random_state=0),
            cv=5,
        ),
    }

    print("\n" + "=" * 75)
    print(f"{'MO HINH':32s} {'TU CAI DAT':>14s} {'SCIKIT-LEARN':>14s}")
    print("=" * 75)
    for name, sk_model in sk_models.items():
        sk_acc = accuracy_score(yte, sk_model.fit(Xtr, ytr).predict(Xte))
        print(f"{name:32s} {res[name][1]:14.4f} {sk_acc:14.4f}")

    # 5. Ma tran nham lan (Confusion Matrix 10x10)
    print("\nMa tran nham lan (Tong so mau du doan dung tren tong 540 mau test):")
    cms = {}
    for name, model in models.items():
        cm = confusion_matrix(yte, model.predict(Xte), labels=classes)
        cms[name] = cm
        correct = np.trace(cm)
        total = np.sum(cm)
        print(f"  {name:32s} -> Dung: {correct}/{total} ({correct/total*100:.2f}%)")

    # 6. Bieu do 1: Accuracy theo so luong mo hinh T
    T_grid = np.arange(1, 101)
    ada_tr = [accuracy_score(ytr, p) for p in ada.staged_predict(Xtr)]
    ada_te = [accuracy_score(yte, p) for p in ada.staged_predict(Xte)]
    bag_te = [accuracy_score(yte, p) for p in bag.staged_predict(Xte)]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(T_grid, ada_tr, "--", c="tab:red", label="AdaBoost (SAMME) - Train")
    ax.plot(T_grid, ada_te, c="tab:red", lw=1.8, label="AdaBoost (SAMME) - Test")
    ax.plot(np.arange(1, 51), bag_te, c="tab:blue", lw=1.8, label="Bagging - Test")
    ax.axhline(res["Stump (1 tang)"][1], c="gray", ls=":", label="Stump don - Test")
    ax.set_xlabel("So luong mo hinh (T)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.15, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Do chinh xac theo so luong mo hinh trong Ensemble (T) - Digits")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "acc_vs_T.png"), dpi=150)
    plt.close(fig)

    # 7. Bieu do 2: So sanh Test Accuracy
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    names_list = list(res.keys())
    test_scores = [res[k][1] for k in names_list]
    colors = [
        "tab:blue" if "Bagging" in k else "tab:red" if "AdaBoost" in k else "tab:green" if "Stacking" in k else "#7f7f7f"
        for k in names_list
    ]
    bars = ax.barh(names_list, test_scores, color=colors)
    for b, score in zip(bars, test_scores):
        ax.text(score + 0.005, b.get_y() + b.get_height() / 2, f"{score:.4f}", va="center", fontsize=9)
    ax.set_xlim(0.15, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("Test Accuracy")
    ax.set_title("So sanh do chinh xac cac mo hinh (Scikit-Learn Digits 0-9)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "compare.png"), dpi=150)
    plt.close(fig)

    # 8. Bieu do 3: Ma tran nham lan truc quan 10x10 cho cac mo hinh
    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    axs_flat = axs.ravel()
    for idx, (name, cm) in zip(range(len(cms)), cms.items()):
        ax = axs_flat[idx]
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val > cm.max() / 2 else "black", fontsize=6.5)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        short_name = name.split(" (")[0]
        acc_val = res[name][1]
        ax.set_title(f"{short_name} (Acc: {acc_val:.3f})", fontsize=10)
        ax.set_xlabel("Du doan", fontsize=8)
        ax.set_ylabel("Thuc te", fontsize=8)

    # O cuoi cung: Hien thi mau cac chu so 0-9 tu dataset
    ax_sample = axs_flat[-1]
    ax_sample.axis("off")
    raw_digits = load_digits()
    for i in range(10):
        sample_img = raw_digits.images[raw_digits.target == i][0]
        sub_ax = ax_sample.inset_axes([0.1 + (i % 5) * 0.17, 0.55 - (i // 5) * 0.45, 0.15, 0.38])
        sub_ax.imshow(sample_img, cmap="gray")
        sub_ax.set_title(f"So {i}", fontsize=8)
        sub_ax.axis("off")
    ax_sample.set_title("Mau cac chu so viet tay (8x8)", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion.png"), dpi=150)
    plt.close(fig)

    # 9. Bieu do 4: Ranh gioi phan lop 2D qua PCA chieu xuong 2 chieu chinh
    pca = PCA(n_components=2, random_state=0)
    X2_tr = pca.fit_transform(Xtr)
    X2_te = pca.transform(Xte)

    m2 = {
        "Stump": DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=0).fit(X2_tr, ytr),
        "Cay sau": DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=0).fit(X2_tr, ytr),
        "KNN (5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(X2_tr, ytr),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)).fit(X2_tr, ytr),
        "Bagging (50)": Bagging(
            base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=0),
            T=50,
            seed=0,
        ).fit(X2_tr, ytr),
        "AdaBoost (100)": AdaBoost(
            base_estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
            T=100,
            random_state=0,
        ).fit(X2_tr, ytr),
    }

    x_min, x_max = X2_tr[:, 0].min() - 2, X2_tr[:, 0].max() + 2
    y_min, y_max = X2_tr[:, 1].min() - 2, X2_tr[:, 1].max() + 2
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    G = np.c_[gx.ravel(), gy.ravel()]

    fig, axs = plt.subplots(1, len(m2), figsize=(3.0 * len(m2), 3.6))
    print("\nThuc nghiem tren khong gian 2D PCA:")
    for ax, (name, model) in zip(axs, m2.items()):
        Z_grid = model.predict(G).reshape(gx.shape)
        ax.contourf(gx, gy, Z_grid, alpha=0.3, cmap="tab10", levels=np.arange(-0.5, 10.5, 1))
        ax.scatter(X2_te[:, 0], X2_te[:, 1], c=yte, cmap="tab10", s=6, alpha=0.6, edgecolor="none")
        acc_2d = accuracy_score(yte, model.predict(X2_te))
        ax.set_title(f"{name}\nTest acc = {acc_2d:.3f}", fontsize=9)
        ax.set_xlabel("PCA Dim 1", fontsize=8)
        ax.set_ylabel("PCA Dim 2", fontsize=8)
        print(f"  {name:16s}: {acc_2d:.4f}")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boundary.png"), dpi=150)
    plt.close(fig)


def repeated(n=10):
    """Lap n lan phan chia Train/Test ngau nhien (70/30) de tinh Mean +- Std cua Test Accuracy tren Digits."""
    names = ["Stump", "Cay sau", "KNN (5)", "LogReg", "Bagging (25)", "AdaBoost (50)", "Stacking"]
    records = {k: [] for k in names}

    for sd in range(n):
        Xtr, ytr, Xte, yte, _ = load_data(seed=sd)
        ms = {
            "Stump": DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=sd).fit(Xtr, ytr),
            "Cay sau": DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=sd).fit(Xtr, ytr),
            "KNN (5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(Xtr, ytr),
            "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=sd)).fit(Xtr, ytr),
            "Bagging (25)": Bagging(
                base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=sd),
                T=25,
                seed=sd,
            ).fit(Xtr, ytr),
            "AdaBoost (50)": AdaBoost(
                base_estimator=DecisionTreeClassifier(max_depth=1, random_state=sd),
                T=50,
                random_state=sd,
            ).fit(Xtr, ytr),
            "Stacking": Stacking(
                base_estimators=[
                    DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=sd),
                    make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
                    make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=sd)),
                ],
                meta_estimator=LogisticRegression(max_iter=2000, random_state=sd),
                k=5,
                seed=sd,
            ).fit(Xtr, ytr),
        }
        for k, m in ms.items():
            records[k].append(accuracy_score(yte, m.predict(Xte)))

    print("\n" + "=" * 75)
    print(f"LAP {n} LAN CHIA TRAIN/TEST NGAU NHIEN TREN DIGITS (Test Accuracy: Mean +- Std)")
    print("=" * 75)
    means = [np.mean(records[k]) for k in names]
    stds = [np.std(records[k]) for k in names]

    for k, m, s_ in zip(names, means, stds):
        print(f"{k:24s}: {m:.4f} +- {s_:.4f}")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = [
        "tab:blue" if "Bagging" in k else "tab:red" if "AdaBoost" in k else "tab:green" if "Stacking" in k else "#7f7f7f"
        for k in names
    ]
    ax.barh(names, means, xerr=stds, color=colors, capsize=3)
    for i, (m, s_) in enumerate(zip(means, stds)):
        ax.text(m + s_ + 0.005, i, f"{m:.4f}", va="center", fontsize=9)
    ax.set_xlim(0.15, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("Test Accuracy (Mean +- Std qua 10 lan)")
    ax.set_title("Do on dinh cua cac mo hinh qua 10 lan phan chia du lieu ngau nhien (Digits)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "compare_repeated.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    experiments()
    repeated(10)
