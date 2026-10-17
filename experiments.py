import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from base_models import Stump, Tree, KNN, LogReg
from bagging import Bagging
from adaboost import AdaBoost
from stacking import Stacking
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG, exist_ok=True)


def load(seed=42):
    d = load_breast_cancer()
    X = d.data
    y = np.where(d.target == 0, 1, -1)      
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in (1, -1):                                  
        ix = rng.permutation(np.where(y == c)[0])
        k = int(0.7 * len(ix))
        tr += ix[:k].tolist()
        te += ix[k:].tolist()
    tr, te = np.array(tr), np.array(te)
    return X[tr], y[tr], X[te], y[te], d.feature_names


def conf(y, p):
    return np.array([[np.sum((y == 1) & (p == 1)), np.sum((y == 1) & (p == -1))],
                     [np.sum((y == -1) & (p == 1)), np.sum((y == -1) & (p == -1))]])


def experiments():
    Xtr, ytr, Xte, yte, names = load()
    print("\n" + "=" * 70, f"\nTHUC NGHIEM: Breast Cancer Wisconsin  (train={len(ytr)}, test={len(yte)}, d={Xtr.shape[1]})\n" + "=" * 70)
    acc = lambda p: np.mean(p == yte)

    models = {}
    models["Stump (1 cay 1 tang)"] = Stump().fit(Xtr, ytr, None)
    models["Cay sau (depth=12)"] = Tree(12).fit(Xtr, ytr)
    models["KNN (k=5)"] = KNN(5).fit(Xtr, ytr)
    models["Logistic Regression"] = LogReg().fit(Xtr, ytr)
    bag = Bagging(T=50, max_depth=12).fit(Xtr, ytr)
    models["Bagging (50 cay)"] = bag
    ada = AdaBoost(T=100).fit(Xtr, ytr)
    models["AdaBoost (100 stump)"] = ada
    bases = [lambda: Tree(4), lambda: KNN(5), lambda: LogReg()]
    stk = Stacking(bases, lambda: LogReg(), k=5).fit(Xtr, ytr)
    models["Stacking (Tree+KNN+LR -> LR)"] = stk

    res = {}
    print(f"{'Mo hinh':32s} {'Train':>7s} {'Test':>7s}")
    for k, m in models.items():
        res[k] = (np.mean(m.predict(Xtr) == ytr), acc(m.predict(Xte)))
        print(f"{k:32s} {res[k][0]:7.4f} {res[k][1]:7.4f}")
    print("Meta LogReg weights [Tree, KNN, LR]:", np.round(stk.meta.w, 3).tolist())
    print("Base learners rieng le (test):",
          {n: round(float(acc(m.predict(Xte))), 4) for n, m in zip(["Tree4", "KNN5", "LogReg"], stk.models)})


    sk = {
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Bagging (50 cay)": BaggingClassifier(DecisionTreeClassifier(criterion="entropy", max_depth=12), n_estimators=50, random_state=0),
        "AdaBoost (100 stump)": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=0),
        "Stacking (Tree+KNN+LR -> LR)": StackingClassifier(
            [("t", DecisionTreeClassifier(criterion="entropy", max_depth=4)), ("k", make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
             ("l", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))],
            final_estimator=LogisticRegression(), cv=5),
    }
    print("\nDOI CHIEU voi sklearn (test accuracy):")
    print(f"{'Mo hinh':32s} {'Tu cai dat':>11s} {'sklearn':>9s}")
    for k, m in sk.items():
        print(f"{k:32s} {res[k][1]:11.4f} {m.fit(Xtr, ytr).score(Xte, yte):9.4f}")

    # ---- confusion matrices
    print("\nMa tran nham lan (hang: that [AcTinh, LanhTinh], cot: du doan [AcTinh, LanhTinh]):")
    cms = {}
    for k, m in models.items():
        cms[k] = conf(yte, m.predict(Xte))
        print(f"{k:32s}", cms[k].tolist())

    # ---- hinh 1: accuracy theo so mo hinh
    Tgrid = np.arange(1, 101)
    ada_tr = [np.mean(p == ytr) for p in ada.staged_predict(Xtr)]
    ada_te = [np.mean(p == yte) for p in ada.staged_predict(Xte)]
    bag_te = [np.mean(p == yte) for p in bag.staged_predict(Xte)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Tgrid, ada_tr, "--", c="tab:red", label="AdaBoost - train")
    ax.plot(Tgrid, ada_te, c="tab:red", label="AdaBoost - test")
    ax.plot(np.arange(1, 51), bag_te, c="tab:blue", label="Bagging - test")
    ax.axhline(res["Stump (1 cay 1 tang)"][1], c="gray", ls=":", label="Stump don - test")
    ax.set_xlabel("So mo hinh T"); ax.set_ylabel("Accuracy"); ax.set_ylim(0.85, 1.01)
    ax.legend(); ax.grid(alpha=.3); ax.set_title("Accuracy theo so mo hinh trong ensemble")
    fig.tight_layout(); fig.savefig(f"{FIG}/acc_vs_T.png", dpi=150); plt.close(fig)
    print(f"\nAdaBoost test: T=1 -> {ada_te[0]:.4f}, T=10 -> {ada_te[9]:.4f}, T=50 -> {ada_te[49]:.4f}, T=100 -> {ada_te[99]:.4f}")
    print(f"Bagging  test: T=1 -> {bag_te[0]:.4f}, T=10 -> {bag_te[9]:.4f}, T=50 -> {bag_te[49]:.4f}")

    # ---- hinh 2: so sanh test accuracy
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names_ = list(res)
    vals = [res[k][1] for k in names_]
    colors = ["#777" if not any(x in k for x in ["Bagging", "AdaBoost", "Stacking"]) else "tab:blue" if "Bagging" in k else "tab:red" if "AdaBoost" in k else "tab:green" for k in names_]
    bars = ax.barh(names_, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + .002, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center")
    ax.set_xlim(0.85, 1.01); ax.invert_yaxis(); ax.set_xlabel("Test accuracy")
    ax.set_title("So sanh cac mo hinh (Breast Cancer Wisconsin)")
    fig.tight_layout(); fig.savefig(f"{FIG}/compare.png", dpi=150); plt.close(fig)

    # ---- hinh 3: ma tran nham lan
    fig, axs = plt.subplots(1, len(cms), figsize=(2.4 * len(cms), 3.2))
    for ax, (k, c) in zip(axs, cms.items()):
        ax.imshow(c, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, c[i, j], ha="center", va="center", color="white" if c[i, j] > c.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Ac", "Lanh"]); ax.set_yticklabels(["Ac", "Lanh"])
        ax.set_title(k.split(" (")[0], fontsize=8); ax.set_xlabel("Du doan"); ax.set_ylabel("That")
    fig.tight_layout(); fig.savefig(f"{FIG}/confusion.png", dpi=150); plt.close(fig)

    # ---- hinh 4: ranh gioi quyet dinh 2D (mean radius, mean concave points)
    f2 = [0, 7]
    X2, X2t = Xtr[:, f2], Xte[:, f2]
    m2 = {
        "Stump": Stump().fit(X2, ytr),
        "Cay sau": Tree(12).fit(X2, ytr),
        "KNN (5)": KNN(5).fit(X2, ytr),
        "LogReg": LogReg().fit(X2, ytr),
        "Bagging (50)": Bagging(T=50, max_depth=12).fit(X2, ytr),
        "AdaBoost (100)": AdaBoost(T=100).fit(X2, ytr),
    }
    gx, gy = np.meshgrid(np.linspace(X2[:, 0].min(), X2[:, 0].max(), 150), np.linspace(X2[:, 1].min(), X2[:, 1].max(), 150))
    G = np.c_[gx.ravel(), gy.ravel()]
    fig, axs = plt.subplots(1, len(m2), figsize=(2.8 * len(m2), 3.6))
    print("\nThuc nghiem 2D (mean radius, mean concave points) - test accuracy:")
    for ax, (k, m) in zip(axs, m2.items()):
        ax.contourf(gx, gy, m.predict(G).reshape(gx.shape), alpha=.25, cmap="coolwarm", levels=[-2, 0, 2])
        ax.scatter(X2[:, 0], X2[:, 1], c=ytr, cmap="coolwarm", s=8, edgecolor="none")
        a = np.mean(m.predict(X2t) == yte)
        ax.set_title(f"{k}\ntest acc = {a:.3f}", fontsize=9); ax.set_xlabel("mean radius"); ax.set_ylabel("mean concave points")
        print(f"  {k:16s} {a:.4f}")
    fig.tight_layout(); fig.savefig(f"{FIG}/boundary.png", dpi=150); plt.close(fig)


def repeated(n=10):
    """Lap n lan chia train/test khac nhau -> trung binh +- do lech chuan (on dinh hon 1 lan chia)."""
    names = ["Stump", "Cay sau", "KNN (5)", "LogReg", "Bagging (25)", "AdaBoost (50)", "Stacking"]
    R = {k: [] for k in names}
    for sd in range(n):
        Xtr, ytr, Xte, yte, _ = load(sd)
        ms = {
            "Stump": Stump().fit(Xtr, ytr),
            "Cay sau": Tree(12).fit(Xtr, ytr),
            "KNN (5)": KNN(5).fit(Xtr, ytr),
            "LogReg": LogReg().fit(Xtr, ytr),
            "Bagging (25)": Bagging(25, 12, seed=sd).fit(Xtr, ytr),
            "AdaBoost (50)": AdaBoost(50).fit(Xtr, ytr),
            "Stacking": Stacking([lambda: Tree(4), lambda: KNN(5), lambda: LogReg()], lambda: LogReg(), seed=sd).fit(Xtr, ytr),
        }
        for k, m in ms.items():
            R[k].append(np.mean(m.predict(Xte) == yte))
    print("\n" + "=" * 70, f"\nLAP {n} LAN CHIA TRAIN/TEST NGAU NHIEN (test accuracy)\n" + "=" * 70)
    mean = [np.mean(R[k]) for k in names]
    std = [np.std(R[k]) for k in names]
    for k, m, s_ in zip(names, mean, std):
        print(f"{k:22s} {m:.4f} +- {s_:.4f}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#777" if not any(x in k for x in ["Bagging", "AdaBoost", "Stacking"]) else "tab:blue" if "Bagging" in k else "tab:red" if "AdaBoost" in k else "tab:green" for k in names]
    ax.barh(names, mean, xerr=std, color=colors, capsize=3)
    for i, (m, s_) in enumerate(zip(mean, std)):
        ax.text(m + s_ + .002, i, f"{m:.3f}", va="center")
    ax.set_xlim(0.88, 1.01); ax.invert_yaxis(); ax.set_xlabel("Test accuracy (TB +- std, 10 lan chia)")
    ax.set_title("So sanh mo hinh - Breast Cancer Wisconsin")
    fig.tight_layout(); fig.savefig(f"{FIG}/compare_repeated.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    experiments()
    repeated(10)
