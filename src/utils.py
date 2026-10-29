"""
Module tien ich: Dinh dang terminal log chuan ML, in bang ASCII va xuat bieu do truc quan hoa.
"""
import os
import sys
import numpy as np

# Dam bao tuong thich ma hoa console tren Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def log_info(msg):
    """Xuat log muc INFO."""
    print(f"[INFO] {msg}")


def log_train(msg):
    """Xuat log muc TRAIN."""
    print(f"[TRAIN] {msg}")


def log_done(msg):
    """Xuat log muc DONE."""
    print(f"[DONE] {msg}")


def print_table(headers, rows, alignments=None):
    """In bang ASCII tieu chuan, can le dep cho cac chi so ML."""
    if alignments is None:
        alignments = ["<"] + [">"] * (len(headers) - 1)

    widths = [max(len(str(val)) for val in [h] + [r[i] for r in rows]) for i, h in enumerate(headers)]
    sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"

    print(sep)
    print("|" + "|".join([f" {h:^{widths[i]}} " for i, h in enumerate(headers)]) + "|")
    print(sep)
    for row in rows:
        cells = [
            f" {str(v):>{w}} " if a == ">" else f" {str(v):^{w}} " if a == "^" else f" {str(v):<{w}} "
            for v, a, w in zip(row, alignments, widths)
        ]
        print("|" + "|".join(cells) + "|")
    print(sep)


def _save_fig(fig, path):
    """Luu bieu do matplotlib va giai phong bo nho."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    p = path.replace("\\", "/")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def save_single_confusion(yte, y_pred, title, path="figs/confusion.png"):
    """Luu 1 ma tran nham lan duy nhat cho mo hinh don."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(yte, y_pred, cmap="Blues", ax=ax)
    ax.set_title(title, fontsize=10)
    return _save_fig(fig, path)


def save_visualizations(
    res,
    ytr,
    yte,
    Xtr,
    Xte,
    ada,
    bag,
    stk,
    adaboost_t,
    bagging_t,
    stk_name,
    fig_dir="figs",
):
    """Xuat 3 bieu do tinh gon cho thi nghiem so sanh: acc_vs_T, compare, confusion."""
    # Bieu do 1: Accuracy theo T
    ada_tr = [accuracy_score(ytr, p) for p in ada.staged_predict(Xtr)]
    ada_te = [accuracy_score(yte, p) for p in ada.staged_predict(Xte)]
    bag_te = [accuracy_score(yte, p) for p in bag.staged_predict(Xte)]
    T_grid = np.arange(1, len(ada_te) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(T_grid, ada_tr, "--", c="tab:red", label="AdaBoost (Train)")
    ax.plot(T_grid, ada_te, c="tab:red", lw=1.8, label="AdaBoost (Test)")
    ax.plot(np.arange(1, len(bag_te) + 1), bag_te, c="tab:blue", lw=1.8, label="Bagging (Test)")
    if "Stump" in res:
        ax.axhline(res["Stump"][1], c="gray", ls=":", label="Stump (Test)")
    ax.set(xlabel="Number of Estimators (T)", ylabel="Accuracy", ylim=(0.15, 1.02), title="Accuracy vs. Number of Estimators")
    ax.legend()
    ax.grid(alpha=0.3)
    p1 = _save_fig(fig, os.path.join(fig_dir, "acc_vs_T.png"))

    # Bieu do 2: So sanh Test Accuracy
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    names = list(res.keys())
    scores = [res[k][1] for k in names]
    colors = [
        "tab:blue" if "Bagging" in k else "tab:red" if "AdaBoost" in k else "tab:green" if "Stacking" in k else "#7f7f7f"
        for k in names
    ]
    bars = ax.barh(names, scores, color=colors)
    for b, score in zip(bars, scores):
        ax.text(score + 0.005, b.get_y() + b.get_height() / 2, f"{score:.4f}", va="center", fontsize=9)
    ax.set(xlim=(0, 1.05), xlabel="Test Accuracy", title="Model Accuracy Comparison")
    ax.invert_yaxis()
    p2 = _save_fig(fig, os.path.join(fig_dir, "compare.png"))

    # Bieu do 3: Ma tran nham lan cua Stacking
    p3 = save_single_confusion(yte, stk.predict(Xte), f"Confusion Matrix ({stk_name})", os.path.join(fig_dir, "confusion.png"))

    return [p1, p2, p3]


def save_stability_plot(names, means, stds, n, fig_dir="figs"):
    """Xuat bieu do so sanh do on dinh Mean +- Std qua n lan lap."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = [
        "tab:blue" if "Bagging" in k else "tab:red" if "AdaBoost" in k else "tab:green" if "Stacking" in k else "#7f7f7f"
        for k in names
    ]
    ax.barh(names, means, xerr=stds, color=colors, capsize=3)
    for i, (m, s_) in enumerate(zip(means, stds)):
        ax.text(m + s_ + 0.005, i, f"{m:.4f}", va="center", fontsize=9)
    ax.set(xlim=(0, 1.05), xlabel="Test Accuracy", title=f"Model Stability ({n} Repeats)")
    ax.invert_yaxis()
    return _save_fig(fig, os.path.join(fig_dir, "compare_repeated.png"))
