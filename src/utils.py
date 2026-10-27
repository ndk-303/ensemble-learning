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


def log_warn(msg):
    """Xuat log muc WARN."""
    print(f"[WARN] {msg}")


def log_error(msg):
    """Xuat log muc ERROR."""
    print(f"[ERROR] {msg}")


def log_eval(msg):
    """Xuat log muc EVAL."""
    print(f"[EVAL] {msg}")


def log(tag, msg):
    """Tuong thich nguoc."""
    print(f"[{tag}] {msg}")


def format_confusion_matrix(cm, classes=None):
    """Dinh dang ma tran nham lan thanh khoi van ban gon gang, can hang cot deu dep."""
    if classes is None:
        classes = list(range(len(cm)))
    header = "       " + " ".join([f"{c:>4d}" for c in classes])
    lines = [header]
    for idx, c in enumerate(classes):
        row_str = " ".join([f"{cm[idx, j]:>4d}" for j in range(len(classes))])
        lines.append(f"  {c:>3d} [ {row_str} ]")
    return "\n".join(lines)


def print_table(headers, rows, alignments=None):
    """In bang ASCII tieu chuan, can le dep cho cac chi so ML."""
    if alignments is None:
        alignments = ["<"] + [">"] * (len(headers) - 1)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    sep = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    header_str = (
        "|"
        + "|".join([f" {h:^{col_widths[i]}} " for i, h in enumerate(headers)])
        + "|"
    )

    print(sep)
    print(header_str)
    print(sep)
    for row in rows:
        row_str = "|"
        for i, (val, align) in enumerate(zip(row, alignments)):
            w = col_widths[i]
            val_str = str(val)
            if align == ">":
                cell = f" {val_str:>{w}} "
            elif align == "^":
                cell = f" {val_str:^{w}} "
            else:
                cell = f" {val_str:<{w}} "
            row_str += cell + "|"
        print(row_str)
    print(sep)


def save_single_confusion(yte, y_pred, title, path="figs/confusion.png"):
    """Luu 1 ma tran nham lan duy nhat cho mo hinh don."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(yte, y_pred, cmap="Blues", ax=ax)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.replace("\\", "/")


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
    os.makedirs(fig_dir, exist_ok=True)
    saved_paths = []

    # Bieu do 1: Accuracy theo T
    T_grid = np.arange(1, adaboost_t + 1)
    ada_tr = [accuracy_score(ytr, p) for p in ada.staged_predict(Xtr)]
    ada_te = [accuracy_score(yte, p) for p in ada.staged_predict(Xte)]
    bag_te = [accuracy_score(yte, p) for p in bag.staged_predict(Xte)]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(T_grid, ada_tr, "--", c="tab:red", label="AdaBoost (SAMME) - Train")
    ax.plot(T_grid, ada_te, c="tab:red", lw=1.8, label="AdaBoost (SAMME) - Test")
    ax.plot(np.arange(1, bagging_t + 1), bag_te, c="tab:blue", lw=1.8, label="Bagging - Test")
    ax.axhline(res["Stump (1 tang)"][1], c="gray", ls=":", label="Stump don - Test")
    ax.set_xlabel("So luong mo hinh (T)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.15, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Do chinh xac theo so luong mo hinh trong Ensemble (T) - Digits")
    fig.tight_layout()
    p1 = os.path.join(fig_dir, "acc_vs_T.png").replace("\\", "/")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_paths.append(p1)

    # Bieu do 2: So sanh Test Accuracy
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
    p2 = os.path.join(fig_dir, "compare.png").replace("\\", "/")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved_paths.append(p2)

    # Bieu do 3: Ma tran nham lan cua Stacking
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(yte, stk.predict(Xte), cmap="Blues", ax=ax)
    ax.set_title(f"Ma tran nham lan - Stacking Classifier (Acc: {res[stk_name][1]:.4f})", fontsize=10)
    fig.tight_layout()
    p3 = os.path.join(fig_dir, "confusion.png").replace("\\", "/")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_paths.append(p3)

    return saved_paths


def save_stability_plot(names, means, stds, n, fig_dir="figs"):
    """Xuat bieu do so sanh do on dinh Mean +- Std qua n lan lap."""
    os.makedirs(fig_dir, exist_ok=True)
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
    ax.set_xlabel(f"Test Accuracy (Mean +/- Std qua {n} lan)")
    ax.set_title("Do on dinh cua cac mo hinh qua cac lan phan chia ngau nhien (Digits)")
    fig.tight_layout()
    p4 = os.path.join(fig_dir, "compare_repeated.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    return p4
