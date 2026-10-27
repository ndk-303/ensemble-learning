"""
Module tien ich: Dinh dang log chuan ML, in bang ASCII va xuat bieu do truc quan hoa.
"""
import os
import sys
from datetime import datetime
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


def log(tag, msg):
    """Xuat log co dinh dang thoi gian va nhan tag chuan ML."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{tag:<7s}] {msg}")


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
    """Xuat 3 bieu do tinh gon cho thi nghiem don: acc_vs_T, compare, confusion."""
    os.makedirs(fig_dir, exist_ok=True)
    log("VISUAL", "Dang xuat cac bieu do truc quan hoa vao thu muc figs/...")

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
    p1 = os.path.join(fig_dir, "acc_vs_T.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    log("VISUAL", f"  -> Da luu: {p1}")

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
    p2 = os.path.join(fig_dir, "compare.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    log("VISUAL", f"  -> Da luu: {p2}")

    # Bieu do 3: Ma tran nham lan cua Stacking
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(yte, stk.predict(Xte), cmap="Blues", ax=ax)
    ax.set_title(f"Ma tran nham lan - Stacking Classifier (Acc: {res[stk_name][1]:.4f})", fontsize=10)
    fig.tight_layout()
    p3 = os.path.join(fig_dir, "confusion.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    log("VISUAL", f"  -> Da luu: {p3}")


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
    log("VISUAL", f"  -> Da luu: {p4}")
