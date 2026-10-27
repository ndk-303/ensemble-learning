"""
Script huan luyen va danh gia cac phuong phap Ensemble Learning tren Scikit-Learn Digits.

Cach su dung:
  python train.py
  python train.py --help
  python train.py --bagging_t 50 --adaboost_t 100 --repeats 10
  python train.py --repeats 0
  python train.py --no_plots
"""
import os
import time
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier

from src.methods import Bagging, AdaBoost, Stacking
from src.data import load_data, get_base_models
from src.utils import log, print_table, save_visualizations, save_stability_plot


def run_experiment(
    seed=42,
    test_size=0.3,
    bagging_t=50,
    adaboost_t=100,
    stacking_k=5,
    fig_dir="figs",
    save_plots=True,
):
    """Chay 1 vong huan luyen, danh gia va doi chieu voi Scikit-Learn."""
    log("DATA", f"Tai tap du lieu Scikit-Learn Digits (Stratified Split: {int((1-test_size)*100)}% Train / {int(test_size*100)}% Test)...")
    Xtr, ytr, Xte, yte, _ = load_data(test_size=test_size, seed=seed)
    classes = np.unique(ytr)
    n_classes = len(classes)
    log("DATA", f"Tong mau: {len(ytr)+len(yte)} | Train: {len(ytr)} | Test: {len(yte)} | So dac trung: {Xtr.shape[1]} | So lop: {n_classes}")

    # 1. Khoi tao va huan luyen cac mo hinh co so
    log("TRAIN", "Bat dau huan luyen cac mo hinh co so & mo hinh Ensemble...")
    models = get_base_models(seed=0)
    fit_times = {}

    for idx, (name, model) in enumerate(list(models.items()), 1):
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_times[name] = time.perf_counter() - t0
        log("TRAIN", f"[{idx}/7] Da huan luyen {name:<28s} trong {fit_times[name]:.4f}s")

    # 2. Khoi tao va huan luyen cac phuong phap Ensemble tu cai dat
    t0 = time.perf_counter()
    bag = Bagging(
        base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=0),
        T=bagging_t,
        seed=0,
    ).fit(Xtr, ytr)
    bag_name = f"Bagging ({bagging_t} cay)"
    fit_times[bag_name] = time.perf_counter() - t0
    models[bag_name] = bag
    log("TRAIN", f"[5/7] Da huan luyen {bag_name:<28s} trong {fit_times[bag_name]:.4f}s")

    t0 = time.perf_counter()
    ada = AdaBoost(
        base_estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
        T=adaboost_t,
        random_state=0,
    ).fit(Xtr, ytr)
    ada_name = f"AdaBoost ({adaboost_t} stump)"
    fit_times[ada_name] = time.perf_counter() - t0
    models[ada_name] = ada
    log("TRAIN", f"[6/7] Da huan luyen {ada_name:<28s} trong {fit_times[ada_name]:.4f}s")

    t0 = time.perf_counter()
    stacking_bases = [
        DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=0),
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)),
    ]
    stk = Stacking(
        base_estimators=stacking_bases,
        meta_estimator=LogisticRegression(max_iter=2000, random_state=0),
        k=stacking_k,
        seed=0,
    ).fit(Xtr, ytr)
    stk_name = "Stacking (Tree+KNN+LR -> LR)"
    fit_times[stk_name] = time.perf_counter() - t0
    models[stk_name] = stk
    log("TRAIN", f"[7/7] Da huan luyen {stk_name:<28s} trong {fit_times[stk_name]:.4f}s")

    # 3. Danh gia Train Acc & Test Acc
    log("EVAL", "Tong hop ket qua danh gia do chinh xac (Accuracy) & thoi gian fit:")
    eval_rows = []
    res = {}
    for name, model in models.items():
        tr_acc = accuracy_score(ytr, model.predict(Xtr))
        te_acc = accuracy_score(yte, model.predict(Xte))
        res[name] = (tr_acc, te_acc)
        eval_rows.append([name, f"{tr_acc:.4f}", f"{te_acc:.4f}", f"{fit_times.get(name, 0):.4f}s"])

    print_table(
        headers=["Mo hinh", "Train Acc", "Test Acc", "Fit Time"],
        rows=eval_rows,
        alignments=["<", ">", ">", ">"],
    )

    # 4. Doi chieu truc tiep voi Scikit-Learn
    log("BENCHMARK", "Kiem thu & doi chieu ket qua truc tiep voi Scikit-Learn:")
    sk_models = {
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)),
        bag_name: BaggingClassifier(
            estimator=DecisionTreeClassifier(criterion="entropy", max_depth=12, random_state=0),
            n_estimators=bagging_t,
            random_state=0,
        ),
        ada_name: AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
            n_estimators=adaboost_t,
            random_state=0,
        ),
        stk_name: StackingClassifier(
            estimators=[
                ("tree", DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=0)),
                ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
                ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))),
            ],
            final_estimator=LogisticRegression(max_iter=2000, random_state=0),
            cv=stacking_k,
        ),
    }

    bench_rows = []
    for name, sk_model in sk_models.items():
        sk_acc = accuracy_score(yte, sk_model.fit(Xtr, ytr).predict(Xte))
        self_acc = res[name][1]
        diff = abs(self_acc - sk_acc)
        status = "KHOP 100%" if diff < 1e-5 else ("TUONG DUONG" if diff < 0.01 else "LECH")
        bench_rows.append([name, f"{self_acc:.4f}", f"{sk_acc:.4f}", f"{diff:+.4f}", status])

    print_table(
        headers=["Mo hinh", "Tu cai dat", "Scikit-Learn", "Chenh lech", "Trang thai"],
        rows=bench_rows,
        alignments=["<", ">", ">", ">", "^"],
    )

    # 5. Thong ke ma tran nham lan
    log("EVAL", "Thong ke so luong mau du doan dung tren tap Test (540 mau):")
    for name, model in models.items():
        cm = confusion_matrix(yte, model.predict(Xte), labels=classes)
        correct = int(np.trace(cm))
        total = int(np.sum(cm))
        pct = (correct / total) * 100
        print(f"       • {name:<30s} -> Dung: {correct:3d}/{total} ({pct:.2f}%)")

    # 6. Luu bieu do
    if save_plots:
        save_visualizations(
            res=res,
            ytr=ytr,
            yte=yte,
            Xtr=Xtr,
            Xte=Xte,
            ada=ada,
            bag=bag,
            stk=stk,
            adaboost_t=adaboost_t,
            bagging_t=bagging_t,
            stk_name=stk_name,
            fig_dir=fig_dir,
        )


def run_repeated(n=10, test_size=0.3, fig_dir="figs", save_plots=True):
    """Lap n lan phan chia ngau nhien de danh gia do on dinh (Mean +- Std)."""
    if n <= 0:
        log("STABILITY", "So lan lap repeats <= 0, bo qua danh gia do on dinh.")
        return

    log("STABILITY", f"Bat dau danh gia do on dinh qua {n} lan phan chia ngau nhien (Monte-Carlo Simulation)...")
    names = ["Stump", "Cay sau", "KNN (5)", "LogReg", "Bagging (25)", "AdaBoost (50)", "Stacking"]
    records = {k: [] for k in names}

    for sd in range(n):
        t_start = time.perf_counter()
        Xtr, ytr, Xte, yte, _ = load_data(test_size=test_size, seed=sd)
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

        dt = time.perf_counter() - t_start
        log("REPEAT", f"[{sd+1:02d}/{n:02d}] Seed {sd:<2d} | Stacking: {records['Stacking'][-1]:.4f} | Bagging: {records['Bagging (25)'][-1]:.4f} | AdaBoost: {records['AdaBoost (50)'][-1]:.4f} ({dt:.2f}s)")

    log("STABILITY", f"Tong hop ket qua sau {n} lan phan chia ngau nhien:")
    means = [np.mean(records[k]) for k in names]
    stds = [np.std(records[k]) for k in names]
    mins = [np.min(records[k]) for k in names]
    maxs = [np.max(records[k]) for k in names]

    rep_rows = []
    for k, m, s_, mi, ma in zip(names, means, stds, mins, maxs):
        rep_rows.append([k, f"{m:.4f} +/- {s_:.4f}", f"{mi:.4f}", f"{ma:.4f}"])

    print_table(
        headers=["Mo hinh", "Test Acc (Mean +/- Std)", "Min Acc", "Max Acc"],
        rows=rep_rows,
        alignments=["<", ">", ">", ">"],
    )

    if save_plots:
        save_stability_plot(names=names, means=means, stds=stds, n=n, fig_dir=fig_dir)


def parse_args():
    """Doc va kiem tra cac tham so dong lenh."""
    parser = argparse.ArgumentParser(
        description="Ensemble Learning Benchmark: Bagging, AdaBoost (SAMME), Stacking tren Scikit-Learn Digits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed cho phan chia du lieu va base models")
    parser.add_argument("--test_size", type=float, default=0.3, help="Ty le tap test (0.3 = 30%%)")
    parser.add_argument("--bagging_t", type=int, default=50, help="So luong cay quyet dinh trong Bagging")
    parser.add_argument("--adaboost_t", type=int, default=100, help="So luong Decision Stump trong AdaBoost SAMME")
    parser.add_argument("--stacking_k", type=int, default=5, help="So fold K-Fold Out-of-Fold (OOF) trong Stacking")
    parser.add_argument("--repeats", type=int, default=10, help="So lan lap ngau nhien de do do on dinh (0 de bo qua)")
    parser.add_argument("--fig_dir", type=str, default="figs", help="Thu muc luu tru bieu do")
    parser.add_argument("--no_plots", action="store_true", help="Bo qua buoc ve va xuat bieu do ra dia")
    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.perf_counter()

    print("\n" + "=" * 80)
    print(" ENSEMBLE LEARNING BENCHMARK PIPELINE: BAGGING - ADABOOST - STACKING")
    print("=" * 80)
    log("CONFIG", "Cau hinh tham so thuc nghiem:")
    print(f"       • Random Seed         : {args.seed}")
    print(f"       • Test Size           : {args.test_size} ({int(args.test_size*100)}%)")
    print(f"       • Bagging Estimators  : {args.bagging_t} cay (entropy, max_depth=12)")
    print(f"       • AdaBoost Estimators : {args.adaboost_t} stump (SAMME multi-class)")
    print(f"       • Stacking Folds      : {args.stacking_k}-fold OOF CV")
    print(f"       • Repeats Evaluation  : {args.repeats} lan")
    print(f"       • Figure Output Dir   : {args.fig_dir}")
    print(f"       • Save Figures        : {not args.no_plots}")
    print("=" * 80)

    # 1. Chay thuc nghiem don
    run_experiment(
        seed=args.seed,
        test_size=args.test_size,
        bagging_t=args.bagging_t,
        adaboost_t=args.adaboost_t,
        stacking_k=args.stacking_k,
        fig_dir=args.fig_dir,
        save_plots=not args.no_plots,
    )

    # 2. Chay thuc nghiem lap
    if args.repeats > 0:
        run_repeated(
            n=args.repeats,
            test_size=args.test_size,
            fig_dir=args.fig_dir,
            save_plots=not args.no_plots,
        )

    total_time = time.perf_counter() - total_start
    print("=" * 80)
    log("DONE", f"Hoan thanh toan bo pipeline trong {total_time:.2f} giay.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
