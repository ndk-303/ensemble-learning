"""
Script huan luyen va danh gia Ensemble Learning tren Scikit-Learn Digits.
Ho tro dung 6 tham so dong lenh (argparse):
  --method:        bagging | boosting | stacking | all (Bat buoc)
  --base_model:    dt | knn | svm | lr (Stacking ho tro nhieu model cach nhau boi dau phay)
  --seed:          int >= 0
  --test_size:     0.0 < float < 1.0
  --n_estimators:  int >= 1 (ap dung cho bagging, boosting, all)
  --learning_rate: float > 0 (ap dung cho boosting, all)
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np

# Dam bao tuong thich ma hoa console tren Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.metrics import accuracy_score

from src.methods import Bagging, AdaBoost, Stacking
from src.data import load_data, build_model, get_base_models, SUPPORTED_MODELS
from src.utils import (
    log_info,
    log_train,
    log_done,
    log_warn,
    log_error,
    log_eval,
    print_table,
    save_single_confusion,
    save_visualizations,
)


def parse_args():
    """Khoi tao argparse voi dung 6 tham so va kiem tra tinh hop le."""
    parser = argparse.ArgumentParser(
        description="Ensemble Learning (Bagging, AdaBoost SAMME, Stacking) tren Scikit-Learn Digits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 1. --method (Bat buoc)
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["bagging", "boosting", "stacking", "all"],
        help="Phuong phap Ensemble: bagging | boosting | stacking | all",
    )

    # 2. --base_model
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model: dt, knn, svm, lr. (Bagging: 1 model; Boosting: chi ho tro dt; Stacking: danh sach vi du dt,knn,svm)",
    )

    # 3. --seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed cho phan chia du lieu va khoi tao mo hinh (int >= 0)",
    )

    # 4. --test_size
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Ty le tap kiem thu test_size (0.0 < float < 1.0)",
    )

    # 5. --n_estimators
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="So luong estimators (int >= 1, ap dung cho bagging va boosting)",
    )

    # 6. --learning_rate
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Toc do hoc learning_rate (float > 0, chi ap dung cho boosting)",
    )

    args = parser.parse_args()

    # --- RANG BUOC GIA TRI CHUNG ---
    if args.seed < 0:
        parser.error("Tham so --seed phai la so nguyen >= 0.")

    if not (0.0 < args.test_size < 1.0):
        parser.error("Tham so --test_size phai thoa man 0.0 < test_size < 1.0.")

    if args.n_estimators is not None and args.n_estimators < 1:
        parser.error("Tham so --n_estimators phai la so nguyen >= 1.")

    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("Tham so --learning_rate phai la so thuc > 0.")

    # --- RANG BUOC THEO TUNG METHOD ---
    method = args.method.lower()

    if method == "all":
        pass

    elif method == "stacking":
        if args.n_estimators is not None:
            parser.error("Tham so --n_estimators khong ap dung cho phuong phap 'stacking'.")
        if args.learning_rate is not None:
            parser.error("Tham so --learning_rate khong ap dung cho phuong phap 'stacking'.")

        if args.base_model is None:
            args.base_model = "dt,knn,svm"

        model_tokens = [m.strip().lower() for m in args.base_model.split(",") if m.strip()]
        if not model_tokens:
            parser.error("Phuong phap 'stacking' can it nhat 1 base model hop le.")
        for m in model_tokens:
            if m not in SUPPORTED_MODELS:
                parser.error(f"Base model '{m}' khong hop le. Cac model ho tro: {', '.join(SUPPORTED_MODELS)}.")
        args.parsed_models = model_tokens

    elif method == "bagging":
        if args.learning_rate is not None:
            parser.error("Tham so --learning_rate khong ap dung cho phuong phap 'bagging'.")

        if args.n_estimators is None:
            args.n_estimators = 50

        if args.base_model is None:
            args.base_model = "dt"
        else:
            args.base_model = args.base_model.strip().lower()
            if "," in args.base_model:
                parser.error("Phuong phap 'bagging' chi nhan 1 base model duy nhat, vi du: --base_model dt")
            if args.base_model not in SUPPORTED_MODELS:
                parser.error(f"Base model '{args.base_model}' khong hop le. Cac model ho tro: {', '.join(SUPPORTED_MODELS)}.")

    elif method == "boosting":
        if args.n_estimators is None:
            args.n_estimators = 100

        if args.learning_rate is None:
            args.learning_rate = 1.0

        if args.base_model is None:
            args.base_model = "dt"
        else:
            args.base_model = args.base_model.strip().lower()
            if args.base_model != "dt":
                parser.error("Phuong phap 'boosting' (AdaBoost) chi ho tro base model 'dt' (Decision Stump voi max_depth=1).")

    return args


def get_base_model_label(code):
    """Chuyen doi ma model thanh ten hien thi de doc."""
    mapping = {
        "dt": "Decision Tree",
        "knn": "K-Nearest Neighbors (k=5)",
        "svm": "Support Vector Machine (RBF)",
        "lr": "Logistic Regression",
    }
    return mapping.get(code.lower(), code.upper())


def train_single_method(args):
    """Huan luyen 1 phuong phap Ensemble don le theo dung 5 phan tieu chuan."""
    method = args.method.lower()

    # ==================================================
    # 1. CONFIGURATION (Chi hien thi argument ap dung cho method hien tai)
    # ==================================================
    print("=" * 50)
    print(" ENSEMBLE LEARNING | TRAINING")
    print("=" * 50)

    if method == "bagging":
        print(f" Method       : Bagging")
        print(f" Base Model   : {get_base_model_label(args.base_model)}")
        print(f" Dataset      : Scikit-Learn Digits (0-9)")
        print(f" Seed         : {args.seed}")
        print(f" Test Size    : {args.test_size}")
        print(f" N Estimators : {args.n_estimators}")
    elif method == "boosting":
        print(f" Method       : Boosting (AdaBoost SAMME)")
        print(f" Base Model   : Decision Tree (Stump, max_depth=1)")
        print(f" Dataset      : Scikit-Learn Digits (0-9)")
        print(f" Seed         : {args.seed}")
        print(f" Test Size    : {args.test_size}")
        print(f" N Estimators : {args.n_estimators}")
        print(f" Learning Rate: {args.learning_rate}")
    elif method == "stacking":
        base_labels = ", ".join([get_base_model_label(m) for m in args.parsed_models])
        print(f" Method       : Stacking")
        print(f" Base Models  : {base_labels}")
        print(f" Meta-Model   : Logistic Regression (max_iter=2000)")
        print(f" CV Folds     : 5 (Out-of-Fold)")
        print(f" Dataset      : Scikit-Learn Digits (0-9)")
        print(f" Seed         : {args.seed}")
        print(f" Test Size    : {args.test_size}")
    print("-" * 50)

    # ==================================================
    # 2. DATASET
    # ==================================================
    Xtr, ytr, Xte, yte, _ = load_data(test_size=args.test_size, seed=args.seed)
    classes = np.unique(ytr)
    n_classes = len(classes)
    log_info(f"Dataset loaded: {len(ytr)+len(yte)} samples, {Xtr.shape[1]} features (10 classes: 0..9)")
    log_info(f"Train: {len(ytr)} | Test: {len(yte)}")

    # ==================================================
    # 3. TRAINING
    # ==================================================
    if method == "bagging":
        log_info(f"Initializing BaggingClassifier ({args.n_estimators} estimators)...")
        log_train("Training started...")

        def bagging_callback(curr, total):
            step = max(1, total // 5)
            if curr == 1 or curr == total or curr % step == 0:
                pct = int((curr / total) * 100)
                log_train(f"Fitting estimators: {curr}/{total} ({pct}%)")

        base_est = build_model(args.base_model, mode="bagging", seed=args.seed)
        model = Bagging(base_estimator=base_est, T=args.n_estimators, seed=args.seed)

        t0 = time.perf_counter()
        model.fit(Xtr, ytr, callback=bagging_callback)
        train_time = time.perf_counter() - t0
        log_done(f"Training completed in {train_time:.2f}s")

    elif method == "boosting":
        log_info(f"Initializing AdaBoost (SAMME, {args.n_estimators} weak learners, lr={args.learning_rate})...")
        log_train("Training started...")

        def boosting_callback(curr, total):
            step = max(1, total // 5)
            if curr == 1 or curr == total or curr % step == 0:
                pct = int((curr / total) * 100)
                log_train(f"Iterating weak learners: {curr}/{total} ({pct}%)")

        base_est = build_model("dt", mode="boosting", seed=args.seed)
        model = AdaBoost(
            base_estimator=base_est,
            T=args.n_estimators,
            learning_rate=args.learning_rate,
            random_state=args.seed,
        )

        t0 = time.perf_counter()
        model.fit(Xtr, ytr, callback=boosting_callback)
        train_time = time.perf_counter() - t0
        log_done(f"Training completed in {train_time:.2f}s")

    elif method == "stacking":
        log_info("Initializing StackingClassifier...")
        log_train("Training started...")

        def stacking_callback(phase):
            if phase == "phase_1":
                log_train("[Phase 1/3] Generating Out-Of-Fold predictions via 5-Fold CV...")
            elif phase == "phase_2":
                log_train("[Phase 2/3] Fitting Level-2 Meta-Learner (Logistic Regression)...")
            elif phase == "phase_3":
                log_train("[Phase 3/3] Refitting base estimators on full training data...")

        base_estimators = [build_model(m, mode="stacking", seed=args.seed) for m in args.parsed_models]
        meta_est = build_model("lr", seed=args.seed)
        model = Stacking(base_estimators=base_estimators, meta_estimator=meta_est, k=5, seed=args.seed)

        t0 = time.perf_counter()
        model.fit(Xtr, ytr, callback=stacking_callback)
        train_time = time.perf_counter() - t0
        log_done(f"Training completed in {train_time:.2f}s")

    # ==================================================
    # 4. EVALUATION
    # ==================================================
    print()
    y_pred = model.predict(Xte)
    acc = accuracy_score(yte, y_pred)

    log_eval(f"Accuracy      : {acc:.4f} ({acc*100:.2f}%)")
    log_eval(f"Training Time : {train_time:.2f}s")

    # ==================================================
    # 5. ARTIFACTS
    # ==================================================
    print()
    cf_path = save_single_confusion(
        yte=yte,
        y_pred=y_pred,
        title=f"Confusion Matrix - {method.upper()} (Acc: {acc:.4f})",
        path="figs/confusion.png",
    )
    print("=" * 50)


def run_all(args):
    """Huan luyen va so sanh toan bo cac phuong phap theo dung 5 phan tieu chuan."""
    bag_t = args.n_estimators if args.n_estimators is not None else 50
    ada_t = args.n_estimators if args.n_estimators is not None else 100
    ada_lr = args.learning_rate if args.learning_rate is not None else 1.0

    # ==================================================
    # 1. CONFIGURATION
    # ==================================================
    print("=" * 50)
    print(" ENSEMBLE LEARNING | TRAINING (ALL METHODS)")
    print("=" * 50)
    print(f" Method       : All (Comparison Mode)")
    print(f" Dataset      : Scikit-Learn Digits (0-9)")
    print(f" Seed         : {args.seed}")
    print(f" Test Size    : {args.test_size}")
    print(f" N Estimators : {bag_t} (Bagging) / {ada_t} (Boosting)")
    print(f" Learning Rate: {ada_lr} (Boosting)")
    print(f" Stacking     : Base (DT, KNN, SVM) -> Meta (LR, cv=5)")
    print("-" * 50)

    # ==================================================
    # 2. DATASET
    # ==================================================
    Xtr, ytr, Xte, yte, _ = load_data(test_size=args.test_size, seed=args.seed)
    classes = np.unique(ytr)
    log_info(f"Dataset loaded: {len(ytr)+len(yte)} samples, {Xtr.shape[1]} features (10 classes: 0..9)")
    log_info(f"Train: {len(ytr)} | Test: {len(yte)}")

    # ==================================================
    # 3. TRAINING
    # ==================================================
    log_info("Initializing all base and ensemble models...")
    log_train("Training started...")
    t_start = time.perf_counter()

    models = get_base_models(seed=args.seed)
    fit_times = {}

    for idx, (name, m) in enumerate(list(models.items()), 1):
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        fit_times[name] = time.perf_counter() - t0
        log_train(f"[{idx}/7] Fitting {name}... done in {fit_times[name]:.4f}s")

    # Bagging
    t0 = time.perf_counter()
    bag = Bagging(base_estimator=build_model("dt", mode="bagging", seed=args.seed), T=bag_t, seed=args.seed).fit(Xtr, ytr)
    bag_name = f"Bagging ({bag_t} cay)"
    fit_times[bag_name] = time.perf_counter() - t0
    models[bag_name] = bag
    log_train(f"[5/7] Fitting {bag_name}... done in {fit_times[bag_name]:.4f}s")

    # AdaBoost
    t0 = time.perf_counter()
    ada = AdaBoost(base_estimator=build_model("dt", mode="boosting", seed=args.seed), T=ada_t, learning_rate=ada_lr, random_state=args.seed).fit(Xtr, ytr)
    ada_name = f"AdaBoost ({ada_t} stump)"
    fit_times[ada_name] = time.perf_counter() - t0
    models[ada_name] = ada
    log_train(f"[6/7] Fitting {ada_name}... done in {fit_times[ada_name]:.4f}s")

    # Stacking
    t0 = time.perf_counter()
    stk_bases = [build_model("dt", mode="stacking", seed=args.seed), build_model("knn", seed=args.seed), build_model("svm", seed=args.seed)]
    meta_est = build_model("lr", seed=args.seed)
    stk = Stacking(base_estimators=stk_bases, meta_estimator=meta_est, k=5, seed=args.seed).fit(Xtr, ytr)
    stk_name = "Stacking (DT+KNN+SVM -> LR)"
    fit_times[stk_name] = time.perf_counter() - t0
    models[stk_name] = stk
    log_train(f"[7/7] Fitting {stk_name}... done in {fit_times[stk_name]:.4f}s")

    total_train_time = time.perf_counter() - t_start
    log_done(f"Training completed in {total_train_time:.2f}s")

    # ==================================================
    # 4. EVALUATION
    # ==================================================
    print()
    log_eval("Comparison Table (Accuracy & Training Time):")
    eval_rows = []
    res = {}
    for name, m in models.items():
        tr_acc = accuracy_score(ytr, m.predict(Xtr))
        y_pred = m.predict(Xte)
        te_acc = accuracy_score(yte, y_pred)
        t_fit = fit_times.get(name, 0.0)
        res[name] = (tr_acc, te_acc)
        eval_rows.append([name, f"{tr_acc:.4f}", f"{te_acc:.4f}", f"{t_fit:.4f}s"])

    print_table(
        headers=["Mo hinh", "Train Acc", "Test Acc", "Fit Time"],
        rows=eval_rows,
        alignments=["<", ">", ">", ">"],
    )

    # ==================================================
    # 5. ARTIFACTS
    # ==================================================
    print()
    saved = save_visualizations(
        res=res,
        ytr=ytr,
        yte=yte,
        Xtr=Xtr,
        Xte=Xte,
        ada=ada,
        bag=bag,
        stk=stk,
        adaboost_t=ada_t,
        bagging_t=bag_t,
        stk_name=stk_name,
        fig_dir="figs",
    )
    print("=" * 50)


def main():
    args = parse_args()
    if args.method.lower() == "all":
        run_all(args)
    else:
        train_single_method(args)


if __name__ == "__main__":
    main()
