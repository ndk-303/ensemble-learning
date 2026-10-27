"""
Script huan luyen va danh gia Ensemble Learning tren Scikit-Learn Digits.
Ho tro dung 6 tham so dong lenh (argparse):
  --method:        bagging | boosting | stacking (Bat buoc)
  --base_model:    dt | knn | svm | lr (Stacking ho tro nhieu model cach nhau boi dau phay)
  --seed:          int >= 0
  --test_size:     0.0 < float < 1.0
  --n_estimators:  int >= 1 (chi ap dung cho bagging, boosting)
  --learning_rate: float > 0 (chi ap dung cho boosting)
"""
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

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier

from src.methods import Bagging, AdaBoost, Stacking
from src.data import load_data, build_model, SUPPORTED_MODELS
from src.utils import log, print_table


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
        choices=["bagging", "boosting", "stacking"],
        help="Phuong phap Ensemble: bagging | boosting | stacking",
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

    if method == "stacking":
        # Argument khong hop le cho Stacking
        if args.n_estimators is not None:
            parser.error("Tham so --n_estimators khong ap dung cho phuong phap 'stacking'.")
        if args.learning_rate is not None:
            parser.error("Tham so --learning_rate khong ap dung cho phuong phap 'stacking'.")

        # Base model cho Stacking: mac dinh dt,knn,svm
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
        # Argument khong hop le cho Bagging
        if args.learning_rate is not None:
            parser.error("Tham so --learning_rate khong ap dung cho phuong phap 'bagging'.")

        # n_estimators mac dinh cho Bagging neu khong truyen
        if args.n_estimators is None:
            args.n_estimators = 50

        # Base model cho Bagging: chi nhan 1 base model, mac dinh dt
        if args.base_model is None:
            args.base_model = "dt"
        else:
            args.base_model = args.base_model.strip().lower()
            if "," in args.base_model:
                parser.error("Phuong phap 'bagging' chi nhan 1 base model duy nhat, vi du: --base_model dt")
            if args.base_model not in SUPPORTED_MODELS:
                parser.error(f"Base model '{args.base_model}' khong hop le. Cac model ho tro: {', '.join(SUPPORTED_MODELS)}.")

    elif method == "boosting":
        # n_estimators mac dinh cho Boosting neu khong truyen
        if args.n_estimators is None:
            args.n_estimators = 100

        # learning_rate mac dinh cho Boosting neu khong truyen
        if args.learning_rate is None:
            args.learning_rate = 1.0

        # Base model cho Boosting: chi ho tro dt (Decision Stump voi max_depth=1)
        if args.base_model is None:
            args.base_model = "dt"
        else:
            args.base_model = args.base_model.strip().lower()
            if args.base_model != "dt":
                parser.error("Phuong phap 'boosting' (AdaBoost) chi ho tro base model 'dt' (Decision Stump voi max_depth=1).")

    return args


def train_and_evaluate(args):
    """Huan luyen mo hinh tu cai dat va doi chieu voi Scikit-Learn."""
    total_start = time.perf_counter()
    method = args.method.lower()

    print("\n" + "=" * 80)
    print(f" ENSEMBLE LEARNING PIPELINE: {method.upper()}")
    print("=" * 80)
    log("CONFIG", "Cau hinh tham so thuc nghiem:")
    print(f"       • Method              : {method}")
    print(f"       • Base Model(s)       : {args.base_model}")
    print(f"       • Random Seed         : {args.seed}")
    print(f"       • Test Size           : {args.test_size} ({int(args.test_size*100)}%)")
    if method in ["bagging", "boosting"]:
        print(f"       • Estimators (T)      : {args.n_estimators}")
    if method == "boosting":
        print(f"       • Learning Rate       : {args.learning_rate}")
    if method == "stacking":
        print(f"       • Meta-Model          : Logistic Regression (max_iter=2000)")
        print(f"       • Cross-Validation    : cv=5 (Out-of-Fold)")
    print("=" * 80)

    # 1. Tai va chia du lieu
    log("DATA", f"Tai tap du lieu Scikit-Learn Digits (Stratified Split: {int((1-args.test_size)*100)}% Train / {int(args.test_size*100)}% Test)...")
    Xtr, ytr, Xte, yte, _ = load_data(test_size=args.test_size, seed=args.seed)
    classes = np.unique(ytr)
    n_classes = len(classes)
    log("DATA", f"Tong mau: {len(ytr)+len(yte)} | Train: {len(ytr)} | Test: {len(yte)} | So dac trung: {Xtr.shape[1]} | So lop: {n_classes}")

    # 2. Khoi tao mo hinh tu cai dat va Scikit-Learn tuong ung
    if method == "bagging":
        base_est = build_model(args.base_model, mode="bagging", seed=args.seed)
        custom_model = Bagging(base_estimator=base_est, T=args.n_estimators, seed=args.seed)
        sk_model = BaggingClassifier(estimator=base_est, n_estimators=args.n_estimators, random_state=args.seed)
        model_display_name = f"Bagging ({args.n_estimators} x {args.base_model.upper()})"

    elif method == "boosting":
        base_est = build_model("dt", mode="boosting", seed=args.seed)
        custom_model = AdaBoost(
            base_estimator=base_est,
            T=args.n_estimators,
            learning_rate=args.learning_rate,
            random_state=args.seed,
        )
        sk_model = AdaBoostClassifier(
            estimator=base_est,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            random_state=args.seed,
        )
        model_display_name = f"AdaBoost ({args.n_estimators} stump, lr={args.learning_rate})"

    elif method == "stacking":
        base_estimators = [build_model(m, mode="stacking", seed=args.seed) for m in args.parsed_models]
        meta_est = build_model("lr", seed=args.seed)
        custom_model = Stacking(base_estimators=base_estimators, meta_estimator=meta_est, k=5, seed=args.seed)
        sk_estimators = [(f"{m}_{i}", est) for i, (m, est) in enumerate(zip(args.parsed_models, base_estimators))]
        sk_model = StackingClassifier(estimators=sk_estimators, final_estimator=meta_est, cv=5)
        model_display_name = f"Stacking ({'+'.join(m.upper() for m in args.parsed_models)} -> LR, cv=5)"

    # 3. Huan luyen mo hinh tu cai dat
    log("TRAIN", f"Dang huan luyen mo hinh tu cai dat: {model_display_name}...")
    t0 = time.perf_counter()
    custom_model.fit(Xtr, ytr)
    custom_fit_time = time.perf_counter() - t0
    log("TRAIN", f"Da huan luyen xong trong {custom_fit_time:.4f}s")

    # 4. Huan luyen mo hinh Scikit-Learn de doi chieu
    log("BENCHMARK", "Dang huan luyen mo hinh Scikit-Learn tuong ung...")
    t0 = time.perf_counter()
    sk_model.fit(Xtr, ytr)
    sk_fit_time = time.perf_counter() - t0
    log("BENCHMARK", f"Da huan luyen Scikit-Learn xong trong {sk_fit_time:.4f}s")

    # 5. Danh gia tren Train va Test
    log("EVAL", "Danh gia do chinh xac (Accuracy) tren tap Train va Test:")
    custom_tr_acc = accuracy_score(ytr, custom_model.predict(Xtr))
    custom_te_acc = accuracy_score(yte, custom_model.predict(Xte))
    sk_tr_acc = accuracy_score(ytr, sk_model.predict(Xtr))
    sk_te_acc = accuracy_score(yte, sk_model.predict(Xte))

    eval_rows = [
        [f"{model_display_name} (Tu cai dat)", f"{custom_tr_acc:.4f}", f"{custom_te_acc:.4f}", f"{custom_fit_time:.4f}s"],
        [f"{model_display_name} (Scikit-Learn)", f"{sk_tr_acc:.4f}", f"{sk_te_acc:.4f}", f"{sk_fit_time:.4f}s"],
    ]

    print_table(
        headers=["Mo hinh", "Train Acc", "Test Acc", "Fit Time"],
        rows=eval_rows,
        alignments=["<", ">", ">", ">"],
    )

    # 6. So sanh Test Accuracy
    diff = abs(custom_te_acc - sk_te_acc)
    status = "KHOP 100%" if diff < 1e-5 else ("TUONG DUONG" if diff < 0.01 else "LECH")
    comp_rows = [
        [model_display_name, f"{custom_te_acc:.4f}", f"{sk_te_acc:.4f}", f"{diff:+.4f}", status]
    ]
    print_table(
        headers=["Phuong phap", "Tu cai dat", "Scikit-Learn", "Chenh lech", "Trang thai"],
        rows=comp_rows,
        alignments=["<", ">", ">", ">", "^"],
    )

    # 7. Thong ke ma tran nham lan
    cm = confusion_matrix(yte, custom_model.predict(Xte), labels=classes)
    correct = int(np.trace(cm))
    total = int(np.sum(cm))
    pct = (correct / total) * 100
    log("EVAL", f"Ma tran nham lan (Tu cai dat): Dung {correct}/{total} mau test ({pct:.2f}%)")

    total_time = time.perf_counter() - total_start
    print("=" * 80)
    log("DONE", f"Hoan thanh danh gia trong {total_time:.2f} giay.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
