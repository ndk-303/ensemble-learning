import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated", category=FutureWarning)
import argparse

# Dam bao tuong thich ma hoa console tren Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.metrics import accuracy_score

from src.methods import Bagging, AdaBoost, Stacking
from src.experiments import MODEL_ORDER, evaluate_split, evaluate_references, summarize_scores
from src.data import load_data, build_model, SUPPORTED_MODELS
from src.utils import (
    log_info,
    log_train,
    log_done,
    print_table,
    save_single_confusion,
    save_visualizations,
    save_stability_plot,
)


def parse_args():
    """Khoi tao argparse va kiem tra tinh hop le."""
    parser = argparse.ArgumentParser(
        description="Ensemble Learning (Bagging, AdaBoost SAMME, Stacking) on Digits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 1. --method (Required)
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["bagging", "boosting", "stacking", "all"],
        help="Ensemble method: bagging | boosting | stacking | all",
    )

    # 2. --base_model
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model: dt, knn, svm, lr (default: dt for bagging/boosting; dt,knn,lr for stacking/all)",
    )

    # 3. --seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # 4. --test_size
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Test size ratio (0.0 < float < 1.0)",
    )

    # 5. --n_estimators
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="Number of estimators (bagging/boosting)",
    )

    # 6. --learning_rate
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (boosting)",
    )
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of repeated splits for --method all")
    parser.add_argument("--no_plots", action="store_true", help="Do not save plots to disk")
    parser.add_argument("--fig_dir", default="figs", help="Directory for plots")

    args = parser.parse_args()

    # --- VALIDATION ---
    if args.seed < 0:
        parser.error("--seed must be >= 0.")

    if not (0.0 < args.test_size < 1.0):
        parser.error("--test_size must be between 0.0 and 1.0.")

    if args.n_estimators is not None and args.n_estimators < 1:
        parser.error("--n_estimators must be >= 1.")

    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning_rate must be > 0.")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1.")

    # --- METHOD-SPECIFIC RULES ---
    method = args.method.lower()

    if method == "all":
        if args.base_model is None:
            args.parsed_models = ["dt", "knn", "lr"]
        else:
            args.parsed_models = [m.strip().lower() for m in args.base_model.split(",") if m.strip()]
            if not args.parsed_models or any(m not in SUPPORTED_MODELS for m in args.parsed_models):
                parser.error("--base_model for 'all' must be chosen from: dt, knn, svm, lr.")

    elif method == "stacking":
        if args.n_estimators is not None:
            parser.error("--n_estimators is not applicable to stacking.")
        if args.learning_rate is not None:
            parser.error("--learning_rate is not applicable to stacking.")

        if args.base_model is None:
            args.base_model = "dt,knn,lr"

        model_tokens = [m.strip().lower() for m in args.base_model.split(",") if m.strip()]
        if not model_tokens:
            parser.error("Stacking requires at least 1 valid base model.")
        for m in model_tokens:
            if m not in SUPPORTED_MODELS:
                parser.error(f"Base model '{m}' is not supported. Supported: {', '.join(SUPPORTED_MODELS)}.")
        args.parsed_models = model_tokens

    elif method == "bagging":
        if args.learning_rate is not None:
            parser.error("--learning_rate is not applicable to bagging.")

        if args.n_estimators is None:
            args.n_estimators = 50

        if args.base_model is None:
            args.base_model = "dt"
        else:
            args.base_model = args.base_model.strip().lower()
            if "," in args.base_model:
                parser.error("Bagging only accepts a single base model, e.g. --base_model dt")
            if args.base_model not in SUPPORTED_MODELS:
                parser.error(f"Base model '{args.base_model}' is not supported. Supported: {', '.join(SUPPORTED_MODELS)}.")

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
                parser.error("Boosting (AdaBoost) only supports base model 'dt' (Decision Stump).")

    return args


def get_base_model_label(code):
    """Map model code to clean display name."""
    mapping = {
        "dt": "Decision Tree",
        "knn": "KNN",
        "svm": "SVM",
        "lr": "Logistic Regression",
    }
    return mapping.get(code.lower(), code.upper())


def train_single_method(args):
    """Huan luyen 1 phuong phap Ensemble don le theo dung 5 phan tieu chuan."""
    method = args.method.lower()

    # ==================================================
    # 1. CONFIGURATION
    # ==================================================
    print("=" * 60)
    print(f" ENSEMBLE TRAINING: {method.upper()}")
    print("=" * 60)
    print(f" Method       : {method.title()}")

    if method == "bagging":
        print(f" Base Model   : {get_base_model_label(args.base_model)}")
        print(f" N Estimators : {args.n_estimators}")
    elif method == "boosting":
        print(f" Base Model   : Decision Tree (stump)")
        print(f" N Estimators : {args.n_estimators}")
        print(f" Learning Rate: {args.learning_rate}")
    elif method == "stacking":
        base_labels = ", ".join([get_base_model_label(m) for m in args.parsed_models])
        print(f" Base Models  : {base_labels}")
        print(f" Meta-Model   : Logistic Regression")
        print(f" CV           : 5-Fold OOF")

    print(f" Seed         : {args.seed}")
    print(f" Test Size    : {args.test_size}")
    print("=" * 60)

    # ==================================================
    # 2. DATASET
    # ==================================================
    Xtr, ytr, Xte, yte, _ = load_data(test_size=args.test_size, seed=args.seed)
    log_info(f"Dataset: {len(ytr)+len(yte)} samples, {Xtr.shape[1]} features (Train: {len(ytr)} | Test: {len(yte)})")

    # ==================================================
    # 3. TRAINING
    # ==================================================
    if method == "bagging":
        log_info(f"Initializing Bagging ({args.n_estimators} estimators)...")
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
        log_info(f"Initializing AdaBoost ({args.n_estimators} estimators, lr={args.learning_rate})...")
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
        log_info("Initializing Stacking...")
        log_train("Training started...")

        def stacking_callback(phase):
            if phase == "phase_1":
                log_train("[Phase 1/3] 5-Fold Out-Of-Fold predictions...")
            elif phase == "phase_2":
                log_train("[Phase 2/3] Fitting Meta-Learner (Logistic Regression)...")
            elif phase == "phase_3":
                log_train("[Phase 3/3] Refitting base estimators...")

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

    print("=" * 60)
    print(f" RESULT | seed={args.seed}")
    print("=" * 60)
    print_table(
        ["Model", "Train acc", "Test acc", "Fit time"],
        [[method.title(), f"{accuracy_score(ytr, model.predict(Xtr)):.4f}",
          f"{acc:.4f}", f"{train_time:.2f}s"]],
    )

    # ==================================================
    # 5. ARTIFACTS
    # ==================================================
    print()
    if not args.no_plots:
        cf_path = save_single_confusion(
            yte=yte,
            y_pred=y_pred,
            title=f"Confusion Matrix ({method.title()})",
            path=os.path.join(args.fig_dir, "confusion.png"),
        )
        log_done(f"Saved: {cf_path}")
    print("=" * 60)


def run_all(args):
    """Compare baselines and ensembles on identical stratified splits."""
    bag_t = args.n_estimators or 50
    ada_t = args.n_estimators or 100
    ada_lr = args.learning_rate or 1.0

    print("\n" + "=" * 60)
    print(" ENSEMBLE LEARNING | COMPARISON")
    print("=" * 60)
    print(f" Dataset       : Digits ({args.test_size*100:.0f}% test split)")
    print(f" Repeats       : {args.repeats} split(s) | seeds {args.seed}..{args.seed + args.repeats - 1}")
    print(f" Bagging       : {bag_t} trees")
    print(f" AdaBoost      : {ada_t} stumps (lr={ada_lr})")
    print(f" Stacking      : {','.join(args.parsed_models)} -> LR (5-fold)")
    print("=" * 60)

    runs = []
    first = None
    for i in range(args.repeats):
        split_seed = args.seed + i
        print(f"\n[RUN {i + 1:>2}/{args.repeats}] seed={split_seed}...")
        start = time.perf_counter()
        result = evaluate_split(split_seed, args.test_size, bag_t, ada_t, ada_lr, args.parsed_models)
        runs.append(result["scores"])
        if first is None:
            first = result
        best_name = max(MODEL_ORDER, key=lambda name: result["scores"][name])
        print(f"[DONE] {time.perf_counter() - start:.1f}s | best: {best_name} ({result['scores'][best_name]:.4f})")

    print("\n" + "=" * 60)
    print(f" FIRST SPLIT | seed={args.seed}")
    print("=" * 60)
    print_table(
        ["Model", "Train acc", "Test acc", "Fit time"],
        [[name, f"{first['train_scores'][name]:.4f}",
          f"{first['scores'][name]:.4f}", f"{first['times'][name]:.2f}s"]
         for name in MODEL_ORDER],
    )

    reference_scores = evaluate_references(first, bag_t, ada_t, ada_lr, args.parsed_models)
    print("\n" + "=" * 60)
    print(f" SCIKIT-LEARN REFERENCE | first split only, seed={args.seed}")
    print("=" * 60)
    print_table(
        ["Ensemble", "Custom acc", "Library acc", "Difference"],
        [[name, f"{first['scores'][name]:.4f}", f"{reference_scores[name]:.4f}",
          f"{first['scores'][name] - reference_scores[name]:+.4f}"]
         for name in ("Bagging", "AdaBoost", "Stacking")],
    )

    summary = summarize_scores(runs)
    print("\n" + "=" * 60)
    split_label = "SPLIT" if args.repeats == 1 else "SPLITS"
    print(f" SUMMARY | {args.repeats} {split_label}")
    print("=" * 60)
    print_table(
        ["Model", "Test acc", "Baseline", "Paired gain"],
        [[name, f"{summary[name]['mean']:.4f} +/- {summary[name]['std']:.4f}",
          summary[name]["baseline"] or "-",
          f"{summary[name]['paired_delta']:+.4f}" if summary[name]["paired_delta"] is not None else "-"]
         for name in MODEL_ORDER],
    )

    if not args.no_plots:
        Xtr, ytr, Xte, yte = first["data"]
        models = first["models"]
        plot_scores = {name: (first["train_scores"][name], first["scores"][name])
                       for name in MODEL_ORDER}
        saved = save_visualizations(
            res=plot_scores, ytr=ytr, yte=yte, Xtr=Xtr, Xte=Xte,
            ada=models["AdaBoost"], bag=models["Bagging"], stk=models["Stacking"],
            adaboost_t=ada_t, bagging_t=bag_t, stk_name="Stacking", fig_dir=args.fig_dir,
        )
        if args.repeats > 1:
            saved.append(save_stability_plot(
                list(MODEL_ORDER), [summary[name]["mean"] for name in MODEL_ORDER],
                [summary[name]["std"] for name in MODEL_ORDER], args.repeats, args.fig_dir,
            ))
        for path in saved:
            print(f"[SAVED] {path}")


def main():
    args = parse_args()
    if args.method.lower() == "all":
        run_all(args)
    else:
        train_single_method(args)


if __name__ == "__main__":
    main()
