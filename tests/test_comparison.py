import unittest
import subprocess
import sys
from pathlib import Path

from src.experiments import build_comparison_models, build_reference_models, summarize_scores


class ComparisonTests(unittest.TestCase):
    def test_reference_models_match_ensemble_hyperparameters(self):
        refs = build_reference_models(42, 7, 9, 0.5, ("dt", "knn", "lr"))
        self.assertEqual(list(refs), ["Bagging", "AdaBoost", "Stacking"])
        self.assertEqual(refs["Bagging"].n_estimators, 7)
        self.assertEqual(refs["AdaBoost"].n_estimators, 9)
        self.assertEqual(refs["AdaBoost"].learning_rate, 0.5)
        self.assertEqual(len(refs["Stacking"].estimators), 3)

    def test_single_method_uses_same_result_table(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "train.py", "--method", "bagging", "--n_estimators", "2", "--no_plots"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT | seed=42", result.stdout)
        self.assertIn("Train acc", result.stdout)
        self.assertIn("Test acc", result.stdout)

    def test_all_cli_reports_shared_splits_and_paired_gains(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "train.py", "--method", "all", "--n_estimators", "2",
             "--repeats", "2", "--no_plots"],
            cwd=root, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("seed=42", result.stdout)
        self.assertIn("seed=43", result.stdout)
        self.assertIn("Paired gain", result.stdout)
        self.assertIn("SCIKIT-LEARN REFERENCE", result.stdout)
        self.assertIn("Bagging", result.stdout)
        self.assertIn("Stacking", result.stdout)

    def test_comparison_has_matching_single_model_baselines(self):
        models = build_comparison_models(seed=42, bag_t=3, ada_t=3, learning_rate=1.0)
        self.assertEqual(
            list(models),
            ["Stump", "Tree", "KNN", "LR", "SVM", "Bagging", "AdaBoost", "Stacking"],
        )
        self.assertEqual(models["Stump"].max_depth, 1)
        self.assertEqual(models["Tree"].max_depth, 12)
        self.assertEqual(models["Bagging"].base_estimator.max_depth, 12)
        self.assertEqual(models["AdaBoost"].base_estimator.max_depth, 1)

    def test_summary_uses_paired_differences_on_same_splits(self):
        runs = [
            {"Tree": 0.8, "Bagging": 0.9, "Stump": 0.2, "AdaBoost": 0.7,
             "KNN": 0.9, "LR": 0.91, "SVM": 0.92, "Stacking": 0.93},
            {"Tree": 0.7, "Bagging": 0.8, "Stump": 0.3, "AdaBoost": 0.6,
             "KNN": 0.8, "LR": 0.81, "SVM": 0.82, "Stacking": 0.83},
        ]
        summary = summarize_scores(runs)
        self.assertAlmostEqual(summary["Bagging"]["mean"], 0.85)
        self.assertAlmostEqual(summary["Bagging"]["std"], 0.05)
        self.assertAlmostEqual(summary["Bagging"]["paired_delta"], 0.1)
        self.assertEqual(summary["Bagging"]["baseline"], "Tree")
        self.assertAlmostEqual(summary["AdaBoost"]["paired_delta"], 0.4)
        self.assertAlmostEqual(summary["Stacking"]["paired_delta"], 0.01)
        self.assertEqual(summary["Stacking"]["baseline"], "SVM")


if __name__ == "__main__":
    unittest.main()
