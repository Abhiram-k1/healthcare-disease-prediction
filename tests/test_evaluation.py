"""Check positive-class formulas and validation-only model selection."""
import unittest

from src.evaluation import metrics_from_counts, select_best


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_positive_class(self):
        metrics = metrics_from_counts(tp=3, tn=4, fp=1, fn=2)
        self.assertAlmostEqual(metrics["accuracy"], 0.7)
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 0.6)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)

    def test_zero_positive_predictions_have_defined_metrics(self):
        metrics = metrics_from_counts(tp=0, tn=4, fp=0, fn=2)
        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1"], 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            metrics_from_counts(0, 0, 0, 0)

    def test_selection_uses_f1_then_accuracy_then_name(self):
        self.assertEqual(select_best({"tree": {"f1": 0.7, "accuracy": 0.9}, "forest": {"f1": 0.8, "accuracy": 0.8}}), "forest")
        self.assertEqual(select_best({"tree": {"f1": 0.8, "accuracy": 0.8}, "forest": {"f1": 0.8, "accuracy": 0.9}}), "forest")
        self.assertEqual(select_best({"tree": {"f1": 0.8, "accuracy": 0.9}, "forest": {"f1": 0.8, "accuracy": 0.9}}), "forest")


if __name__ == "__main__":
    unittest.main()
