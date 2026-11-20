"""Evaluate class 1 explicitly instead of relying on weighted averages."""


def metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    if total == 0:
        raise ValueError("Cannot evaluate an empty prediction set")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "rows": total,
    }


def evaluate(predictions):
    from pyspark.sql import functions as F
    from src.data import TARGET

    counts = predictions.agg(
        F.sum(F.when((F.col(TARGET) == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col(TARGET) == 0) & (F.col("prediction") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col(TARGET) == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col(TARGET) == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn"),
    ).first()
    return metrics_from_counts(*[int(counts[name] or 0) for name in ("tp", "tn", "fp", "fn")])


def select_best(validation_metrics):
    """Maximize validation F1, then accuracy, then stable model name."""
    return sorted(validation_metrics, key=lambda name: (-validation_metrics[name]["f1"], -validation_metrics[name]["accuracy"], name))[0]
