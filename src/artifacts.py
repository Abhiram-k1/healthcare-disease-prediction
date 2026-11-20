"""Save small aggregate reports and figures. Spark writes full tabular results."""
import csv
import hashlib
import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def dataset_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def save_metrics(output, metrics):
    path = Path(output) / "results/performance_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"model": name, "split": split, **values} for split, models in metrics.items() for name, values in models.items()]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figures(output, metrics, class_counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    names = list(metrics["test"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for axis, measure in zip(axes, ["accuracy", "precision", "recall", "f1"]):
        axis.bar(names, [metrics["test"][name][measure] for name in names], color=["#3f6c97", "#648b65", "#a37044"])
        axis.set_title("Held-out test " + measure)
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=30)
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    axis.bar(["Outcome 0", "Outcome 1"], [class_counts.get("0", 0), class_counts.get("1", 0)], color=["#3f6c97", "#a37044"])
    axis.set_ylabel("Cleaned records")
    axis.set_title("Observed class distribution")
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)


def save_interpretation(output, best_name, metrics, counts):
    best_validation = metrics["validation"][best_name]
    best_test = metrics["test"][best_name]
    report = f"""# Results interpretation\n\nThis report is generated from the actual run.\n\nThe selected model is **{best_name}**, chosen by validation class-1 F1 ({best_validation['f1']:.4f}) with accuracy as the tie breaker.\n\nThe separate test split contains {counts['test']} records. Its selected-model accuracy is {best_test['accuracy']:.4f}, positive-class precision is {best_test['precision']:.4f}, recall is {best_test['recall']:.4f}, and F1 is {best_test['f1']:.4f}. False positives total {best_test['false_positive']}, and false negatives total {best_test['false_negative']}.\n\nPrecision measures how often predicted positives match observed positives. Recall measures how many observed positives the model finds. F1 combines precision and recall. Accuracy also counts correct negatives and should be read alongside the class distribution.\n\nAll three models use the same disjoint splits. The test metrics were computed after validation selected the model. Hyperparameters are fixed in src/models.py, so this run is a baseline comparison. The model is saved from its training-only fit.\n\nThe demonstration patient file contains real held-out records with their labels removed. It shows inference on unseen records, while the observed labels remain in the saved test manifest for evaluation.\n\nThe source cohort contains female Pima Indian patients aged at least 21. The analysis has that population scope, a small sample, and historical measurements. These results support the coursework demonstration.\n"""
    path = Path(output) / "reports/results_interpretation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
