# Methodology

## Load and inspect

An initial streaming standard-library CSV check rejects short or extra records and malformed quoting, while preserving empty field values for imputation. Spark then loads the authentic headerless Pima CSV with nine string fields, preserving malformed measurements for the audit. The parser accepts quoted multiline values and doubled quote escaping consistently with the preflight checker. The schema and first rows are displayed. Aggregate audits count missing, nonnumeric, nonfinite, negative and zero-missing measurements, invalid targets and age or pregnancy domain violations.

After cleaning, `RecordID` is a SHA-256 hash of canonical numeric feature values and the target. It identifies records for repeatable tracking and never enters the model feature vector. Numeric forms such as `1`, `1.0` and surrounding whitespace resolve to the same value. Duplicate canonical feature/target rows are removed. The audit separately counts exact raw duplicates and canonical duplicates. This policy treats equal cleaned measurements with equal observed labels as duplicate observations.

## Clean measurements

Targets must be numeric 0 or 1. An invalid target is removed because its observed class cannot be inferred. Missing or invalid predictor values become null. The source describes a female cohort aged at least 21, so lower or fractional ages become null. Pregnancy counts must be nonnegative integers. Other negative or nonfinite measurements become null. Five medical measurements use zero as a missing placeholder. Pregnancy zero remains valid.

No data-derived thresholds, means, medians or standard deviations are computed before the split for modeling. The preprocessing function applies the same fixed domain rules to training and inference input.

## Repeatable disjoint splits

The pipeline creates a SHA-256 key from canonical feature values, with explicit null placeholders and the target excluded. It computes `pmod(xxhash64(feature_key, seed), 10000)`. Buckets below 6000 form training, 6000 through 7999 form validation, and 8000 through 9999 form testing. The ratios are approximate because feature groups are hashed individually. Equal cleaned feature records share membership even if labels conflict. Actual split sizes are saved.

Each split must contain both target classes. Training must contain at least one valid value for every feature. Otherwise an informative error requests a larger dataset or another seed. The validation set chooses the model, leaving the test set independent of that choice.

## Spark ML pipelines

Each classifier gets a separate fitted Spark pipeline. The `Imputer` learns exact feature medians using only training data and writes eight imputed columns. `VectorAssembler` converts those columns into the required Spark ML `Vector`.

Logistic Regression receives a centered, standardized vector from `StandardScaler` fitted on training data. Its fixed baseline parameters are `maxIter=100` and `regParam=0.1`. The Decision Tree uses `maxDepth=5`. Random Forest uses 50 trees with `maxDepth=5`. The tree algorithms operate on the assembled vector because threshold ordering is unchanged by scaling. Randomness uses the configured seed.

The classifier and statistical transformers are fitted through Spark `Pipeline.fit`. Their JVM-backed estimators use Spark's partitioned execution. No Pandas training or scikit-learn classifier substitutes are used.

## Evaluation

Predicted labels are compared with observed labels using distributed Spark aggregates. For positive class 1, TP counts true positives, TN true negatives, FP false positives and FN false negatives.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

A metric with a zero positive-class denominator is assigned 0 and its confusion counts remain available. Empty evaluation sets raise an error. Every metric is computed for every model on both validation and test sets.

The selection rule maximizes validation class-1 F1. Ties use validation accuracy, then model name for deterministic ordering. Models retain their original training-only fit when tested and saved. Test scores are available for the requested comparison, while the selection rule remains validation-based.

## Unseen-patient inference

Training exports up to ten actual held-out feature records, ordered by their record identifiers, with their labels removed. It saves the chosen model, loads it again, applies the shared feature-cleaning function and predicts those unseen records. The test membership manifest retains the original labels for evaluation.

The independent `predict` command uses the same cleaning contract followed by the persisted imputation, assembly, scaling and classifier stages. No fitting occurs during prediction. Its output includes the predicted binary outcome and the classifier probability for class 1.

## Output and interpretation

Large tabular results use Spark's distributed CSV writer. Small aggregate comparison tables and reports are written by the driver. Matplotlib renders charts from the actual aggregate metrics and class counts. The interpretation report states the selected model, measured scores and error counts.

The dataset's small size and historical cohort limit generalization. A local run demonstrates Spark ML APIs and reproducibility. Claims about cluster speed or clinical performance require separate experiments and evidence.
