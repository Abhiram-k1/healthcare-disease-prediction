# Project explanation and distributed Spark ML

## How the code connects

The root CLI parses input, output, master, partition and seed settings. The training entry point creates Spark, loads a DataFrame, audits and cleans it, makes disjoint splits, builds the three Spark pipelines, fits each on training data, compares validation predictions, evaluates test predictions, saves the best model, and produces reports and patient predictions.

The input features are the eight numerical medical attributes in `src/data.py`. `Outcome` is the observed binary label. `RecordID`, raw text and split membership are excluded from the feature vector. The selected saved model includes the fitted medians, vector assembly and any scaling stage. The CLI always applies the documented deterministic cleaning function before inference.

## Spark execution and scale

DataFrame transformations build a logical execution plan. Actions such as `count`, statistical fitting, writing and metric aggregation execute that plan. Spark partitions the data and schedules tasks for the partitions. Shuffles occur when grouping, repartitioning or aggregating requires records to move between workers.

Median imputation and StandardScaler need global feature statistics. Spark computes these from partitioned training records and combines summaries. Logistic Regression iteratively aggregates gradient information to update model parameters. Decision Tree and Random Forest use distributed feature statistics to choose split thresholds. Spark transforms partitions independently when generating predictions.

The configured `spark.sql.shuffle.partitions` controls the default number of SQL shuffle partitions. The cleaned DataFrame is explicitly repartitioned to the requested count and cached. Spark's local master uses threads on one machine. With a compatible Spark cluster master, its scheduler uses executors across worker machines and the same pipeline code can operate on distributed data. Inputs and outputs must use storage paths visible to every participating process.

## Observe execution

Run training with `--ui-hold-seconds 180` and open the printed UI URL before it exits. Jobs shows actions, Stages shows task scheduling and shuffle boundaries, Executors shows worker resource information, and SQL shows DataFrame query plans. Text plans are saved for the test prediction transformations, and event logs retain the application's execution events.

To inspect completed runs, configure a Spark History Server's `spark.history.fs.logDirectory` to this project's `outputs/spark-events` directory. Start the server using the installed Spark distribution's history-server script. The history server commonly uses port 18080. This optional inspection infrastructure is separate from the default pipeline.

## How to explain the results

Read `performance_comparison.csv` alongside the generated interpretation and class-distribution chart. Accuracy describes all correct classifications. Precision describes the reliability of positive predictions, recall measures observed positives recovered, and F1 combines precision and recall. A model with higher accuracy can still miss more positives, so the selection rule uses validation F1.

The independent test split measures performance after model selection. Fixed parameters and a recorded seed make this baseline traceable. Different data, Spark versions or configurations may change the measured results. Generated predictions are model outputs for the historical source target and have the population scope stated in the data dictionary.
