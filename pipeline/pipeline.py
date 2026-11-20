"""End-to-end Spark training and persisted-model prediction pipelines."""
import json
import time
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F

from src.artifacts import dataset_digest, save_figures, save_interpretation, save_metrics, write_json
from src.data import FEATURES, TARGET, clean_features, inspect, prepare_training, read_patients, read_training, split_records
from src.evaluation import evaluate, select_best
from src.models import build_pipelines


def create_spark(args):
    output = args.output.resolve()
    events = output / "spark-events"
    events.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder.appName("Assignment3-Diabetes-Spark-MLlib")
             .master(args.master)
             .config("spark.sql.shuffle.partitions", args.partitions)
             .config("spark.eventLog.enabled", "true")
             .config("spark.eventLog.dir", events.as_uri())
             .config("spark.ui.enabled", "true")
             .config("spark.sql.warehouse.dir", (output / "spark-warehouse").as_uri())
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark UI: {spark.sparkContext.uiWebUrl}")
    return spark


def save_frame(frame, path):
    frame.write.mode("overwrite").option("header", True).csv(Path(path).resolve().as_uri())


def prediction_columns(predictions, patient_id="PatientID"):
    return predictions.select(
        patient_id, *FEATURES,
        F.col("prediction").cast("int").alias("PredictedOutcome"),
        vector_to_array("probability")[1].alias("DiabetesProbability"),
    )


def train(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spark = create_spark(args)
    try:
        raw = read_training(spark, args.input, args.header)
        raw.printSchema()
        raw.show(10, truncate=False)
        audit = inspect(raw)
        clean = prepare_training(raw).repartition(args.partitions).cache()
        audit["cleaned_rows"] = clean.count()
        audit["exact_raw_duplicate_rows"] = raw.count() - raw.dropDuplicates(FEATURES + [TARGET]).count()
        audit["canonical_duplicate_rows"] = raw.count() - audit["invalid_target_rows"] - audit["cleaned_rows"]
        audit["policy"] = "Remove invalid targets. Canonicalize numeric features, convert invalid measurements to null, and remove canonical feature/label duplicates. Keep equal feature records in one split. Learn medians from training only."
        audit["nulls_after_cleaning"] = clean.agg(*[F.sum(F.when(F.col(name).isNull(), 1).otherwise(0)).alias(name) for name in FEATURES]).first().asDict()
        write_json(output / "reports/data_quality.json", audit)
        save_frame(clean, output / "processed/cleaned_records")
        training, validation, test = split_records(clean, args.seed)
        counts = {name: frame.count() for name, frame in [("train", training), ("validation", validation), ("test", test)]}
        class_counts = {str(int(row[TARGET])): row["count"] for row in clean.groupBy(TARGET).count().collect()}
        save_frame(training.select("RecordID", TARGET), output / "results/splits/train")
        save_frame(validation.select("RecordID", TARGET), output / "results/splits/validation")
        save_frame(test.select("RecordID", TARGET), output / "results/splits/test")
        metrics = {"validation": {}, "test": {}}
        models = {}
        timings = {}
        for name, estimator in build_pipelines(args.seed).items():
            started = time.perf_counter()
            model = estimator.fit(training)
            timings[name] = {"fit_seconds": time.perf_counter() - started}
            models[name] = model
            metrics["validation"][name] = evaluate(model.transform(validation))
        best_name = select_best(metrics["validation"])
        # Evaluate the independent test set after selecting the model on validation.
        for name, model in models.items():
            predicted = model.transform(test)
            metrics["test"][name] = evaluate(predicted)
            save_frame(predicted.select("RecordID", TARGET, F.col("prediction").cast("int").alias("PredictedOutcome"), vector_to_array("probability")[1].alias("DiabetesProbability")), output / f"results/test_predictions/{name}")
            plan = predicted._jdf.queryExecution().toString()
            plans = output / "reports/execution_plans"
            plans.mkdir(parents=True, exist_ok=True)
            (plans / f"{name}.txt").write_text(plan, encoding="utf-8")
        best = models[best_name]
        model_path = output / "models/best_model"
        best.write().overwrite().save(model_path.as_uri())
        metadata = {
            "best_model": best_name,
            "selection_metric": "validation positive-class F1, then accuracy, then model name",
            "features": FEATURES,
            "target": TARGET,
            "positive_class": 1,
            "seed": args.seed,
            "split_policy": "Canonical feature-only content-hash approximate 60/20/20 train/validation/test",
            "split_counts": counts,
            "class_counts": class_counts,
            "dataset_sha256": dataset_digest(args.input),
            "dataset_bytes": Path(args.input).stat().st_size,
            "spark_version": spark.version,
            "master": spark.sparkContext.master,
            "partitions": args.partitions,
            "timings": timings,
            "preprocessing_contract": "src.data.clean_features followed by saved training-fitted Imputer, VectorAssembler and optional StandardScaler",
        }
        write_json(output / "models/model_metadata.json", metadata)
        write_json(output / "results/performance_comparison.json", metrics)
        save_metrics(output, metrics)
        save_figures(output, metrics, class_counts)
        save_interpretation(output, best_name, metrics, counts)
        # Authentic held-out features provide an unseen-patient demonstration.
        demo = test.orderBy("RecordID").limit(args.demo_patients).select(F.col("RecordID").alias("PatientID"), *FEATURES)
        demo_rows = demo.collect()
        import csv
        patient_path = output / "results/demo_patients.csv"
        with patient_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["PatientID"] + FEATURES)
            writer.writeheader()
            writer.writerows([row.asDict() for row in demo_rows])
        loaded = PipelineModel.load(model_path.as_uri())
        demo_prepared = clean_features(read_patients(spark, patient_path))
        demo_predictions = prediction_columns(loaded.transform(demo_prepared)).orderBy("PatientID")
        demo_predictions.show(args.demo_patients, truncate=False)
        save_frame(demo_predictions, output / "results/new_patient_predictions")
        write_json(output / "reports/run_summary.json", {**metadata, "metrics": metrics, "ui_url": spark.sparkContext.uiWebUrl})
        print(json.dumps({"best_model": best_name, "split_counts": counts, "output": str(output)}, indent=2))
        if args.ui_hold_seconds:
            print(f"Keeping Spark UI available for {args.ui_hold_seconds} seconds. Capture Jobs, Stages and SQL pages now.")
            time.sleep(args.ui_hold_seconds)
    finally:
        spark.stop()


def predict(args):
    model_path = args.model.resolve()
    if not (model_path / "metadata").is_dir():
        raise FileNotFoundError(f"Saved Spark PipelineModel missing at {model_path}. Run python main.py train first.")
    spark = create_spark(args)
    try:
        patients = read_patients(spark, args.input)
        audit = inspect(patients, labeled=False)
        if audit["rows"] == 0:
            raise ValueError("Patient CSV contains no records")
        prepared = clean_features(patients)
        model = PipelineModel.load(model_path.as_uri())
        result = prediction_columns(model.transform(prepared)).orderBy("PatientID")
        save_frame(result, args.output / "results/new_patient_predictions")
        write_json(args.output / "reports/patient_input_quality.json", audit)
        result.show(20, truncate=False)
        print(f"Predictions saved to {(args.output / 'results/new_patient_predictions').resolve()}")
    finally:
        spark.stop()
