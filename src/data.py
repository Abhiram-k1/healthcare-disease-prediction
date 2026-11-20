"""Read real source records and preserve missing or invalid values for an audit."""
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from src.csv_validation import validate_csv_shape

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
ZERO_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def read_training(spark, path, header=False):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing at {path}. Run python scripts/download_dataset.py first.")
    fields = FEATURES + [TARGET]
    validate_csv_shape(path, expected_width=9, header=header, expected_header=fields if header else None)
    schema = StructType([StructField(name, StringType(), True) for name in fields + ["_corrupt_record"]])
    frame = (spark.read.schema(schema).option("header", header)
             .option("multiLine", True).option("escape", '"')
             .option("mode", "PERMISSIVE").option("columnNameOfCorruptRecord", "_corrupt_record")
             .csv(path.as_uri()).cache())
    frame.count()  # Materialize before querying Spark's corrupt-record column.
    if frame.filter(F.col("_corrupt_record").isNotNull()).limit(1).count():
        raise ValueError("Dataset contains malformed rows. Every record must contain exactly nine CSV fields.")
    frame = frame.drop("_corrupt_record")
    return frame.withColumn("RecordID", F.sha2(F.concat_ws("|", *[F.coalesce(F.col(c), F.lit("<null>")) for c in fields]), 256))


def read_patients(spark, path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Patient CSV missing at {path}. It must have a header and the eight documented features.")
    columns = validate_csv_shape(path, header=True)
    missing = set(FEATURES) - set(columns)
    if missing:
        raise ValueError(f"Patient CSV lacks columns: {', '.join(sorted(missing))}")
    frame = (spark.read.option("header", True).option("mode", "FAILFAST")
             .option("multiLine", True).option("escape", '"').csv(path.as_uri()))
    if "PatientID" not in frame.columns:
        frame = frame.withColumn("PatientID", F.sha2(F.concat_ws("|", *[F.coalesce(F.col(c), F.lit("<null>")) for c in FEATURES]), 256))
    return frame.select("PatientID", *FEATURES)


def inspect(frame, labeled=True):
    fields = FEATURES + ([TARGET] if labeled else [])
    expressions = [F.count(F.lit(1)).alias("rows")]
    for name in fields:
        value = F.trim(F.col(name)).cast("double")
        missing = F.col(name).isNull() | (F.trim(F.col(name)) == "")
        invalid = (~missing) & (value.isNull() | F.isnan(value) | (F.abs(value) == float("inf")))
        expressions.extend([
            F.sum(F.when(missing, 1).otherwise(0)).alias(name + "_missing"),
            F.sum(F.when(invalid, 1).otherwise(0)).alias(name + "_nonnumeric_or_nonfinite"),
            F.sum(F.when(value < 0, 1).otherwise(0)).alias(name + "_negative"),
        ])
        if name in ZERO_MISSING:
            expressions.append(F.sum(F.when(value == 0, 1).otherwise(0)).alias(name + "_zero_missing"))
        if name == "Age":
            expressions.append(F.sum(F.when((value < 21) | (value != F.floor(value)), 1).otherwise(0)).alias("Age_outside_cohort_or_fractional"))
        if name == "Pregnancies":
            expressions.append(F.sum(F.when(value != F.floor(value), 1).otherwise(0)).alias("Pregnancies_fractional"))
    result = frame.agg(*expressions).first().asDict()
    if labeled:
        result["invalid_target_rows"] = frame.filter(~F.col(TARGET).cast("double").isin(0.0, 1.0) | F.col(TARGET).cast("double").isNull()).count()
    return result


def clean_features(frame):
    """Apply the same deterministic policy before training and saved-model inference."""
    for name in FEATURES:
        value = F.trim(F.col(name)).cast("double")
        invalid = value.isNull() | F.isnan(value) | (F.abs(value) == float("inf")) | (value < 0)
        if name in ZERO_MISSING:
            invalid = invalid | (value == 0)
        if name == "Age":
            invalid = invalid | (value < 21) | (value != F.floor(value))
        if name == "Pregnancies":
            invalid = invalid | (value != F.floor(value))
        canonical = F.when(value == 0, F.lit(0.0)).otherwise(value)
        frame = frame.withColumn(name, F.when(invalid, F.lit(None).cast("double")).otherwise(canonical))
    return frame


def prepare_training(frame):
    fields = FEATURES + [TARGET]
    numeric = frame.withColumn(TARGET, F.col(TARGET).cast("double"))
    labeled = numeric.filter(F.col(TARGET).isin(0.0, 1.0))
    labeled = labeled.withColumn(TARGET, F.when(F.col(TARGET) == 0, F.lit(0.0)).otherwise(F.lit(1.0)))
    canonical = clean_features(labeled).dropDuplicates(fields)
    key = F.concat_ws("|", *[F.coalesce(F.col(name).cast("string"), F.lit("<null>")) for name in fields])
    return canonical.withColumn("RecordID", F.sha2(key, 256))


def split_records(frame, seed):
    """Canonical feature hashes keep equivalent measurements in one split."""
    feature_key = F.sha2(F.concat_ws("|", *[F.coalesce(F.col(name).cast("string"), F.lit("<null>")) for name in FEATURES]), 256)
    bucket = F.pmod(F.xxhash64(feature_key, F.lit(seed)), F.lit(10000))
    split = frame.withColumn("_split_bucket", bucket)
    train = split.filter(F.col("_split_bucket") < 6000).drop("_split_bucket").cache()
    validation = split.filter((F.col("_split_bucket") >= 6000) & (F.col("_split_bucket") < 8000)).drop("_split_bucket").cache()
    test = split.filter(F.col("_split_bucket") >= 8000).drop("_split_bucket").cache()
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        labels = {row[TARGET] for row in part.select(TARGET).distinct().collect()}
        if labels != {0.0, 1.0}:
            raise ValueError(f"The {name} split must contain both classes. Use more records or another --seed.")
    for feature in FEATURES:
        if train.filter(F.col(feature).isNotNull()).limit(1).count() == 0:
            raise ValueError(f"Training feature {feature} has no valid observations, so median imputation is impossible.")
    return train, validation, test
