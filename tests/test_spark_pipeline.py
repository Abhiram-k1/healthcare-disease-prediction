"""Validate cleaning, disjoint splits and model persistence on authentic records."""
from pathlib import Path

import pytest
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, functions as F

from src.data import FEATURES, TARGET, prepare_training, read_training, split_records
from src.models import build_pipelines


@pytest.mark.integration
def test_real_source_split_and_model_reload(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/raw/03_Dataset.csv"
    if not source.is_file():
        pytest.skip("Download the authentic Pima source to run this check")
    spark = SparkSession.builder.master("local[2]").appName("Assignment3-Integration-Test").config("spark.sql.shuffle.partitions", 2).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        raw = read_training(spark, source)
        assert raw.count() == 768
        first = raw.orderBy("RecordID").limit(1)
        equivalent = first.select(*[F.concat(F.lit(" "), F.col(name).cast("double").cast("string"), F.lit(" ")).alias(name) for name in FEATURES + [TARGET]], "RecordID")
        original_id = prepare_training(first).first()["RecordID"]
        equivalent_id = prepare_training(equivalent).first()["RecordID"]
        assert original_id == equivalent_id
        assert prepare_training(first.unionByName(equivalent)).count() == 1
        clean = prepare_training(raw).cache()
        assert clean.filter(F.col("Glucose") == 0).count() == 0
        assert clean.filter(F.col("Insulin").isNull()).count() > 0
        training, validation, test = split_records(clean, 42)
        train_ids = {row["RecordID"] for row in training.select("RecordID").collect()}
        validation_ids = {row["RecordID"] for row in validation.select("RecordID").collect()}
        test_ids = {row["RecordID"] for row in test.select("RecordID").collect()}
        assert not train_ids & validation_ids
        assert not train_ids & test_ids
        assert not validation_ids & test_ids
        assert len(train_ids | validation_ids | test_ids) == clean.count()
        train_features = {tuple(row[name] for name in FEATURES) for row in training.select(*FEATURES).collect()}
        validation_features = {tuple(row[name] for name in FEATURES) for row in validation.select(*FEATURES).collect()}
        test_features = {tuple(row[name] for name in FEATURES) for row in test.select(*FEATURES).collect()}
        assert not train_features & validation_features
        assert not train_features & test_features
        assert not validation_features & test_features
        model = build_pipelines(42)["logistic_regression"].fit(training)
        output = (tmp_path / "model").resolve().as_uri()
        model.write().overwrite().save(output)
        reloaded = PipelineModel.load(output)
        predictions = model.transform(test).select("RecordID", "prediction").orderBy("RecordID").collect()
        restored = reloaded.transform(test.drop(TARGET)).select("RecordID", "prediction").orderBy("RecordID").collect()
        assert predictions == restored
        assert all(row["prediction"] in (0.0, 1.0) for row in restored)
    finally:
        spark.stop()
