"""Build the three required Spark ML pipelines."""
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler

from src.data import FEATURES, TARGET


def build_pipelines(seed):
    classifiers = {
        "logistic_regression": LogisticRegression(featuresCol="scaled_features", labelCol=TARGET, maxIter=100, regParam=0.1),
        "decision_tree": DecisionTreeClassifier(featuresCol="features", labelCol=TARGET, maxDepth=5, seed=seed),
        "random_forest": RandomForestClassifier(featuresCol="features", labelCol=TARGET, numTrees=50, maxDepth=5, seed=seed),
    }
    pipelines = {}
    imputed = [name + "_imputed" for name in FEATURES]
    for name, classifier in classifiers.items():
        stages = [
            Imputer(strategy="median", inputCols=FEATURES, outputCols=imputed, relativeError=0.0),
            VectorAssembler(inputCols=imputed, outputCol="features", handleInvalid="error"),
        ]
        if name == "logistic_regression":
            stages.append(StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True))
        stages.append(classifier)
        pipelines[name] = Pipeline(stages=stages)
    return pipelines
