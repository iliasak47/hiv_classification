from pyspark.sql.functions import col, lit, when
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, udf
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import mlflow
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F

def create_spark_session():
    return SparkSession.builder \
        .appName("HIV_Classification_Training") \
        .getOrCreate()

def load_parquet_dataset(spark, input_path):
    return spark.read.parquet(input_path)

def split_dataset(df, test_ratio=0.2, seed=42):
    return df.randomSplit([1 - test_ratio, test_ratio], seed=seed)


def train_with_cross_validation(train_df):
    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        seed=42
    )

    # Large param grid
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [100, 200, 300]) \
        .addGrid(rf.maxDepth, [4, 6, 8]) \
        .addGrid(rf.minInstancesPerNode, [1, 2]) \
        .addGrid(rf.subsamplingRate, [0.7, 1.0]) \
        .build()

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4
    )

    print("🔁 Starting cross-validation...")
    cv_model = cv.fit(train_df)
    print("✅ Best model selected.")

    # Get best model hyperparams
    best_model = cv_model.bestModel
    best_params = {
        "numTrees": best_model.getNumTrees,
        "maxDepth": best_model.getOrDefault("maxDepth"),
        "minInstancesPerNode": best_model.getOrDefault("minInstancesPerNode"),
        "subsamplingRate": best_model.getOrDefault("subsamplingRate")
    }

    # Log params to MLflow
    for k, v in best_params.items():
        mlflow.log_param(k, v)

    return cv_model


def evaluate_model(model, test_df, threshold=0.38):
    # Prédictions
    predictions = model.transform(test_df)

    # Extraire P(classe=1) sans UDF
    probs = vector_to_array(col("probability"))[1]
    predictions = predictions.withColumn("custom_prediction",
                                         when(probs >= lit(threshold), lit(1.0)).otherwise(lit(0.0)))

    # Évaluateurs (custom threshold)
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="custom_prediction", metricName="accuracy")
    eval_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="custom_prediction", metricName="f1")
    eval_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    acc = eval_acc.evaluate(predictions)
    f1  = eval_f1.evaluate(predictions)
    auc = eval_auc.evaluate(predictions)

    print(f"✅ Accuracy (threshold={threshold}): {acc:.4f}")
    print(f"✅ F1-score (threshold={threshold}): {f1:.4f}")
    print(f"✅ ROC AUC: {auc:.4f}")

    # Log MLflow
    mlflow.log_param("decision_threshold", threshold)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("auc", auc)

    # Sauvegarde du meilleur modèle
    mlflow.spark.log_model(model.bestModel, "random_forest_model")

    return predictions

if __name__ == "__main__":
    spark = create_spark_session()
    df = load_parquet_dataset(spark, "data/processed/hiv_prepared.parquet")
    train_df, test_df = split_dataset(df, test_ratio=0.2)

    mlflow.set_experiment("HIV RandomForest Experiment")

    with mlflow.start_run():
        model = train_with_cross_validation(train_df)
        predictions = evaluate_model(model, test_df)

    spark.stop()
