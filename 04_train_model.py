import argparse
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to churn_fe.parquet")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("ChurnModel").master("local[*]").getOrCreate()
    df = spark.read.parquet(args.data)

    # Categorical columns to encode
    cat_cols = ["gender", "InternetService", "Contract", "tenure_bucket"]
    idx_cols = [c + "_idx" for c in cat_cols]
    ohe_cols = [c + "_ohe" for c in cat_cols]

    indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in cat_cols]
    encoder = OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols, handleInvalid="keep")

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "is_senior", "has_partner", "has_dependents", "has_phone"]
    assembler = VectorAssembler(inputCols=num_cols + ohe_cols, outputCol="features")

    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30, regParam=0.0)

    pipeline = Pipeline(stages=indexers + [encoder, assembler, lr])

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    model = pipeline.fit(train)
    pred = model.transform(test)

    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(pred)

    print(f"\nTest AUC: {auc:.4f}")
    pred.select("customerID", "label", "probability", "prediction").show(10, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
