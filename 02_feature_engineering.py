import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to churn.csv")
    parser.add_argument("--out", type=str, required=True, help="Output parquet path")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("ChurnFeatureEngineering").master("local[*]").getOrCreate()

    df = spark.read.csv(args.data, header=True, inferSchema=True)

    # Label
    df = df.withColumn("label", when(col("Churn") == "Yes", 1).otherwise(0))

    # Simple binary features
    df = df.withColumn("is_senior", col("SeniorCitizen").cast("int"))
    df = df.withColumn("has_partner", when(col("Partner") == "Yes", 1).otherwise(0))
    df = df.withColumn("has_dependents", when(col("Dependents") == "Yes", 1).otherwise(0))
    df = df.withColumn("has_phone", when(col("PhoneService") == "Yes", 1).otherwise(0))

    # Tenure bucket
    df = df.withColumn(
        "tenure_bucket",
        when(col("tenure") < 6, "0-5")
        .when(col("tenure") < 12, "6-11")
        .when(col("tenure") < 24, "12-23")
        .when(col("tenure") < 48, "24-47")
        .otherwise("48+")
    )

    # (Optional) make sure numeric types are consistent
    df = df.withColumn("MonthlyCharges", col("MonthlyCharges").cast("double"))
    df = df.withColumn("TotalCharges", col("TotalCharges").cast("double"))
    df = df.withColumn("tenure", col("tenure").cast("int"))

    df.write.mode("overwrite").parquet(args.out)
    print(f"Wrote features to {args.out}")

    spark.stop()

if __name__ == "__main__":
    main()
