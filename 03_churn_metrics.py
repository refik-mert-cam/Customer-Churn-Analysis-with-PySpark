import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to churn_fe.parquet")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("ChurnMetrics").master("local[*]").getOrCreate()
    df = spark.read.parquet(args.data)

    print("\nOverall churn rate:")
    df.select(avg(col("label")).alias("churn_rate")).show()

    print("\nChurn by Contract:")
    df.groupBy("Contract").agg(
        count("*").alias("n"),
        avg("label").alias("churn_rate")
    ).orderBy(col("churn_rate").desc()).show(truncate=False)

    print("\nChurn by InternetService:")
    df.groupBy("InternetService").agg(
        count("*").alias("n"),
        avg("label").alias("churn_rate")
    ).orderBy(col("churn_rate").desc()).show(truncate=False)

    print("\nChurn by tenure bucket:")
    df.groupBy("tenure_bucket").agg(
        count("*").alias("n"),
        avg("label").alias("churn_rate")
    ).orderBy("tenure_bucket").show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
