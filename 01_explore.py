import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to churn.csv")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("ChurnExplore").master("local[*]").getOrCreate()

    df = spark.read.csv(args.data, header=True, inferSchema=True)

    print("\nSchema:")
    df.printSchema()

    print("\nSample rows:")
    df.show(5, truncate=False)

    print("\nChurn distribution:")
    df.groupBy("Churn").count().show()

    print("\nBasic stats:")
    df.select("tenure", "MonthlyCharges", "TotalCharges").describe().show()

    print("\nNull counts per column:")
    nulls = df.select([
        (col(c).isNull().cast("int").alias(c)) for c in df.columns
    ]).groupBy().sum()
    nulls.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
