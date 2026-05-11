from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum as spark_sum, round as spark_round
from pyspark.sql.types import StructField, StructType, StringType, FloatType

###### CONFIG #####

spark = SparkSession.builder.appName("Medallion").getOrCreate()
path = "data"


def bronze_level(path=path):

    schema = StructType([
        StructField("order_id", StringType(), False),
        StructField("customer_id", StringType()),
        StructField("product", StringType()),
        StructField("amount", FloatType()),
        StructField("status", StringType())
    ])
    df = spark.read.csv(f"{path}/orders.csv", header=True, schema = schema)
    df.write.mode("overwrite").parquet(f"{path}/bronze/orders")


def silver_level(path=path):
    df = spark.read.parquet(f"{path}/bronze/orders")
    na_df = df.dropna(subset=["amount", "status"])
    na_df = na_df.filter(col("status") == "completed")
    dedup_df = na_df.dropDuplicates(["order_id"])

    dedup_df.write.mode("overwrite").parquet(f"{path}/silver/orders")


def gold_level(path=path):
    df = spark.read.parquet(f"{path}/silver/orders")

    df = df.withColumn("high_val", col("amount") > 1000)

    clean_df = df.groupBy("customer_id").agg(
        spark_round(spark_sum("amount"), 2).alias("total_spent")
    ).orderBy(col("total_spent").desc())

    clean_df.write.mode("overwrite").parquet(f"{path}/gold/orders")


bronze_level()
silver_level()
gold_level()