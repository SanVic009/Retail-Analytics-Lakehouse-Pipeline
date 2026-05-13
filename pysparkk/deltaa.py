
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum as spark_sum, round as spark_round
from pyspark.sql.types import StructField, StructType, StringType, FloatType
from delta import configure_spark_with_delta_pip

###### CONFIG #####

builder = SparkSession.builder \
    .appName("Medallion") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
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
    df.write.format("delta").mode("overwrite").save(f"{path}/bronze/orders")


def silver_level(path=path):
    df = spark.read.format("delta").load(f"{path}/bronze/orders")
    na_df = df.dropna(subset=["amount", "status"])
    na_df = na_df.filter(col("status") == "completed")
    dedup_df = na_df.dropDuplicates(["order_id"])

    dedup_df.write.format("delta").mode("overwrite").save(f"{path}/silver/orders")


def gold_level(path=path):
    df = spark.read.format("delta").load(f"{path}/silver/orders")

    df = df.withColumn("high_val", col("amount") > 1000)

    clean_df = df.groupBy("customer_id").agg(
        spark_round(spark_sum("amount"), 2).alias("total_spent")
    ).orderBy(col("total_spent").desc())

    clean_df.write.format("delta").mode("overwrite").save(f"{path}/gold/orders")


bronze_level()
silver_level()
gold_level()

spark.sql("DESCRIBE HISTORY delta.`/home/sanvict/Documents/Code/ETL/data/gold/orders`").show(truncate=False)