from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, StructField, StructType, IntegerType, FloatType 
from pyspark.sql.functions import col, count, sum as spark_sum
from delta import configure_spark_with_delta_pip
builder = SparkSession.builder \
    .appName("Medallion") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
spark = configure_spark_with_delta_pip(builder).getOrCreate()
df = spark.read.format("delta").load("data/silver/product")
print(type(df))
new = df.groupBy("Country", "Description").agg(
    spark_sum("Amount").alias("Revenue")
).orderBy(col("Revenue").desc())
print(type(new))
new.show(5)