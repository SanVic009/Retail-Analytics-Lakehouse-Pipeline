from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, StructField, StructType, IntegerType, FloatType 
from pyspark.sql.functions import col, count, sum as spark_sum
from delta import configure_spark_with_delta_pip


builder = SparkSession.builder \
    .appName("Medallion") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()
path = "data"

def bronze(path=path):

    schema = StructType([
        StructField("InvoiceNo", StringType()),
        StructField("StockCode", StringType()),
        StructField("Description", StringType()),
        StructField("Quantity", IntegerType()),
        StructField("InvoiceDate", StringType()),
        StructField("UnitPrice", FloatType()),
        StructField("CustomerID", IntegerType()),
        StructField("Country", StringType())
    ])

    df = spark.read.csv(f"{path}/data.csv", header=True, schema=schema)
    df.write.format("delta").mode("overwrite").save(f"{path}/bronze/data")


def silver(path=path):
    df = spark.read.format("delta").load(f"{path}/bronze/data")
    df = df.dropDuplicates()
    prc_df = df.filter(col("UnitPrice") > 0)
    clean_df = prc_df.filter(col("Quantity") > 0)

    silver_customers(clean_df)
    silver_product(clean_df)


# Table 1
def silver_customers(df: DataFrame):
    cust_df = df.dropna(subset=["CustomerID"])
    amt_df = cust_df.withColumn("Amount", col("UnitPrice")*col("Quantity"))
    amt_df.write.format("delta").mode("overwrite").save(f"{path}/silver/customer")


# Table 2
def silver_product(df: DataFrame):
    amt_df = df.withColumn("Amount", col("UnitPrice")*col("Quantity"))
    amt_df.write.format("delta").mode("overwrite").save(f"{path}/silver/product")

def gold(path=path):
    gold_1()
    gold_2()

def gold_1():
    df = spark.read.format("delta").load(f"{path}/silver/customer")
    new = df.groupBy("CustomerID").agg(
        spark_sum("Amount").alias("Revenue")
    ).orderBy(col("Revenue").desc())

    new.write.format("delta").mode("overwrite").save(f"{path}/gold/customer")


def gold_2():
    df = spark.read.format("delta").load(f"{path}/silver/product")
    new = df.groupBy("Country", "Description").agg(
        spark_sum("Amount").alias("Revenue")
    ).orderBy(col("Revenue").desc())
    new.write.format("delta").mode("overwrite").save(f"{path}/gold/product")

if __name__ == "__main__":
    bronze()
    silver()
    gold()
