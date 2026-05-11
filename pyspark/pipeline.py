from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum as spark_sum
from pyspark.sql.types import StructType, StructField, FloatType, StringType

spark = SparkSession.builder.appName("Pipeline").getOrCreate()

# Schema: order_id,customer_id,product,amount,status
schema = StructType([
    StructField("order_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("product", StringType(), True ),
    StructField("amount", FloatType(), True ),
    StructField("status", StringType(), True )
])
df = spark.read.csv("data/orders.csv", header=True,schema=schema)


# df_new = df.filter(rdd.isNotNull())

na_df = df.dropna(subset=["amount", "status"])
completed_df = na_df.filter(col("status") == "completed")

cust_df = completed_df.groupBy("customer_id").agg(
    spark_sum("amount").alias("amount_spent")
)

final_df = cust_df.withColumn("high_val", col("amount_spent") > 1000)

# final_df.write.parquet("data/orders")