from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, avg, count


spark = SparkSession.builder \
    .appName("MyFirstApp") \
    .getOrCreate()
    
#df = spark.read.csv("your_file.csv", header=True, inferSchema=True)

df = spark.createDataFrame([
    (1, "Alice", 25),
    (2, "Bob", 30),
    (3, "Charlie", 22)
], ["id", "name", "age"])

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True)
])
df = spark.read.csv("your_file.csv", header=True, schema=schema)


new_df = df.filter(col("age").isNotNull())
new_df = new_df.withColumn("senior", col("age") > 27)
new_df.groupBy("senior").agg(
    count("id"),
    avg("age")
)

new_df.createOrReplaceTempView("People")
spark.sql("""
    SELECT 
        age > 24 as senior,
        COUNT(id) as total_people,
        AVG(age) as avg_age
    FROM people
    GROUP BY age > 24
""")
print(new_df.rdd.getNumPartitions())

# new_df.show()

#  1. Filter
# df.filter(df['age'] > 24).show()

#  2. Select
# df.select("name", "age").show()
# df.select("name", "age").explain()

#  3. Add a new column
# from pyspark.sql.functions import col
# df.withColumn("age_plus_10", col("age") + 10).show()

#  4. This is the big one - run this and tell me what you see
# df.filter(df['age'] > 24).explain()

# df.show()
# df.printSchema()
# df.count()