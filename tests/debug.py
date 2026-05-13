import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").getOrCreate()

df = spark.createDataFrame([{"a": 1}, {"a": 2}])

df.show()