import os
import sys
import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StringType, StructField, StructType, IntegerType, DoubleType 
from pyspark.sql.functions import col, count, sum as spark_sum, lit, current_timestamp
from delta import configure_spark_with_delta_pip

# for pipeline/exc.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipeline")))

# for OpenExchange/pilot.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from OpenExchange.pilot import get_exchange_rates

builder = SparkSession.builder \
    .appName("Medallion") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
spark = configure_spark_with_delta_pip(builder).getOrCreate()
path = "data"

def bronze():
    bronze_orders()
    bronze_exchange_rate()
    bronze_mapping()

def bronze_orders(path=path):
    schema = StructType([
        StructField("InvoiceNo", StringType()),
        StructField("StockCode", StringType()),
        StructField("Description", StringType()),
        StructField("Quantity", IntegerType()),
        StructField("InvoiceDate", StringType()),
        StructField("UnitPrice", DoubleType()),
        StructField("CustomerID", IntegerType()),
        StructField("Country", StringType())
    ])
    df = spark.read.csv(f"{path}/data.csv", header=True, schema=schema)
    df.write.format("delta").mode("overwrite").save(f"{path}/bronze/data")

def bronze_exchange_rate(path=path):
    data = {'disclaimer': 'Usage subject to terms: https://openexchangerates.org/terms', 'license': 'https://openexchangerates.org/license', 'timestamp': 1778587200, 'base': 'USD', 'rates': {'AED': 3.6725, 'AFN': 63.249998, 'ALL': 81.251649, 'AMD': 368.013978, 'ANG': 1.79, 'AOA': 913.116, 'ARS': 1393.1848, 'AUD': 1.383375, 'AWG': 1.8, 'AZN': 1.7, 'BAM': 1.665737, 'BBD': 2, 'BDT': 122.916559, 'BGN': 1.665298, 'BHD': 0.377386, 'BIF': 2976.572698, 'BMD': 1, 'BND': 1.272943, 'BOB': 6.909768, 'BRL': 4.8993, 'BSD': 1, 'BTC': 1.2383455e-05, 'BTN': 95.574584, 'BWP': 13.497654, 'BYN': 2.795772, 'BZD': 2.011097, 'CAD': 1.3704, 'CDF': 2279.894579, 'CHF': 0.781053, 'CLF': 0.0228, 'CLP': 897.33, 'CNH': 6.79243, 'CNY': 6.7921, 'COP': 3761.89, 'CRC': 458.445445, 'CUC': 1, 'CUP': 25.75, 'CVE': 93.806721, 'CZK': 20.71095, 'DJF': 178.062269, 'DKK': 6.362014, 'DOP': 59.013903, 'DZD': 132.259342, 'EGP': 52.9615, 'ERN': 15, 'ETB': 156.133921, 'EUR': 0.851498, 'FJD': 2.18635, 'FKP': 0.73895, 'GBP': 0.73895, 'GEL': 2.675, 'GGP': 0.73895, 'GHS': 11.289162, 'GIP': 0.73895, 'GMD': 73.000001, 'GNF': 8774.150304, 'GTQ': 7.629479, 'GYD': 209.200861, 'HKD': 7.82812, 'HNL': 26.589787, 'HRK': 6.415299, 'HTG': 130.64142, 'HUF': 304.069622, 'IDR': 17496.642694, 'ILS': 2.9102, 'IMP': 0.73895, 'INR': 95.685545, 'IQD': 1309.902437, 'IRR': 1311552.5, 'ISK': 122.44, 'JEP': 0.73895, 'JMD': 158.005708, 'JOD': 0.709, 'JPY': 157.57366667, 'KES': 129.17, 'KGS': 87.45, 'KHR': 4011.487757, 'KMF': 419.000301, 'KPW': 900, 'KRW': 1488.371388, 'KWD': 0.30802, 'KYD': 0.833284, 'KZT': 463.807083, 'LAK': 21920.355972, 'LBP': 89555.517385, 'LKR': 322.987988, 'LRD': 182.99351, 'LSL': 16.528834, 'LYD': 6.326359, 'MAD': 9.125952, 'MDL': 17.113657, 'MGA': 4178.40571, 'MKD': 52.476556, 'MMK': 2099.81, 'MNT': 3569.47, 'MOP': 8.063178, 'MRU': 39.888088, 'MUR': 46.699999, 'MVR': 15.41, 'MWK': 1734.037845, 'MXN': 17.23272, 'MYR': 3.934, 'MZN': 63.899993, 'NAD': 16.528764, 'NGN': 1371.01, 'NIO': 36.801716, 'NOK': 9.16625, 'NPR': 152.91916, 'NZD': 1.6803, 'OMR': 0.384508, 'PAB': 1, 'PEN': 3.427133, 'PGK': 4.355547, 'PHP': 61.452002, 'PKR': 278.561996, 'PLN': 3.616522, 'PYG': 6104.062402, 'QAR': 3.644997, 'RON': 4.4344, 'RSD': 99.956, 'RUB': 73.826257, 'RWF': 1462.5054, 'SAR': 3.751783, 'SBD': 8.032258, 'SCR': 13.866578, 'SDG': 600.5, 'SEK': 9.270775, 'SGD': 1.2719, 'SHP': 0.73895, 'SLE': 24.6, 'SLL': 20969.5, 'SOS': 571.487172, 'SRD': 37.4035, 'SSP': 130.26, 'STD': 22281.8, 'STN': 20.866514, 'SVC': 8.74942, 'SYP': 13002, 'SZL': 16.522873, 'THB': 32.41, 'TJS': 9.349521, 'TMT': 3.51, 'TND': 2.908098, 'TOP': 2.40776, 'TRY': 45.400473, 'TTD': 6.786301, 'TWD': 31.515, 'TZS': 2587.5, 'UAH': 43.949562, 'UGX': 3758.5142, 'USD': 1, 'UYU': 39.765445, 'UZS': 12130.738534, 'VES': 499.834804, 'VND': 26338.5, 'VUV': 119.389, 'WST': 2.74422, 'XAF': 558.54631, 'XAG': 0.01188743, 'XAU': 0.00021246, 'XCD': 2.70255, 'XCG': 1.802186, 'XDR': 0.69336, 'XOF': 558.54631, 'XPD': 0.00067179, 'XPF': 101.610783, 'XPT': 0.0004832, 'YER': 238.600042, 'ZAR': 16.50638, 'ZMW': 18.82407, 'ZWG': 25.3626, 'ZWL': 322}}
    schema = StructType([
        StructField("Currency_Code", StringType()),
        StructField("Rate", DoubleType()),
    ])
    # data = get_exchange_rates()
    rates = data['rates']
    records = [{"Currency_Code": k, "Rate": float(v)} for k, v in rates.items()]
    df = spark.createDataFrame(records, schema=schema)
    df.write.format("delta").mode("overwrite").save(f"{path}/bronze/exchange_rate")

def bronze_mapping():
    mapping = [ ("United Kingdom", "GBP"), ("France", "EUR"), ("Australia", "AUD"), ("Netherlands", "EUR"), ("Germany", "EUR"), ("Norway", "NOK"), ("EIRE", "EUR"), ("Switzerland", "CHF"), ("Spain", "EUR"), ("Poland", "PLN"), ("Portugal", "EUR"), ("Italy", "EUR"), ("Belgium", "EUR"), ("Lithuania", "EUR"), ("Japan", "JPY"), ("Iceland", "ISK"), ("Channel Islands", "GBP"), ("Denmark", "DKK"), ("Cyprus", "EUR"), ("Sweden", "SEK"), ("Austria", "EUR"), ("Israel", "ILS"), ("Finland", "EUR"), ("Bahrain", "BHD"), ("Greece", "EUR"), ("Hong Kong", "HKD"), ("Singapore", "SGD"), ("Lebanon", "LBP"), ("United Arab Emirates", "AED"), ("Saudi Arabia", "SAR"), ("Czech Republic", "CZK"), ("Canada", "CAD"), ("Unspecified", None), ("Brazil", "BRL"), ("USA", "USD"), ("European Community", "EUR"), ("Malta", "EUR"), ("RSA", "ZAR"), ]
    mapping_df = spark.createDataFrame(mapping, ["Country", "Currency_Code"])
    mapping_df.write.format("delta").mode("overwrite").save(f"{path}/static/mapping")

def silver(path=path):
    df = spark.read.format("delta").load(f"{path}/bronze/data")
    df = df.dropDuplicates()
    prc_df = df.filter(col("UnitPrice") > 0)
    clean_df = prc_df.filter(col("Quantity") > 0)

    silver_customers(clean_df)
    silver_product(clean_df)

def silver_customers(df: DataFrame):
    cust_df = df.dropna(subset=["CustomerID"])
    amt_df = cust_df.withColumn("Amount", col("UnitPrice")*col("Quantity"))
    amt_df.write.format("delta").mode("overwrite").save(f"{path}/silver/customer")


# Table 2
def silver_product(df: DataFrame):
    amt_df = df.withColumn("Amount", col("UnitPrice")*col("Quantity"))
    amt_df.write.format("delta").mode("overwrite").save(f"{path}/silver/product")


def silver_exchange_rate():
    df = spark.read.format("delta").load(f"{path}/bronze/exchange_rate")
    dupe_df = df.dropDuplicates()
    neg_df = dupe_df.filter(col("Rate") > 0)
    neg_df.write.format("delta").mode("overwrite").save(f"{path}/silver/exchange_rate")

def gold(path=path):
    gold_customer()
    gold_product()
    gold_exchange_rate()

def gold_customer():
    df = spark.read.format("delta").load(f"{path}/silver/customer")
    new = df.groupBy("CustomerID").agg(
        spark_sum("Amount").alias("Revenue")
    ).orderBy(col("Revenue").desc())
    new.write.format("delta").mode("overwrite").save(f"{path}/gold/customer")

def gold_product():
    df = spark.read.format("delta").load(f"{path}/silver/product")
    new = df.groupBy("Country", "Description").agg(
        spark_sum("Amount").alias("Revenue")
    ).orderBy(col("Revenue").desc())
    new.write.format("delta").mode("overwrite").save(f"{path}/gold/product")

def gold_exchange_rate():
    df = spark.read.format("delta").load(f"{path}/silver/product")
    mapping = spark.read.format("delta").load(f"{path}/static/mapping")
    exchange_rate = spark.read.format("delta").load(f"{path}/silver/exchange_rate")

    new = df.groupBy("Country", "Description").agg(
        spark_sum("Amount").alias("Revenue")
    ).orderBy(col("Revenue").desc())

    final_rev = new.alias("d") \
                .join(mapping.alias("m"), col("d.Country") == col("m.Country")) \
                .join(exchange_rate.alias("e"), col("m.Currency_Code") == col("e.Currency_Code")) \
                .select(
                    col("d.Country"),
                    col("d.Revenue").alias("Revenue_local"),
                    col("d.Description"),
                    (col("d.Revenue")*col("e.Rate")).alias("Revenue_USD")
                )
    final_rev.write.format("delta").mode("overwrite").save(f"{path}/gold/exchange")