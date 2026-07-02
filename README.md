# Medallion Data Pipeline with PySpark, Delta Lake, and Airflow

This repository implements a three-tier Medallion architecture data pipeline (Bronze, Silver, and Gold) to process online retail transaction data and integrate daily currency exchange rates for local-to-USD revenue conversions.

## Architecture

The pipeline follows the Lakehouse design pattern, decoupling compute (Apache Spark) and storage (Delta Lake on a filesystem/object storage), organized into three distinct processing zones:

### 1. Bronze Layer (Ingestion)
* **Orders Data**: Reads raw transaction data from CSV and writes it to Delta format at `data/bronze/data`.
* **Exchange Rates**: Ingests currency rates relative to USD and writes to `data/bronze/exchange_rate`.
* **Currency Mapping**: Creates a static mapping table (`data/static/mapping`) linking countries to their respective currency codes.

### 2. Silver Layer (Deduplication and Cleaning)
* **Orders Cleaning**: Reads from bronze orders, removes exact duplicate records, and filters for high-quality transactions (where `UnitPrice > 0` and `Quantity > 0`).
* **Customer Table**: Drops transactions lacking a valid `CustomerID` and calculates order amount. Persisted at `data/silver/customer`.
* **Product Table**: Calculates order amount for all clean records. Persisted at `data/silver/product`.
* **Exchange Rate Cleaning**: Deduplicates the ingested exchange rates and filters out negative/invalid rates. Persisted at `data/silver/exchange_rate`.

### 3. Gold Layer (Business Aggregations)
* **Customer Analytics**: Groups cleaned transactions by `CustomerID` to aggregate total revenue. Persisted at `data/gold/customer`.
* **Product Analytics**: Groups transactions by `Country` and `Description` to calculate total revenue. Persisted at `data/gold/product`.
* **Currency Conversion**: Joins product aggregates with country-to-currency mappings and daily exchange rates. Computes local-currency revenue and converted USD revenue. Persisted at `data/gold/exchange`.

## Technology Stack

* **Processing Engine**: Apache Spark (PySpark)
* **Table Format**: Delta Lake (enabling ACID transactions, schema enforcement, and version history)
* **Orchestrator**: Apache Airflow

## Airflow Orchestration

The pipeline is orchestrated using an Airflow DAG (`medallion_pipeline`) defined in `dags/dag.py`.

* **Schedule Interval**: `@daily`
* **Workflow Tasks**:
  1. `run_bronze`: Triggers Bronze layer ingestion.
  2. `run_silver`: Cleans orders and populates customer and product silver tables.
  3. `run_silver_exchange`: Cleans the exchange rates table.
  4. `run_gold`: Performs customer and product level aggregations.
  5. `run_gold_exchange`: Computes currency conversion and generates the final Gold reporting table.
* **Dependencies**:
  * `run_bronze` must complete before starting `run_silver` or `run_silver_exchange`.
  * `run_silver` must complete before starting `run_gold`.
  * Both `run_silver` and `run_silver_exchange` must complete before executing `run_gold_exchange`.

## Project Structure

* `dags/`: Contains the Airflow DAG definition (`dag.py`).
* `pipeline/`: Contains the core data processing functions (`exc.py`) using PySpark and Delta Lake.
* `data/`: Local storage directory for the raw CSV and Delta Lake files.
* `pysparkk/`: Contains standalone PySpark pipeline implementations and utilities.
* `scripts/`: Benchmark, data generation, and formatting scripts.
* `tests/`: Testing and data inspection scripts.

## Setup

1. **Clone the Repository**:
   Clone this project repository to your local system.

2. **Environment Setup**:
   Create a Python environment and install the package requirements listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Acquisition**:
   Download the Online Retail II dataset from Kaggle:
   https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
   
   Place the downloaded CSV file (`online_retail_II.csv`) inside the `data/` directory.

4. **Data Linkage**:
   The ingestion code expects the source file to be named `data.csv`. Create a symbolic link in the `data/` directory pointing to the downloaded Kaggle file:
   ```bash
   ln -sf online_retail_II.csv data/data.csv
   ```

## Running the Pipeline

### Running Standalone
To execute the pipeline manually using PySpark without Airflow:
```bash
python -c "import sys; sys.path.append('pipeline'); import exc; exc.bronze(); exc.silver(); exc.silver_exchange_rate(); exc.gold()"
```

### Running with Airflow
1. Ensure your Airflow environment is configured to append the project directory to the Python path.
2. Copy the DAG file to your Airflow DAGs directory:
   ```bash
   cp dags/dag.py ~/airflow/dags/
   ```
3. Start the Airflow scheduler and webserver to trigger and monitor the pipeline.
