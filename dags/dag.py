import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)                    # for OpenExchange
sys.path.append(os.path.join(project_root, "pipeline"))  # for exc.py

from airflow.decorators import dag, task
from datetime import datetime
from exc import bronze, silver, silver_exchange_rate, gold, gold_exchange_rate

@dag(schedule="@daily", start_date=datetime(2025, 1, 1), catchup=False)
def medallion_pipeline():

    @task
    def run_bronze():
        bronze()

    @task
    def run_silver():
        silver()

    @task
    def run_silver_exchange():
        silver_exchange_rate()

    @task
    def run_gold():
       gold() 

    @task
    def run_gold_exchange():
        gold_exchange_rate()

    # dependency chain
    b = run_bronze()
    s = run_silver()
    se = run_silver_exchange()
    g = run_gold()
    ge = run_gold_exchange()

    b >> s >> g
    b >> se
    [s, se] >> ge

medallion_pipeline()