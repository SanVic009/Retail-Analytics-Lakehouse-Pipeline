import csv
import random
import string
from datetime import datetime, timedelta

import time

# ── CONFIG ──────────────────────────────────────────────────────────────────
NUM_ROWS = 10000000
OUTPUT_FILE = "data/dataset.csv"
# ────────────────────────────────────────────────────────────────────────────


start_time = time.time()

def random_char():
    """Single random uppercase letter."""
    return random.choice(string.ascii_uppercase)

def random_datetime(start_year=2020, end_year=2025):
    """Random datetime between two years."""
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def generate_row(row_id):
    return [
        row_id,                                                  # int   (ID)
        random_datetime().strftime("%Y-%m-%d %H:%M:%S"),        # datetime
        random_char(),                                           # char
        random_char(),                                           # char
        random.randint(0, 10_000),                              # int
        random.randint(0, 10_000),                              # int
    ]

HEADERS = ["id", "timestamp", "char_1", "char_2", "value_1", "value_2"]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADERS)
    for i in range(1, NUM_ROWS + 1):
        writer.writerow(generate_row(i))

end_time = time.time()
print("Time taken: ", end_time-start_time)
print(f"✅  Generated {NUM_ROWS} rows → {OUTPUT_FILE}")

