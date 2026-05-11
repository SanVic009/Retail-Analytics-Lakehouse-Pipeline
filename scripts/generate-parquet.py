import random
import string
from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

import time

# ── CONFIG ───────────────────────────────────────────────────────────────────
NUM_ROWS = 1000000
OUTPUT_FILE = "data/dataset.parquet"
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.time()
def random_char():
    return random.choice(string.ascii_uppercase)

def random_datetime(start_year=2020, end_year=2025):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

# ── Build columns ─────────────────────────────────────────────────────────────
ids        = list(range(1, NUM_ROWS + 1))
timestamps = [random_datetime() for _ in range(NUM_ROWS)]
char_1     = [random_char() for _ in range(NUM_ROWS)]
char_2     = [random_char() for _ in range(NUM_ROWS)]
value_1    = [random.randint(0, 10_000) for _ in range(NUM_ROWS)]
value_2    = [random.randint(0, 10_000) for _ in range(NUM_ROWS)]

# ── Build PyArrow table with explicit schema ──────────────────────────────────
schema = pa.schema([
    ("id",        pa.int32()),
    ("timestamp", pa.timestamp("s")),   # second-level precision
    ("char_1",    pa.string()),
    ("char_2",    pa.string()),
    ("value_1",   pa.int32()),
    ("value_2",   pa.int32()),
])

table = pa.table(
    {
        "id":        pa.array(ids,        type=pa.int32()),
        "timestamp": pa.array(timestamps, type=pa.timestamp("s")),
        "char_1":    pa.array(char_1,     type=pa.string()),
        "char_2":    pa.array(char_2,     type=pa.string()),
        "value_1":   pa.array(value_1,    type=pa.int32()),
        "value_2":   pa.array(value_2,    type=pa.int32()),
    },
    schema=schema,
)
end_time = time.time()
print("Time taken: ", end_time-start_time)

# ── Write ─────────────────────────────────────────────────────────────────────
pq.write_table(table, OUTPUT_FILE, compression="snappy")

t2 = time.time()
print("End time: ", t2-start_time)

print(f"✅  Generated {NUM_ROWS} rows → {OUTPUT_FILE}")
print(table.schema)