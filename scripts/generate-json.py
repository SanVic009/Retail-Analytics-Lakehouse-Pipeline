import json
import random
import string
from datetime import datetime, timedelta

import time

# ── CONFIG ───────────────────────────────────────────────────────────────────
NUM_ROWS    = 1000000
OUTPUT_FILE = "data/dataset.json"
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.time()
def random_char():
    return random.choice(string.ascii_uppercase)

def random_datetime(start_year=2020, end_year=2025):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def generate_row(row_id):
    return {
        "id":        row_id,
        "timestamp": random_datetime().strftime("%Y-%m-%dT%H:%M:%S"),
        "char_1":    random_char(),
        "char_2":    random_char(),
        "value_1":   random.randint(0, 10_000),
        "value_2":   random.randint(0, 10_000),
    }

# ── Generate ──────────────────────────────────────────────────────────────────
dataset = [generate_row(i) for i in range(1, NUM_ROWS + 1)]

end_time = time.time()
print("Time taken: ", end_time-start_time)
# ── Write ─────────────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w") as f:
    json.dump(dataset, f, indent=2)

t2 = time.time()
print("End time: ", t2-start_time)

print(f"✅  Generated {NUM_ROWS} rows → {OUTPUT_FILE}")