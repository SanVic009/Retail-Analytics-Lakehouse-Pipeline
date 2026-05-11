import csv
import io
import json
import string
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── CONFIG ───────────────────────────────────────────────────────────────────
NUM_ROWS    = 100000000
OUTPUT_FILE = "data/dataset.json"
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = ["id", "timestamp", "char_1", "char_2", "value_1", "value_2"]

# Precomputed once at module level — not inside the function (fix #2)
_DT_START = int(datetime(2020, 1, 1).timestamp())
_DT_END   = int(datetime(2025, 12, 31).timestamp())
_DT_RANGE = _DT_END - _DT_START
_CHARS    = np.array(list(string.ascii_uppercase))

# ── Vectorized generator (fix #1 + #2) ───────────────────────────────────────

def generate_chunk(args):
    start_id, end_id = args
    size = end_id - start_id

    ids        = np.arange(start_id, end_id, dtype=np.int32)
    timestamps = (np.random.randint(0, _DT_RANGE, size=size) + _DT_START).astype("datetime64[s]")
    char_1     = _CHARS[np.random.randint(0, 26, size=size)]
    char_2     = _CHARS[np.random.randint(0, 26, size=size)]
    value_1    = np.random.randint(0, 10_001, size=size, dtype=np.int32)
    value_2    = np.random.randint(0, 10_001, size=size, dtype=np.int32)

    # Column-oriented — no per-row dicts (fix #3)
    return {
        "id":        ids,
        "timestamp": timestamps,
        "char_1":    char_1,
        "char_2":    char_2,
        "value_1":   value_1,
        "value_2":   value_2,
    }

def chunk_ranges(total, n_chunks):
    size = total // n_chunks
    ranges = []
    for i in range(n_chunks):
        start = i * size + 1
        end   = (i + 1) * size + 1 if i < n_chunks - 1 else total + 1
        ranges.append((start, end))
    return ranges

# ── Writers ───────────────────────────────────────────────────────────────────

def write_csv(chunks, output_file):
    with open(output_file, "w", newline="") as f:
        f.write(",".join(HEADERS) + "\n")
        for c in chunks:
            rows = np.column_stack([
                c["id"],
                c["timestamp"].astype(str),
                c["char_1"],
                c["char_2"],
                c["value_1"],
                c["value_2"],
            ])
            buf = io.StringIO()
            np.savetxt(buf, rows, delimiter=",", fmt="%s")
            f.write(buf.getvalue())

def write_json(chunks, output_file):
    with open(output_file, "w") as f:
        f.write("[\n")
        first = True
        for c in chunks:
            size = len(c["id"])
            for i in range(size):
                if not first:
                    f.write(",\n")
                json.dump({
                    "id":        int(c["id"][i]),
                    "timestamp": str(c["timestamp"][i]),
                    "char_1":    str(c["char_1"][i]),
                    "char_2":    str(c["char_2"][i]),
                    "value_1":   int(c["value_1"][i]),
                    "value_2":   int(c["value_2"][i]),
                }, f)
                first = False
        f.write("\n]")

def write_jsonl(chunks, output_file):
    with open(output_file, "w") as f:
        for c in chunks:
            size = len(c["id"])
            for i in range(size):
                f.write(json.dumps({
                    "id":        int(c["id"][i]),
                    "timestamp": str(c["timestamp"][i]),
                    "char_1":    str(c["char_1"][i]),
                    "char_2":    str(c["char_2"][i]),
                    "value_1":   int(c["value_1"][i]),
                    "value_2":   int(c["value_2"][i]),
                }) + "\n")

def write_parquet(chunks, output_file):
    # Concatenate arrays directly — no list comprehension over dicts (fix #3)
    schema = pa.schema([
        ("id",        pa.int32()),
        ("timestamp", pa.timestamp("s")),
        ("char_1",    pa.string()),
        ("char_2",    pa.string()),
        ("value_1",   pa.int32()),
        ("value_2",   pa.int32()),
    ])

    table = pa.concat_tables([
        pa.table({
            "id":        pa.array(c["id"],                       type=pa.int32()),
            "timestamp": pa.array(c["timestamp"].astype("datetime64[ms]").tolist(), type=pa.timestamp("s")),
            "char_1":    pa.array(c["char_1"].tolist(),          type=pa.string()),
            "char_2":    pa.array(c["char_2"].tolist(),          type=pa.string()),
            "value_1":   pa.array(c["value_1"],                  type=pa.int32()),
            "value_2":   pa.array(c["value_2"],                  type=pa.int32()),
        }, schema=schema)
        for c in chunks
    ])

    pq.write_table(table, output_file, compression="snappy")

# ── Router ────────────────────────────────────────────────────────────────────

WRITERS = {
    ".csv":     write_csv,
    ".json":    write_json,
    ".jsonl":   write_jsonl,
    ".parquet": write_parquet,
}

def write(chunks, output_file):
    ext = Path(output_file).suffix.lower()
    writer = WRITERS.get(ext)
    if not writer:
        supported = ", ".join(WRITERS.keys())
        raise ValueError(f"Unsupported format '{ext}'. Supported: {supported}")
    writer(chunks, output_file)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t1 = time.time()

    cores  = cpu_count()
    ranges = chunk_ranges(NUM_ROWS, cores)

    print(f"⚙️  Using {cores} cores, {NUM_ROWS:,} rows...")

    with Pool(processes=cores) as pool:
        chunks = pool.map(generate_chunk, ranges)

    write(chunks, OUTPUT_FILE)

    t2 = time.time()
    print(f"⏱️  Time taken: {t2 - t1:.2f}s")
    print(f"✅  Generated {NUM_ROWS:,} rows → {OUTPUT_FILE}")