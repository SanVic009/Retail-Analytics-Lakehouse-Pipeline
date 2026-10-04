"""
Synthetic dataset generator — sharded, multi-format, memory-bounded.

Architecture (validated via profiling, see profile_harness.py history):
  - Each worker GENERATES its own chunk AND WRITES it to its own shard file.
    Nothing is ever shipped back to the parent process. No pickling of
    NumPy/Arrow arrays across the process boundary, no IPC, no parent-side
    aggregation. This was the single biggest lever found during profiling.
  - Chunk size is DERIVED from a validated memory formula, not hardcoded.
    The formula was empirically bracketed against a hard RLIMIT_AS ceiling
    (see JSONL: known-good at 1.2M rows/worker, known-bad at 1.23M) and
    found accurate to ~1-3%. A 10% safety margin covers that residual.
  - Generator -> Sink split: each output format owns its own serialization
    cost model (Resident/Transient bytes per row), because Parquet stays
    columnar end-to-end while CSV/JSONL are inherently row-oriented and
    have a structurally different (and higher) cost. One shared chunk-size
    formula across formats was the bug that caused the original mismatch.

Output: a sharded directory per run (part-00000.<ext>, part-00001.<ext>, ...).
No merge step. Merging Parquet shards back into one file would require
re-materializing multiple chunks in one process — reintroducing the exact
memory problem this design eliminates. CSV/JSONL shards CAN be safely
`cat`-ed together at the OS level after the fact if a single file is ever
needed; that's a cheap, separate, optional step, not part of generation.
"""

import argparse
import multiprocessing
import resource
import string
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── Hardcoded, validated constants ──────────────────────────────────────────
# These came from the profiling harness runs, not guesses. See conversation
# history / profile_report.md for the empirical basis of each number.

NUM_WORKERS = 12
MEMORY_BUDGET_GB = 16.0
SAFETY_MARGIN = 0.90  # covers ~1-3% run-to-run measurement variance observed

# Per-format cost model: (resident_bytes_per_row, transient_bytes_per_row)
# Measured at 1M-row chunks, RSS-based, via the profiling harness.
FORMAT_COST_MODEL = {
    "parquet": {"resident": 28.00, "transient": 112.87, "ext": "parquet"},
    "csv":     {"resident": 28.00, "transient": 327.42, "ext": "csv"},
    "jsonl":   {"resident": 28.00, "transient": 408.74, "ext": "jsonl"},
}

# Measured empty-spawn-worker VMS baseline (interpreter + imports, before
# any data generation). Same across formats since it's the same Python
# process baseline regardless of which sink it's about to use.
WORKER_BASELINE_VMS_BYTES = int(851.4 * 1024 * 1024)

HEADERS = ["id", "timestamp", "char_1", "char_2", "value_1", "value_2"]

_DT_START = int(datetime(2020, 1, 1).timestamp())
_DT_END = int(datetime(2025, 12, 31).timestamp())
_DT_RANGE = _DT_END - _DT_START
_CHARS = np.array(list(string.ascii_uppercase))


# ── Chunk sizing ─────────────────────────────────────────────────────────────

def safe_rows_per_worker(fmt: str, num_workers: int = NUM_WORKERS,
                          budget_gb: float = MEMORY_BUDGET_GB) -> int:
    """
    Derives a safe per-worker row count for a hard RLIMIT_AS-style memory
    ceiling, using the validated formula:

        safe_rows = ((budget/N - baseline_vms) * margin) / (resident + transient per row)

    This is NOT a hardcoded constant — it's recomputed from the format's
    measured cost model so it stays correct if FORMAT_COST_MODEL or the
    worker count ever changes.
    """
    model = FORMAT_COST_MODEL[fmt]
    budget_per_worker = (budget_gb * 1e9) / num_workers
    net_budget = (budget_per_worker - WORKER_BASELINE_VMS_BYTES) * SAFETY_MARGIN
    if net_budget <= 0:
        raise ValueError(
            f"Memory budget too small: {budget_per_worker/1e6:.1f}MB per worker "
            f"doesn't even cover the {WORKER_BASELINE_VMS_BYTES/1e6:.1f}MB baseline."
        )
    bytes_per_row = model["resident"] + model["transient"]
    return int(net_budget / bytes_per_row)


# ── Data generation ──────────────────────────────────────────────────────────

def generate_chunk(start_id: int, size: int) -> dict:
    """Vectorized, columnar generation. No per-row Python objects."""
    ids = np.arange(start_id, start_id + size, dtype=np.int32)
    timestamps = (np.random.randint(0, _DT_RANGE, size=size) + _DT_START).astype("datetime64[s]")
    char_1 = _CHARS[np.random.randint(0, 26, size=size)]
    char_2 = _CHARS[np.random.randint(0, 26, size=size)]
    value_1 = np.random.randint(0, 10_001, size=size, dtype=np.int32)
    value_2 = np.random.randint(0, 10_001, size=size, dtype=np.int32)
    return {
        "id": ids, "timestamp": timestamps,
        "char_1": char_1, "char_2": char_2,
        "value_1": value_1, "value_2": value_2,
    }


# ── Sinks: each owns its own serialization + write strategy ─────────────────

class ParquetSink:
    """Stays columnar end-to-end. No concat_tables, no .tolist() on strings."""

    SCHEMA = pa.schema([
        ("id", pa.int32()),
        ("timestamp", pa.timestamp("s")),
        ("char_1", pa.string()),
        ("char_2", pa.string()),
        ("value_1", pa.int32()),
        ("value_2", pa.int32()),
    ])

    def write(self, data: dict, path: Path) -> None:
        table = pa.table({
            "id": pa.array(data["id"], type=pa.int32()),
            "timestamp": pa.array(data["timestamp"], type=pa.timestamp("s")),
            "char_1": pa.array(data["char_1"], type=pa.string()),
            "char_2": pa.array(data["char_2"], type=pa.string()),
            "value_1": pa.array(data["value_1"], type=pa.int32()),
            "value_2": pa.array(data["value_2"], type=pa.int32()),
        }, schema=self.SCHEMA)
        pq.write_table(table, path, compression="snappy", use_dictionary=True)


class CsvSink:
    """Row-oriented by nature. .tolist() cost is real and unavoidable here."""

    def write(self, data: dict, path: Path) -> None:
        ids = data["id"].tolist()
        timestamps = data["timestamp"].astype(str).tolist()
        char_1 = data["char_1"].tolist()
        char_2 = data["char_2"].tolist()
        value_1 = data["value_1"].tolist()
        value_2 = data["value_2"].tolist()
        with open(path, "w", newline="") as f:
            f.write(",".join(HEADERS) + "\n")
            for row in zip(ids, timestamps, char_1, char_2, value_1, value_2):
                f.write(",".join(str(v) for v in row) + "\n")


class JsonlSink:
    """Row-oriented, worst-case Transient cost of the three (measured, not assumed)."""

    def write(self, data: dict, path: Path) -> None:
        import orjson
        ids = data["id"].tolist()
        timestamps = data["timestamp"].astype(str).tolist()
        char_1 = data["char_1"].tolist()
        char_2 = data["char_2"].tolist()
        value_1 = data["value_1"].tolist()
        value_2 = data["value_2"].tolist()
        with open(path, "wb") as f:
            for i in range(len(ids)):
                f.write(orjson.dumps({
                    "id": ids[i], "timestamp": timestamps[i],
                    "char_1": char_1[i], "char_2": char_2[i],
                    "value_1": value_1[i], "value_2": value_2[i],
                }))
                f.write(b"\n")


SINKS = {"parquet": ParquetSink, "csv": CsvSink, "jsonl": JsonlSink}


# ── Worker: generate AND write, own process, own memory, no IPC ────────────

def worker_main(worker_id: int, fmt: str, start_id: int, num_rows: int,
                 output_dir: str, limit_bytes: int) -> None:
    """
    Runs entirely inside one worker process. Generates its chunk, writes its
    shard, exits. The only thing that ever crosses back to the parent is
    this process's exit code — no data.
    """
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    try:
        ext = FORMAT_COST_MODEL[fmt]["ext"]
        shard_path = Path(output_dir) / f"part-{worker_id:05d}.{ext}"
        data = generate_chunk(start_id, num_rows)
        sink = SINKS[fmt]()
        sink.write(data, shard_path)
    except MemoryError:
        print(f"[worker {worker_id}] MemoryError: exceeded {limit_bytes/1e6:.0f}MB cap "
              f"at ~{num_rows:,} assigned rows. Chunk size formula needs revisiting.",
              file=sys.stderr)
        sys.exit(3)
    except Exception:
        import traceback
        print(f"[worker {worker_id}] Unexpected failure (NOT a memory-cap issue):",
              file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


# ── Orchestration ────────────────────────────────────────────────────────────

CONCURRENCY = 6  # empirically measured throughput peak; 8+ regresses (see sweep results)

def generate_dataset(fmt: str, total_rows: int, output_dir: str,
                      budget_gb: float = MEMORY_BUDGET_GB,
                      concurrency: int = CONCURRENCY) -> None:
    if fmt not in SINKS:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from: {list(SINKS)}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    max_safe_chunk = safe_rows_per_worker(fmt, concurrency, budget_gb)
    num_shards = -(-total_rows // max_safe_chunk)  # ceil div
    rows_per_shard = -(-total_rows // num_shards)  # ceil div
    limit_bytes = int((budget_gb * 1e9) / concurrency)

    print(f"Generating {total_rows:,} rows as {fmt.upper()}")
    print(f"  {num_shards} shards, ~{rows_per_shard:,} rows/shard (safe ceiling: {max_safe_chunk:,})")
    print(f"  Running {concurrency} concurrent workers/wave, "
          f"{-(-num_shards // concurrency)} wave(s)")
    print(f"  Per-worker RLIMIT_AS: {limit_bytes/1e6:.0f}MB")

    t0 = time.perf_counter()
    all_exitcodes = []
    row_cursor = 1
    shard_id = 0
    while shard_id < num_shards:
        wave = []
        for _ in range(concurrency):
            if shard_id >= num_shards:
                break
            n = min(rows_per_shard, total_rows - (row_cursor - 1))
            p = multiprocessing.Process(
                target=worker_main,
                args=(shard_id, fmt, row_cursor, n, str(out_path), limit_bytes),
            )
            p.start()
            wave.append(p)
            row_cursor += n
            shard_id += 1
        for p in wave:
            p.join()
            all_exitcodes.append(p.exitcode)

    elapsed = time.perf_counter() - t0
    mem_failed = sum(1 for c in all_exitcodes if c == 3)
    other_failed = sum(1 for c in all_exitcodes if c not in (0, 3))
    if mem_failed:
        print(f"FAILED: {mem_failed}/{num_shards} shards hit their memory cap (exit 3). "
              f"The chunk-size formula needs revisiting for this format/scale.", file=sys.stderr)
    if other_failed:
        print(f"FAILED: {other_failed}/{num_shards} shards crashed for an unrelated "
              f"reason. Check stderr above.", file=sys.stderr)
    if mem_failed or other_failed:
        sys.exit(1)

    print(f"Done in {elapsed:.2f}s -> {out_path}/ ({num_shards} shards)")


def main():
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=list(SINKS), required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="data/output")
    args = parser.parse_args()
    generate_dataset(args.format, args.rows, args.output_dir)


if __name__ == "__main__":
    main()