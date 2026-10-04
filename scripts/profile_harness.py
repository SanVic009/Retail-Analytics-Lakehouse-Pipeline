import os
import sys
import gc
import csv
import json
import time
import string
import argparse
import threading
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import matplotlib.pyplot as plt

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = Path("data")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# ── DATA GENERATION ───────────────────────────────────────────────────────────

def generate_data(size, start_id=1):
    """
    Generates synthetic data:
    - id: int32
    - timestamp: datetime64[s]
    - char_1, char_2: single uppercase letters (U1)
    - value_1, value_2: int32
    """
    ids = np.arange(start_id, start_id + size, dtype=np.int32)
    
    # Random timestamps between 2020-01-01 and 2025-12-31
    dt_start = int(datetime(2020, 1, 1).timestamp())
    dt_end = int(datetime(2025, 12, 31).timestamp())
    timestamps = (np.random.randint(0, dt_end - dt_start, size=size) + dt_start).astype("datetime64[s]")
    
    # Single uppercase letters
    chars = np.array(list(string.ascii_uppercase))
    char_1 = chars[np.random.randint(0, 26, size=size)]
    char_2 = chars[np.random.randint(0, 26, size=size)]
    
    # Random int32 values
    value_1 = np.random.randint(0, 10001, size=size, dtype=np.int32)
    value_2 = np.random.randint(0, 10001, size=size, dtype=np.int32)
    
    return {
        "id": ids,
        "timestamp": timestamps,
        "char_1": char_1,
        "char_2": char_2,
        "value_1": value_1,
        "value_2": value_2
    }

# ── SINKS ─────────────────────────────────────────────────────────────────────

class ParquetSink:
    def serialize(self, data):
        schema = pa.schema([
            ("id", pa.int32()),
            ("timestamp", pa.timestamp("s")),
            ("char_1", pa.string()),
            ("char_2", pa.string()),
            ("value_1", pa.int32()),
            ("value_2", pa.int32()),
        ])
        table = pa.table({
            "id": pa.array(data["id"], type=pa.int32()),
            "timestamp": pa.array(data["timestamp"], type=pa.timestamp("s")),
            "char_1": pa.array(data["char_1"], type=pa.string()),
            "char_2": pa.array(data["char_2"], type=pa.string()),
            "value_1": pa.array(data["value_1"], type=pa.int32()),
            "value_2": pa.array(data["value_2"], type=pa.int32()),
        }, schema=schema)
        return table
    
    def write(self, serialized_data, path):
        pq.write_table(serialized_data, path, compression="snappy")


class CsvSink:
    def serialize(self, data):
        # Convert numpy arrays to Python list of rows
        ids = data["id"].tolist()
        timestamps = data["timestamp"].astype(str).tolist()
        char_1 = data["char_1"].tolist()
        char_2 = data["char_2"].tolist()
        value_1 = data["value_1"].tolist()
        value_2 = data["value_2"].tolist()
        
        # Zip into row-oriented structure
        rows = list(zip(ids, timestamps, char_1, char_2, value_1, value_2))
        return rows
    
    def write(self, serialized_data, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "timestamp", "char_1", "char_2", "value_1", "value_2"])
            writer.writerows(serialized_data)


class JsonSink:
    def serialize(self, data):
        # Convert to Python collections
        ids = data["id"].tolist()
        timestamps = data["timestamp"].astype(str).tolist()
        char_1 = data["char_1"].tolist()
        char_2 = data["char_2"].tolist()
        value_1 = data["value_1"].tolist()
        value_2 = data["value_2"].tolist()
        
        # Serialize each row as a JSON line
        lines = [
            json.dumps({
                "id": ids[i],
                "timestamp": timestamps[i],
                "char_1": char_1[i],
                "char_2": char_2[i],
                "value_1": value_1[i],
                "value_2": value_2[i]
            })
            for i in range(len(ids))
        ]
        return lines
    
    def write(self, serialized_data, path):
        with open(path, "w") as f:
            for line in serialized_data:
                f.write(line + "\n")


def get_sink(name):
    if name.lower() == "parquet":
        return ParquetSink()
    elif name.lower() == "csv":
        return CsvSink()
    elif name.lower() in ("json", "jsonl"):
        return JsonSink()
    else:
        raise ValueError(f"Unknown sink type: {name}")

# ── CHILD WORKER FOR SINK PROFILING ───────────────────────────────────────────

def child_profiler_worker(conn, sink_name, size, path, phase_val):
    """
    Subprocess worker to profile a single sink.
    Updates the shared `phase_val` to communicate phase transitions:
      1: Generate
      2: Serialize
      3: Write
      4: Finished
    """
    try:
        import psutil
        proc = psutil.Process()
        baseline_vms = proc.memory_info().vms
        baseline_rss = proc.memory_info().rss
        
        gc.collect()
        
        # 1. Generate
        phase_val.value = 1
        t0 = time.perf_counter()
        data = generate_data(size)
        t1 = time.perf_counter()
        gen_time = t1 - t0
        
        # Measure resident bytes
        resident_bytes = sum(arr.nbytes for arr in data.values())
        
        # Hold briefly so the parent's RSS monitor can catch the steady-state memory of generated data
        time.sleep(0.15)
        
        # 2. Serialize
        phase_val.value = 2
        sink = get_sink(sink_name)
        t2 = time.perf_counter()
        serialized = sink.serialize(data)
        t3 = time.perf_counter()
        serialize_time = t3 - t2
        
        # Hold briefly to catch peak serialization RSS
        time.sleep(0.15)
        
        # 3. Write
        phase_val.value = 3
        t4 = time.perf_counter()
        sink.write(serialized, path)
        t5 = time.perf_counter()
        write_time = t5 - t4
        
        # Measure file size
        output_bytes = Path(path).stat().st_size
        
        # Cleanup file immediately to save space
        try:
            Path(path).unlink()
        except OSError:
            pass
            
        phase_val.value = 4
        
        # Send metrics back
        conn.send({
            "success": True,
            "gen_time": gen_time,
            "serialize_time": serialize_time,
            "write_time": write_time,
            "resident_bytes": resident_bytes,
            "output_bytes": output_bytes,
            "baseline_vms": baseline_vms,
            "baseline_rss": baseline_rss
        })
    except Exception as e:
        import traceback
        conn.send({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        conn.close()

# ── PROFILING A SINGLE SINK ───────────────────────────────────────────────────

def profile_sink(sink_name, size):
    """
    Runs profiling of a sink in a separate subprocess.
    The parent process monitors RSS of the child at a very high frequency.
    """
    print(f"Profiling {sink_name.upper()} with {size:,} rows...")
    parent_conn, child_conn = multiprocessing.Pipe()
    phase_val = multiprocessing.Value('i', 0)
    temp_path = BASE_DIR / f"temp_profile_{sink_name}.{sink_name}"
    
    p = multiprocessing.Process(
        target=child_profiler_worker,
        args=(child_conn, sink_name, size, str(temp_path), phase_val)
    )
    p.start()
    
    mem_history = []
    
    # High-frequency polling loop
    while p.is_alive():
        try:
            proc = psutil.Process(p.pid)
            # Include child processes if any
            total_rss = proc.memory_info().rss
            total_vms = proc.memory_info().vms
            for child in proc.children(recursive=True):
                try:
                    total_rss += child.memory_info().rss
                    total_vms += child.memory_info().vms
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            mem_history.append((time.perf_counter(), phase_val.value, total_rss, total_vms))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.002)  # sample every 2ms
        
    p.join()
    
    if parent_conn.poll():
        res = parent_conn.recv()
    else:
        res = {"success": False, "error": "No response from worker"}
        
    if not res.get("success"):
        print(f"  ❌ Failed: {res.get('error')}")
        if "traceback" in res:
            print(res["traceback"])
        return None
        
    # Analyze memory history to separate phases
    # Phases: 1 (generate), 2 (serialize), 3 (write)
    rss_by_phase = {0: [], 1: [], 2: [], 3: [], 4: []}
    vms_by_phase = {0: [], 1: [], 2: [], 3: [], 4: []}
    for t, ph, rss, vms in mem_history:
        rss_by_phase[ph].append(rss)
        vms_by_phase[ph].append(vms)
        
    # Extract baseline RSS (before generation starts)
    baseline_rss = rss_by_phase[1][0] if rss_by_phase[1] else 0
    # Extract steady-state RSS after generation completes (start of serialization)
    generated_rss = rss_by_phase[2][0] if rss_by_phase[2] else (rss_by_phase[1][-1] if rss_by_phase[1] else 0)
    
    # Peak RSS during serialization phase
    peak_serialize_rss = max(rss_by_phase[2]) if rss_by_phase[2] else generated_rss
    # Peak RSS during writing phase
    peak_write_rss = max(rss_by_phase[3]) if rss_by_phase[3] else peak_serialize_rss
    
    # Calculations
    transient_serialize_rss = max(0, peak_serialize_rss - generated_rss)
    transient_write_rss = max(0, peak_write_rss - generated_rss)
    transient_peak = max(transient_serialize_rss, transient_write_rss)
    
    # Repeat calculations for VMS (Virtual Memory Size)
    baseline_vms = vms_by_phase[1][0] if vms_by_phase[1] else 0
    generated_vms = vms_by_phase[2][0] if vms_by_phase[2] else (vms_by_phase[1][-1] if vms_by_phase[1] else 0)
    
    peak_serialize_vms = max(vms_by_phase[2]) if vms_by_phase[2] else generated_vms
    peak_write_vms = max(vms_by_phase[3]) if vms_by_phase[3] else peak_serialize_vms
    
    transient_serialize_vms = max(0, peak_serialize_vms - generated_vms)
    transient_write_vms = max(0, peak_write_vms - generated_vms)
    transient_peak_vms = max(transient_serialize_vms, transient_write_vms)
    
    res.update({
        "size": size,
        "baseline_rss": baseline_rss,
        "generated_rss": generated_rss,
        "peak_serialize_rss": peak_serialize_rss,
        "peak_write_rss": peak_write_rss,
        "transient_serialize": transient_serialize_rss,
        "transient_write": transient_write_rss,
        "transient_peak": transient_peak,
        "peak_rss_total": max(r[2] for r in mem_history) if mem_history else 0,
        
        "baseline_vms": baseline_vms,
        "generated_vms": generated_vms,
        "peak_serialize_vms": peak_serialize_vms,
        "peak_write_vms": peak_write_vms,
        "transient_peak_vms": transient_peak_vms,
        "peak_vms_total": max(r[3] for r in mem_history) if mem_history else 0
    })
    
    print(f"  Done. Gen: {res['gen_time']:.2f}s, Serialize: {res['serialize_time']:.2f}s, Write: {res['write_time']:.2f}s")
    print(f"  Resident: {res['resident_bytes']/(1024**2):.1f} MB, Transient peak RSS: {res['transient_peak']/(1024**2):.1f} MB, Transient peak VMS: {res['transient_peak_vms']/(1024**2):.1f} MB")
    
    return res

# ── RSS MONITOR FOR COMBINED PROCESSES ────────────────────────────────────────

class RSSMonitor(threading.Thread):
    def __init__(self, interval=0.020):
        super().__init__()
        self.interval = interval
        self.history = []
        self.running = False
        self.parent = psutil.Process()
        self.lock = threading.Lock()
        self.peak_combined_rss = 0
        self.peak_combined_vms = 0

    def run(self):
        self.running = True
        start_time = time.perf_counter()
        while self.running:
            try:
                children = self.parent.children(recursive=True)
                total_rss = 0
                total_vms = 0
                for child in children:
                    try:
                        total_rss += child.memory_info().rss
                        total_vms += child.memory_info().vms
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                t = time.perf_counter() - start_time
                with self.lock:
                    self.history.append((t, total_rss, total_vms))
                    if total_rss > self.peak_combined_rss:
                        self.peak_combined_rss = total_rss
                    if total_vms > self.peak_combined_vms:
                        self.peak_combined_vms = total_vms
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False

# ── EXIT CODE INTERPRETER ─────────────────────────────────────────────────────

def interpret_exit_codes(exit_codes):
    if not exit_codes:
        return "No workers executed."
    interpretations = []
    for i, code in enumerate(exit_codes):
        if code == 0:
            status = "Success"
        elif code == 3:
            status = "RLIMIT_AS Limit Exceeded (MemoryError raised in Python)"
        elif code == -9 or code == 9:
            status = "Kernel OOM Kill (SIGKILL)"
        elif code == -11 or code == 11:
            status = "Segmentation Fault (SIGSEGV - likely allocator allocation failure under limit)"
        elif code == 2:
            status = "RLIMIT_AS Initialization Failed"
        elif code == 1:
            status = "General Execution Exception"
        elif code < 0:
            import signal
            sig_num = -code
            try:
                sig_name = signal.Signals(sig_num).name
                status = f"Killed by Signal {sig_num} ({sig_name})"
            except ValueError:
                status = f"Killed by Signal {sig_num}"
        else:
            status = f"Unknown Error (Exit Code: {code})"
        interpretations.append(f"    Worker {i}: {status}")
    return "\n".join(interpretations)

# ── SWEEP WORKER ──────────────────────────────────────────────────────────────

def sweep_worker(worker_id, format_name, rows_per_worker, output_dir, limit_bytes=None):
    """
    Worker process for the throughput / RSS sweep.
    Generates and writes a single chunk.
    """
    if limit_bytes is not None:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception as e:
            print(f"Worker {worker_id}: failed to set RLIMIT_AS to {limit_bytes} bytes: {e}")
            sys.exit(2)
            
    try:
        sink = get_sink(format_name)
        ext = "jsonl" if format_name == "json" else format_name
        path = Path(output_dir) / f"shard_{worker_id}.{ext}"
        
        # Generate
        data = generate_data(rows_per_worker, start_id=worker_id * rows_per_worker + 1)
        
        # Serialize
        serialized = sink.serialize(data)
        
        # Write
        sink.write(serialized, path)
    except MemoryError:
        print(f"Worker {worker_id}: MemoryError caught! Row footprint exceeded limits.")
        sys.exit(3)
    except Exception as e:
        print(f"Error in worker {worker_id}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ── SWEEP PIPELINE ────────────────────────────────────────────────────────────

def run_worker_sweep(format_name, total_rows, worker_counts, memory_limit_gb=None):
    """
    Runs the pipeline at different worker counts, keeping total rows the same.
    Measures throughput (rows/sec) and combined peak RSS.
    """
    results = {}
    print(f"\nRunning Throughput vs. Worker Count Sweep for {format_name.upper()}...")
    print(f"Total Rows to distribute: {total_rows:,}")
    if memory_limit_gb is not None:
        print(f"Memory Limit: {memory_limit_gb} GB combined (applied per-worker via RLIMIT_AS)")
    
    sweep_dir = BASE_DIR / "sweep_shards"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    for w in worker_counts:
        # Cleanup previous files
        for f in sweep_dir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
                
        rows_per_worker = total_rows // w
        # Handle remainder on first worker
        rows_first_worker = rows_per_worker + (total_rows % w)
        
        limit_bytes = None
        if memory_limit_gb is not None:
            limit_bytes = int((memory_limit_gb * 1024**3) / w)
            print(f"  Worker Count: {w:<2} | Rows/Worker: ~{rows_per_worker:,} | Per-Worker Limit: {limit_bytes / 1024**2:.1f} MB")
        else:
            print(f"  Worker Count: {w:<2} | Rows/Worker: ~{rows_per_worker:,}")
        
        monitor = RSSMonitor(interval=0.020)
        monitor.start()
        
        t0 = time.perf_counter()
        
        processes = []
        for i in range(w):
            r = rows_first_worker if i == 0 else rows_per_worker
            p = multiprocessing.Process(
                target=sweep_worker,
                args=(i, format_name, r, str(sweep_dir), limit_bytes)
            )
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
        t1 = time.perf_counter()
        elapsed = t1 - t0
        
        monitor.stop()
        monitor.join()
        
        # Check exits
        success = all(p.exitcode == 0 for p in processes)
        exit_codes = [p.exitcode for p in processes]
        if not success:
            print(f"    ⚠️ Some workers failed or were killed. Exit interpretations:\n{interpret_exit_codes(exit_codes)}")
            
        if not success:
            throughput = None
            peak_rss = None
            peak_vms = None
            print(f"    Time: {elapsed:.2f}s | Throughput: N/A (worker failure) | Peak Combined RSS: N/A | Peak Combined VMS: N/A")
        else:
            throughput = total_rows / elapsed if elapsed > 0 else 0
            peak_rss = monitor.peak_combined_rss
            peak_vms = monitor.peak_combined_vms
            print(f"    Time: {elapsed:.2f}s | Throughput: {throughput:,.0f} rows/s | Peak Combined RSS: {peak_rss/(1024**3):.2f} GB | Peak Combined VMS: {peak_vms/(1024**3):.2f} GB")
            
        results[w] = {
            "elapsed_s": elapsed,
            "throughput_rows_s": throughput,
            "peak_rss_bytes": peak_rss,
            "peak_vms_bytes": peak_vms,
            "success": success,
            "history": monitor.history,
            "exit_codes": exit_codes
        }
        
        # Cleanup files
        for f in sweep_dir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
                
    try:
        sweep_dir.rmdir()
    except OSError:
        pass
        
    return results

# ── STRESS TEST TO EXCEED 16GB RSS ────────────────────────────────────────────

def run_stress_test(format_name, num_workers, rows_per_worker, memory_limit_gb=None, sink_results=None):
    """
    Stress test designed to push the combined RSS past 16GB.
    """
    print(f"\n🔥 Running Memory Stress Test to push peak RSS over 16GB...")
    print(f"Targeting {num_workers} workers, each with {rows_per_worker:,} rows ({format_name.upper()})")
    if memory_limit_gb is not None:
        print(f"Memory Limit: {memory_limit_gb} GB combined (applied per-worker via RLIMIT_AS)")
    
    stress_dir = BASE_DIR / "stress_shards"
    stress_dir.mkdir(parents=True, exist_ok=True)
    
    # Cleanup directory
    for f in stress_dir.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
            
    monitor = RSSMonitor(interval=0.050)
    monitor.start()
    
    t0 = time.perf_counter()
    
    limit_bytes = None
    if memory_limit_gb is not None:
        limit_bytes = int((memory_limit_gb * 1024**3) / num_workers)
        print(f"Enforcing memory limit: {limit_bytes / 1024**2:.1f} MB per worker")
    
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=sweep_worker,
            args=(i, format_name, rows_per_worker, str(stress_dir), limit_bytes)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    t1 = time.perf_counter()
    elapsed = t1 - t0
    
    monitor.stop()
    monitor.join()
    
    # Check exits and worker codes
    failed = any(p.exitcode != 0 for p in processes)
    exit_codes = [p.exitcode for p in processes]
    
    # Cleanup files
    for f in stress_dir.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    try:
        stress_dir.rmdir()
    except OSError:
        pass
        
    peak_rss = monitor.peak_combined_rss
    peak_vms = monitor.peak_combined_vms
    print(f"  Stress Test finished. Time: {elapsed:.2f}s")
    print(f"  Peak Combined RSS: {peak_rss/(1024**3):.4f} GB ({peak_rss / (1024**2):,.1f} MB)")
    print(f"  Peak Combined VMS: {peak_vms/(1024**3):.4f} GB ({peak_vms / (1024**2):,.1f} MB)")
    print("  Worker Exit Interpretations:")
    print(interpret_exit_codes(exit_codes))
    
    # Predict peak RSS & VMS using Phase 1 coefficients
    res_per_row_rss = 28.0
    trans_per_row_rss = 408.49
    res_per_row_vms = 28.0
    trans_per_row_vms = 408.49
    worker_baseline_vms = 250 * 1024**2  # Default base VMS of clean spawn child
    
    using_fallback_baseline = True
    if sink_results and "json" in sink_results:
        res = sink_results["json"]
        num_rows = res.get("size", 1_000_000)
        res_per_row_rss = res["resident_bytes"] / num_rows
        trans_per_row_rss = res["transient_peak"] / num_rows
        
        res_per_row_vms = res_per_row_rss
        trans_per_row_vms = res.get("transient_peak_vms", res["transient_peak"]) / num_rows
        if "baseline_vms" in res and res["baseline_vms"] > 0:
            worker_baseline_vms = res["baseline_vms"]
            using_fallback_baseline = False
            
    predicted_peak_rss = num_workers * rows_per_worker * (res_per_row_rss + trans_per_row_rss)
    # VMS has worker overhead: each spawned child has baseline VMS + incremental allocation VMS
    predicted_peak_vms = num_workers * (worker_baseline_vms + rows_per_worker * (res_per_row_vms + trans_per_row_vms))
    
    print("\n--- Predicted vs Observed Memory Comparison ---")
    if using_fallback_baseline:
        print("⚠️ Warning: Using hardcoded fallback baseline VMS (250.0 MB) because Phase 1 baseline metrics are missing.")
    else:
        print(f"ℹ️ Info: Using empirically measured baseline VMS ({worker_baseline_vms / 1024**2:.1f} MB) from Phase 1.")
        
    print("1. Physical Memory (RSS):")
    print(f"   Predicted peak RSS: {predicted_peak_rss / (1024**3):.4f} GB ({predicted_peak_rss / (1024**2):,.1f} MB)")
    print(f"   Observed peak RSS:  {peak_rss / (1024**3):.4f} GB ({peak_rss / (1024**2):,.1f} MB)")
    if predicted_peak_rss > 0:
        rss_delta = (abs(peak_rss - predicted_peak_rss) / predicted_peak_rss) * 100
        print(f"   RSS Delta:          {rss_delta:.2f}%")
        
    print("\n2. Virtual Address Space (VMS / RLIMIT_AS basis):")
    print(f"   Predicted peak VMS: {predicted_peak_vms / (1024**3):.4f} GB ({predicted_peak_vms / (1024**2):,.1f} MB)")
    print(f"   Observed peak VMS:  {peak_vms / (1024**3):.4f} GB ({peak_vms / (1024**2):,.1f} MB)")
    if predicted_peak_vms > 0:
        vms_delta = (abs(peak_vms - predicted_peak_vms) / predicted_peak_vms) * 100
        print(f"   VMS Delta:          {vms_delta:.2f}%")
        if vms_delta > 15.0:
            print("   ⚠️ VMS Delta is off by more than 15%. The Section 2 formula needs a correction factor for JSONL VMS.")
    print("------------------------------------------------\n")
        
    return {
        "elapsed_s": elapsed,
        "peak_rss_bytes": peak_rss,
        "peak_vms_bytes": peak_vms,
        "exit_codes": exit_codes,
        "failed": failed,
        "history": monitor.history,
        "predicted_rss_bytes": predicted_peak_rss,
        "predicted_vms_bytes": predicted_peak_vms
    }

# ── MAIN RUNNER / CLI ─────────────────────────────────────────────────────────

def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    parser = argparse.ArgumentParser(description="Synthetic Generator Profiling Harness")
    parser.add_argument("--mode", type=str, choices=["sinks", "sweep", "stress", "all"], default="all",
                        help="Harness execution mode (default: all)")
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="Chunk size for single-sink profiling (default: 1M)")
    parser.add_argument("--sweep-total-rows", type=int, default=120_000_000,
                        help="Total rows to divide across workers in sweep (default: 120M)")
    parser.add_argument("--sweep-format", type=str, default="parquet",
                        help="Format to use for the worker count sweep (default: parquet)")
    parser.add_argument("--stress-workers", type=int, default=8,
                        help="Number of workers for stress test (default: 8)")
    parser.add_argument("--stress-rows", type=int, default=2_000_000,
                        help="Rows per worker for stress test (default: 2M)")
    parser.add_argument("--memory-limit", type=float, default=None,
                        help="Combined memory limit in GB to enforce via RLIMIT_AS (default: None)")
    
    args = parser.parse_args()
    
    sink_results = {}
    sweep_results = {}
    stress_result = None
    
    # Try to load cached sink results from previous runs
    cache_path = BASE_DIR / "sink_results_cache.json"
    if cache_path.exists():
        try:
            import json
            with open(cache_path, "r") as f:
                sink_results = json.load(f)
            print(f"Loaded cached single-sink metrics from {cache_path}")
        except Exception as e:
            print(f"⚠️ Could not load cached sink results: {e}")
            
    # Mode 1: Single-Sink Profiling
    if args.mode in ("sinks", "all"):
        print("================================================================================")
        print("PHASE 1: SINK INDEPENDENT PROFILING (1M Rows)")
        print("================================================================================")
        for fmt in ["parquet", "csv", "json"]:
            res = profile_sink(fmt, args.chunk_size)
            if res:
                sink_results[fmt] = res
                
        # Cache results for subsequent runs
        if sink_results:
            try:
                import json
                BASE_DIR.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(sink_results, f, indent=2)
                print(f"Saved single-sink metrics to cache: {cache_path}")
            except Exception as e:
                print(f"⚠️ Failed to cache sink results: {e}")
                
    # Mode 2: Worker Count Throughput Sweep
    if args.mode in ("sweep", "all"):
        print("================================================================================")
        print("PHASE 2: THROUGHPUT VS WORKER COUNT SWEEP")
        print("================================================================================")
        # Determine workers
        core_count = multiprocessing.cpu_count()
        worker_counts = [1, 2, 4, 6, 8, core_count + 1]
        # De-duplicate and sort
        worker_counts = sorted(list(set(worker_counts)))
        
        sweep_results = run_worker_sweep(args.sweep_format, args.sweep_total_rows, worker_counts, args.memory_limit)
        
    # Mode 3: Memory Stress Test
    if args.mode in ("stress", "all"):
        print("================================================================================")
        print("PHASE 3: MEMORY CEILING STRESS TEST")
        print("================================================================================")
        # Use json (JSONL) as it is the most memory intensive sink to force memory usage
        stress_result = run_stress_test("json", args.stress_workers, args.stress_rows, args.memory_limit, sink_results)

    # ── REPORT GENERATION ─────────────────────────────────────────────────────
    print("================================================================================")
    print("PHASE 4: ANALYSIS AND PLOTTING")
    print("================================================================================")
    
    # Create matplotlib plots
    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1e1e1e")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, color="#bbbbbb", fontsize=10)
        ax.set_ylabel(ylabel, color="#bbbbbb", fontsize=10)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444444")
        ax.grid(color="#333333", linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        
    if args.mode == "sinks" and sink_results:
        # Create a 1x2 grid for sinks
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor("#121212")
        
        # Plot 1: Sink phase times
        ax1 = axes[0]
        formats = [f.upper() for f in sink_results.keys()]
        gen_times = [res["gen_time"] for res in sink_results.values()]
        ser_times = [res["serialize_time"] for res in sink_results.values()]
        wr_times = [res["write_time"] for res in sink_results.values()]
        
        x = np.arange(len(formats))
        width = 0.25
        
        ax1.bar(x - width, gen_times, width, label="Generate", color="#3498db")
        ax1.bar(x, ser_times, width, label="Serialize", color="#e67e22")
        ax1.bar(x + width, wr_times, width, label="Write", color="#2ecc71")
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(formats)
        ax1.legend(facecolor="#1e1e1e", labelcolor="white")
        style_ax(ax1, f"Phase Durations per {args.chunk_size:,} Rows", "Format", "Seconds")
        
        # Plot 2: Sink memory comparison
        ax2 = axes[1]
        resident = [res["resident_bytes"] / (1024**2) for res in sink_results.values()]
        transient = [res["transient_peak"] / (1024**2) for res in sink_results.values()]
        output = [res["output_bytes"] / (1024**2) for res in sink_results.values()]
        
        ax2.bar(x - width, resident, width, label="Resident", color="#9b59b6")
        ax2.bar(x, transient, width, label="Transient Peak", color="#e74c3c")
        ax2.bar(x + width, output, width, label="Output File Size", color="#1abc9c")
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(formats)
        ax2.legend(facecolor="#1e1e1e", labelcolor="white")
        style_ax(ax2, f"Memory Footprint per {args.chunk_size:,} Rows", "Format", "MB")
        
        plt.tight_layout()
        plot_path = BASE_DIR / "profile_sinks.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📊 Sinks plot saved to: {plot_path.absolute()}")

    elif args.mode == "sweep" and sweep_results:
        # Create a single plot for sweep
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.patch.set_facecolor("#121212")
        
        valid_workers = [w for w in sorted(sweep_results.keys()) if sweep_results[w]["success"]]
        if valid_workers:
            tp = [sweep_results[w]["throughput_rows_s"] / 1000 for w in valid_workers] # K rows/sec
            ax.plot(valid_workers, tp, marker="o", linewidth=2.5, color="#f1c40f", markersize=8)
            for w, val in zip(valid_workers, tp):
                ax.annotate(f"{val:.1f}k", (w, val), textcoords="offset points", xytext=(0,10), ha="center", color="white", fontweight="bold")
            style_ax(ax, f"Throughput vs Worker Count ({args.sweep_format.upper()} - {args.sweep_total_rows:,} rows)", "Workers", "K Rows / Sec")
            ax.set_xticks(valid_workers)
        
        plt.tight_layout()
        plot_path = BASE_DIR / "profile_sweep.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📊 Sweep plot saved to: {plot_path.absolute()}")

    elif args.mode == "stress" and stress_result:
        # Create a single plot for stress
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.patch.set_facecolor("#121212")
        
        hist_data = stress_result["history"]
        title = f"Stress Test Memory Profile ({args.stress_workers} workers, {args.stress_rows:,} rows JSONL)"
        
        if hist_data:
            times = [h[0] for h in hist_data]
            rss_gb = [h[1] / (1024**3) for h in hist_data]
            ax.plot(times, rss_gb, color="#e74c3c", linewidth=2)
            ax.axhline(16.0, color="#f39c12", linestyle="--", linewidth=1.5, label="16 GB Limit")
            ax.legend(facecolor="#1e1e1e", labelcolor="white")
            style_ax(ax, title, "Time (Seconds)", "Combined RSS (GB)")
            
        plt.tight_layout()
        plot_path = BASE_DIR / "profile_stress.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📊 Stress test plot saved to: {plot_path.absolute()}")

    elif args.mode == "all":
        # Create matplotlib plots (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.patch.set_facecolor("#121212")
        
        # Plot 1: Sink phase times
        ax1 = axes[0, 0]
        if sink_results:
            formats = [f.upper() for f in sink_results.keys()]
            gen_times = [res["gen_time"] for res in sink_results.values()]
            ser_times = [res["serialize_time"] for res in sink_results.values()]
            wr_times = [res["write_time"] for res in sink_results.values()]
            
            x = np.arange(len(formats))
            width = 0.25
            
            ax1.bar(x - width, gen_times, width, label="Generate", color="#3498db")
            ax1.bar(x, ser_times, width, label="Serialize", color="#e67e22")
            ax1.bar(x + width, wr_times, width, label="Write", color="#2ecc71")
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(formats)
            ax1.legend(facecolor="#1e1e1e", labelcolor="white")
            style_ax(ax1, f"Phase Durations per {args.chunk_size:,} Rows", "Format", "Seconds")
            
        # Plot 2: Sink memory comparison
        ax2 = axes[0, 1]
        if sink_results:
            formats = [f.upper() for f in sink_results.keys()]
            resident = [res["resident_bytes"] / (1024**2) for res in sink_results.values()]
            transient = [res["transient_peak"] / (1024**2) for res in sink_results.values()]
            output = [res["output_bytes"] / (1024**2) for res in sink_results.values()]
            
            x = np.arange(len(formats))
            width = 0.25
            
            ax2.bar(x - width, resident, width, label="Resident", color="#9b59b6")
            ax2.bar(x, transient, width, label="Transient Peak", color="#e74c3c")
            ax2.bar(x + width, output, width, label="Output File Size", color="#1abc9c")
            
            ax2.set_xticks(x)
            ax2.set_xticklabels(formats)
            ax2.legend(facecolor="#1e1e1e", labelcolor="white")
            style_ax(ax2, f"Memory Footprint per {args.chunk_size:,} Rows", "Format", "MB")
            
        # Plot 3: Throughput vs worker count
        ax3 = axes[1, 0]
        if sweep_results:
            valid_workers = [w for w in sorted(sweep_results.keys()) if sweep_results[w]["success"]]
            if valid_workers:
                tp = [sweep_results[w]["throughput_rows_s"] / 1000 for w in valid_workers] # K rows/sec
                ax3.plot(valid_workers, tp, marker="o", linewidth=2.5, color="#f1c40f", markersize=8)
                for w, val in zip(valid_workers, tp):
                    ax3.annotate(f"{val:.1f}k", (w, val), textcoords="offset points", xytext=(0,10), ha="center", color="white", fontweight="bold")
                style_ax(ax3, f"Throughput vs Worker Count ({args.sweep_format.upper()} - {args.sweep_total_rows:,} rows)", "Workers", "K Rows / Sec")
                ax3.set_xticks(valid_workers)
            
        # Plot 4: Time-aligned combined RSS (Stress Test or Sweep)
        ax4 = axes[1, 1]
        hist_data = None
        title = "Time-Aligned Combined RSS"
        
        if stress_result and stress_result["history"]:
            hist_data = stress_result["history"]
            title = f"Stress Test Memory Profile ({args.stress_workers} workers, {args.stress_rows:,} rows JSONL)"
        elif sweep_results:
            max_w = max(sweep_results.keys())
            hist_data = sweep_results[max_w]["history"]
            title = f"Combined RSS Time-Series ({max_w} Workers, {args.sweep_format.upper()})"
            
        if hist_data:
            times = [h[0] for h in hist_data]
            rss_gb = [h[1] / (1024**3) for h in hist_data]
            ax4.plot(times, rss_gb, color="#e74c3c", linewidth=2)
            ax4.axhline(16.0, color="#f39c12", linestyle="--", linewidth=1.5, label="16 GB Limit")
            ax4.legend(facecolor="#1e1e1e", labelcolor="white")
            style_ax(ax4, title, "Time (Seconds)", "Combined RSS (GB)")
            
        plt.tight_layout()
        plot_path = BASE_DIR / "profile_summary.png"
        plt.savefig(plot_path, dpi=150)
        print(f"📊 Summary plot saved to: {plot_path.absolute()}")
    
    # Write a Markdown Report
    report_path = BASE_DIR / "profile_report.md"
    
    # Implied safe chunk size work calculations
    safe_calc_work = {}
    memory_limit = 16.0 * 1e9  # 16 GB in bytes
    workers_list = [1, 2, 4, 8, 12]
    
    for fmt, res in sink_results.items():
        res_bytes_per_row = res["resident_bytes"] / args.chunk_size
        trans_bytes_per_row = res["transient_peak"] / args.chunk_size
        denom = res_bytes_per_row + trans_bytes_per_row
        
        # Get baseline VMS for the worker, default to 250MB if not measured
        worker_baseline_vms = res.get("baseline_vms", 250 * 1024**2)
        
        by_workers = {}
        for w in workers_list:
            budget_per_worker = memory_limit / w
            net_budget_per_worker = budget_per_worker - worker_baseline_vms
            if net_budget_per_worker <= 0 or denom <= 0:
                safe_size = 0
            else:
                safe_size = int(net_budget_per_worker / denom)
            by_workers[w] = safe_size
            
        safe_calc_work[fmt] = {
            "res_per_row": res_bytes_per_row,
            "trans_per_row": trans_bytes_per_row,
            "denom": denom,
            "baseline_vms": worker_baseline_vms,
            "by_workers": by_workers
        }
        
    # Generate markdown content
    md = []
    md.append("# Synthetic Generator Profiling Report\n")
    md.append("This profiling report provides empirical measurements of resident/transient memory usage and execution times across three sink formats: **Parquet**, **CSV**, and **JSONL**.\n")
    
    md.append("## 1. Single-Sink Profiling (1M Rows)")
    md.append("Independent runs on a single representative chunk of 1,000,000 rows.\n")
    
    md.append("| Format | Resident Memory (MB) | Transient Peak (MB) | Output Size (MB) | Gen Time (s) | Serialize Time (s) | Write Time (s) |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for fmt in ["parquet", "csv", "json"]:
        if fmt in sink_results:
            r = sink_results[fmt]
            disp_name = "JSONL" if fmt == "json" else fmt.upper()
            md.append(f"| {disp_name} | {r['resident_bytes']/(1024**2):.2f} | {r['transient_peak']/(1024**2):.2f} | {r['output_bytes']/(1024**2):.2f} | {r['gen_time']:.3f} | {r['serialize_time']:.3f} | {r['write_time']:.3f} |")
    md.append("\n")
    
    md.append("### Memory Limit Analysis & Derived Safe Chunk Sizes")
    md.append("Derived safe chunk sizes (`rows_per_chunk`) under a **16GB combined RAM limit** across $N$ workers.")
    md.append("Formula (Baseline-Corrected): `safe_rows_per_chunk = ((16.0 * 1e9 / N) - worker_baseline_vms) / (Resident_per_row + Transient_per_row)`\n")
    
    for fmt, calc in safe_calc_work.items():
        disp_name = "JSONL" if fmt == "json" else fmt.upper()
        md.append(f"#### {disp_name}")
        md.append(f"- **Resident memory per row**: {calc['res_per_row']:.2f} bytes")
        md.append(f"- **Transient memory peak per row**: {calc['trans_per_row']:.2f} bytes")
        md.append(f"- **Combined denominator (Resident + Transient per row)**: {calc['denom']:.2f} bytes")
        md.append(f"- **Worker baseline VMS**: {calc['baseline_vms'] / 1024**2:.1f} MB")
        md.append("- **Implied Safe Rows per Worker Chunk**:")
        for w, safe_size in calc["by_workers"].items():
            if safe_size == 0:
                md.append(f"  - **$N = {w}$ workers**: `0` rows per chunk — *(Baseline VMS overhead exceeds per-worker memory budget)*")
            else:
                md.append(f"  - **$N = {w}$ workers**: `{safe_size:,}` rows per chunk")
                md.append(f"    *Calculation: ((16.0e9 / {w}) - {calc['baseline_vms']/1024**2:.1f}MB) / {calc['denom']:.2f}*")
        md.append("\n")
        
    md.append("## 2. Worker Count vs. Throughput Sweep")
    md.append(f"Running total of {args.sweep_total_rows:,} rows using **{args.sweep_format.upper()}**.\n")
    
    md.append("| Workers | Elapsed Time (s) | Throughput (Rows/s) | Peak Combined RSS (GB) | Success |")
    md.append("| --- | --- | --- | --- | --- |")
    if sweep_results:
        for w in sorted(sweep_results.keys()):
            s = sweep_results[w]
            if s["success"]:
                md.append(f"| {w} | {s['elapsed_s']:.2f} | {s['throughput_rows_s']:,.0f} | {s['peak_rss_bytes']/(1024**3):.2f} | {s['success']} |")
            else:
                md.append(f"| {w} | {s['elapsed_s']:.2f} | N/A (worker failure) | N/A | {s['success']} |")
    md.append("\n")
    
    md.append("## 3. Memory Ceiling Stress Test")
    if stress_result:
        md.append(f"Targeted stress test using **JSONL** sink with **{args.stress_workers} workers** and **{args.stress_rows:,} rows** per worker.")
        md.append(f"- **Elapsed time**: {stress_result['elapsed_s']:.2f} seconds")
        md.append(f"- **Peak Combined RSS**: **{stress_result['peak_rss_bytes']/(1024**3):.2f} GB**")
        md.append(f"- **Exit status**: {'Failed/Killed' if stress_result['failed'] else 'Success'}")
        md.append(f"- **Worker Exit Codes**: `{stress_result['exit_codes']}`")
    else:
        md.append("Stress test was not run.")
    md.append("\n")
    
    md.append("## 4. Bottleneck Hypotheses & Insights")
    md.append("1. **CSV & JSONL Serialization Overhead**: Both formats show high transient peak RAM due to Python object materialization and string conversions. In contrast, Parquet remains columnar and exhibits minimal transient overhead during the serialization phase.")
    md.append("2. **Throughput Scaling**: Review the Throughput vs Worker Count sweep to observe potential scaling bottlenecks. If throughput plateaus or drops as worker count exceeds physical cores, this suggests memory bandwidth or disk write I/O contention (labeled as a hypothesis until isolated).")
    md.append("3. **16GB RAM Edge**: The stress test results show the actual combined memory usage under load. When pushing RSS past 16GB, notice if swap activity significantly degrades throughput or leads to OOM termination.")
    
    # Save markdown file
    with open(report_path, "w") as f:
        f.write("\n".join(md))
        
    print(f"📄 Report written to: {report_path.absolute()}")
    print("\n================================================================================")
    print("PROFILING RUN COMPLETED SUCCESSFULLY")
    print("================================================================================")


if __name__ == "__main__":
    main()
