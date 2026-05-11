import tracemalloc

def benchmark(name, path, reader, runs):
    if not path.exists():
        print(f"  ⚠️  {path} not found, skipping.")
        return None

    size_mb = path.stat().st_size / (1024 ** 2)
    times, mem_deltas = [], []
    rows = 0

    for run in range(1, runs + 1):
        gc.collect()
        mem_before = mem_mb()
        print(f"  RAM before run {run}: {mem_before:.0f} MB")

        t1 = time.perf_counter()
        df = reader(path)
        t2 = time.perf_counter()

        mem_after = mem_mb()
        elapsed   = t2 - t1
        rows      = len(df)

        times.append(elapsed)
        mem_deltas.append(mem_after - mem_before)

        print(f"  Run {run}: {elapsed:.3f}s  |  RAM {mem_before:.0f} → {mem_after:.0f} MB  (Δ {mem_after - mem_before:+.0f} MB)  ({rows:,} rows)")

        del df
        gc.collect()

        mem_after_del = mem_mb()
        print(f"  RAM after del+gc: {mem_after_del:.0f} MB  (freed {mem_after - mem_after_del:+.0f} MB)")

    return {
        "format":    name,
        "size_mb":   size_mb,
        "rows":      rows,
        "avg_s":     np.mean(times),
        "best_s":    np.min(times),
        "worst_s":   np.max(times),
        "runs":      times,
        "mb_per_s":  size_mb / np.mean(times),
        "avg_mem":   np.mean(mem_deltas),
    }