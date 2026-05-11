import gc
import tracemalloc
import time
from pathlib import Path
import psutil
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR = Path("data")
FILES = {
    "parquet": BASE_DIR / "dataset.parquet",
    "csv":     BASE_DIR / "dataset.csv",
}
RUNS = 3
PLOT_FILE = BASE_DIR / "benchmark.png"
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "csv":     "#4C9BE8",
    "parquet": "#4CE87A",
}

# ── Readers ───────────────────────────────────────────────────────────────────

def read_csv(path):     return pd.read_csv(path)
def read_parquet(path): return pq.read_table(path).to_pandas()

READERS = {
    "csv":     read_csv,
    "parquet": read_parquet,
}

# ── Benchmark ─────────────────────────────────────────────────────────────────

def mem_mb():
    return PROCESS.memory_info().rss / (1024 ** 2)

def benchmark(name, path, reader, runs):
    if not path.exists():
        print(f"  ⚠️  {path} not found, skipping.")
        return None

    size_mb = path.stat().st_size / (1024 ** 2)
    times   = []
    rows    = 0

    for run in range(1, runs + 1):
        gc.collect()
        mem_before = mem_mb()

        t1 = time.perf_counter()
        df = reader(path)
        t2 = time.perf_counter()

        mem_after = mem_mb()
        elapsed   = t2 - t1
        times.append(elapsed)
        mem_deltas.append(mem_after - mem_before)

        print(f"  Run {run}: {elapsed:.3f}s  |  RAM Δ {mem_after - mem_before:+.0f} MB  ({len(df):,} rows)")

        del df          # explicit delete
import gc
import time
from multiprocessing import Process, Queue
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

# ── CONFIG ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path("data")
FILES     = {
    "parquet": BASE_DIR / "dataset.parquet",
    "csv":     BASE_DIR / "dataset.csv",
}
RUNS      = 3
PLOT_FILE = BASE_DIR / "benchmark.png"
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "csv":     "#4C9BE8",
    "parquet": "#4CE87A",
}

# ── Readers ───────────────────────────────────────────────────────────────────
def read_csv(path):
    return pa_csv.read_csv(
        path,
        convert_options=pa_csv.ConvertOptions(
            column_types={
                "id":      pa.int32(),
                "value_1": pa.int32(),
                "value_2": pa.int32(),
                "char_1":  pa.dictionary(pa.int32(), pa.string()),  # int8 → int32
                "char_2":  pa.dictionary(pa.int32(), pa.string()),  # int8 → int32
            }
        )
    )

def read_parquet(path):
    return pq.read_table(path)   # stays as Arrow Table, no pandas

READERS = {
    "csv":     read_csv,
    "parquet": read_parquet,
}

# ── Worker (runs in isolated subprocess) ─────────────────────────────────────

def _worker(name, path, runs, q):
    """
    Each format benchmark runs in its own subprocess.
    When this process exits, the OS reclaims ALL memory — guaranteed.
    No shared heap with the main process.
    """
    import gc
    import time
    import numpy as np
    import psutil
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    import pyarrow.parquet as pq

    process = psutil.Process()
    def mem_mb(): return process.memory_info().rss / (1024 ** 2)

    readers = {
        "csv":     read_csv,
        "parquet": read_parquet,
    }

    reader   = readers[name]
    path     = Path(path)
    size_mb  = path.stat().st_size / (1024 ** 2)
    times, mem_deltas = [], []
    rows = 0

    for run in range(1, runs + 1):
        gc.collect()
        mem_before = mem_mb()

        t1  = time.perf_counter()
        df  = reader(path)
        t2  = time.perf_counter()

        mem_after = mem_mb()
        elapsed   = t2 - t1
        rows      = len(df)

        times.append(elapsed)
        mem_deltas.append(mem_after - mem_before)

        print(f"  Run {run}: {elapsed:.3f}s  |  "
              f"RAM {mem_before:.0f} → {mem_after:.0f} MB  "
              f"(Δ {mem_after - mem_before:+.0f} MB)  "
              f"({rows:,} rows)")

        del df
        gc.collect()

        mem_freed = mem_mb()
        print(f"  RAM after del+gc: {mem_freed:.0f} MB  "
              f"(freed {mem_after - mem_freed:+.0f} MB)")

    q.put({
        "format":   name,
        "size_mb":  size_mb,
        "rows":     rows,
        "avg_s":    np.mean(times),
        "best_s":   np.min(times),
        "worst_s":  np.max(times),
        "runs":     times,
        "mb_per_s": size_mb / np.mean(times),
        "avg_mem":  np.mean(mem_deltas),
    })

# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(name, path):
    if not path.exists():
        print(f"  ⚠️  {path} not found, skipping.")
        return None

    print(f"\n📂  Reading {name.upper()} ({path.name})...")

    q = Queue()
    p = Process(target=_worker, args=(name, str(path), RUNS, q))
    p.start()
    p.join()    # block until subprocess exits — OS reclaims its entire heap

    return q.get()

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(results):
    results = [r for r in results if r is not None]
    if not results:
        print("\nNo benchmark plot to generate.")
        return

    names  = [r["format"].upper() for r in results]
    colors = [COLORS[r["format"]]  for r in results]
    avgs   = [r["avg_s"]           for r in results]
    bests  = [r["best_s"]          for r in results]
    worsts = [r["worst_s"]         for r in results]
    sizes  = [r["size_mb"]         for r in results]
    mbps   = [r["mb_per_s"]        for r in results]
    runs   = [r["runs"]            for r in results]
    mems   = [r["avg_mem"]         for r in results]

    err_lo = [a - b for a, b in zip(avgs, bests)]
    err_hi = [w - a for w, a in zip(worsts, avgs)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle("File Format Read Benchmark", color="white",
                 fontsize=16, fontweight="bold", y=0.98)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1a1a1a")
        ax.set_title(title, color="white", fontsize=11, pad=10)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333333")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.grid(axis="y", color="#2a2a2a", linewidth=0.7, linestyle="--")
        ax.set_axisbelow(True)

    def bar_labels(ax, bars, fmt="{:.2f}"):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + h * 0.02,
                fmt.format(h),
                ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold"
            )

    x = np.arange(len(names))

    # ── 1. Avg read time + error bars ─────────────────────────────────────────
    ax1 = axes[0, 0]
    bars = ax1.bar(x, avgs, color=colors, width=0.5, zorder=3)
    ax1.errorbar(x, avgs, yerr=[err_lo, err_hi],
                 fmt="none", color="white", capsize=6, linewidth=1.5, zorder=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, color="white")
    bar_labels(ax1, bars, fmt="{:.2f}s")
    style_ax(ax1, "Avg Read Time (lower = better)", "Format", "Seconds")

    # ── 2. Per-run scatter ────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    for i, (name, run_times, color) in enumerate(zip(names, runs, colors)):
        ax2.scatter([i] * len(run_times), run_times, color=color,
                    s=80, zorder=4, label=name)
        ax2.plot([i] * len(run_times), run_times, color=color,
                 alpha=0.3, linewidth=1)
        ax2.hlines(np.mean(run_times), i - 0.15, i + 0.15,
                   colors=color, linewidth=2.5, zorder=5)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, color="white")
    ax2.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    style_ax(ax2, "Per-Run Times (— = mean)", "Format", "Seconds")

    # ── 3. File size on disk ──────────────────────────────────────────────────
    ax3 = axes[0, 2]
    bars = ax3.bar(x, sizes, color=colors, width=0.5, zorder=3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, color="white")
    bar_labels(ax3, bars, fmt="{:.1f} MB")
    style_ax(ax3, "File Size on Disk (lower = better)", "Format", "MB")

    # ── 4. Throughput ─────────────────────────────────────────────────────────
    ax4 = axes[1, 0]
    bars = ax4.bar(x, mbps, color=colors, width=0.5, zorder=3)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, color="white")
    bar_labels(ax4, bars, fmt="{:.1f} MB/s")
    style_ax(ax4, "Read Throughput (higher = better)", "Format", "MB/s")

    # ── 5. Peak memory usage ──────────────────────────────────────────────────
    ax5 = axes[1, 1]
    bars = ax5.bar(x, mems, color=colors, width=0.5, zorder=3)
    ax5.set_xticks(x)
    ax5.set_xticklabels(names, color="white")
    bar_labels(ax5, bars, fmt="{:.0f} MB")
    style_ax(ax5, "Avg Peak RAM Usage (lower = better)", "Format", "MB")

    # ── 6. MB/s vs RAM scatter ────────────────────────────────────────────────
    ax6 = axes[1, 2]
    for i, (name, mb, mem, color) in enumerate(zip(names, mbps, mems, colors)):
        ax6.scatter(mem, mb, color=color, s=120, zorder=4, label=name)
        ax6.annotate(name, (mem, mb), textcoords="offset points",
                     xytext=(6, 4), color=color, fontsize=9)
    ax6.set_xlabel("Peak RAM (MB)", color="#aaaaaa", fontsize=9)
    ax6.set_ylabel("Throughput (MB/s)", color="#aaaaaa", fontsize=9)
    ax6.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    style_ax(ax6, "Throughput vs RAM (top-left = best)", "Peak RAM (MB)", "MB/s")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\n📊  Plot saved → {PLOT_FILE}")

# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results):
    results = [r for r in results if r is not None]
    if not results:
        print("\nNo benchmark results to report.")
        return

    best = min(results, key=lambda r: r["avg_s"])

    print("\n" + "═" * 74)
    print(f"  {'FORMAT':<10} {'SIZE MB':>8} {'AVG':>8} {'BEST':>8} {'WORST':>8} {'MB/s':>8} {'RAM MB':>8}")
    print("─" * 74)
    for r in results:
        win = " ✅" if r["format"] == best["format"] else ""
        print(f"  {r['format']:<10} {r['size_mb']:>8.1f} {r['avg_s']:>7.3f}s "
              f"{r['best_s']:>7.3f}s {r['worst_s']:>7.3f}s "
              f"{r['mb_per_s']:>7.1f} {r['avg_mem']:>7.0f}{win}")
    print("═" * 74)
    print(f"\n  🏆  Fastest: {best['format'].upper()}  "
          f"(avg {best['avg_s']:.3f}s over {RUNS} runs)\n")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []
    for name, path in FILES.items():
        results.append(benchmark(name, path))

    print_report(results)
    plot(results)