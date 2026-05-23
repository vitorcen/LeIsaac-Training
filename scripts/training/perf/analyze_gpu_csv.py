"""Analyze sample_gpu.sh CSV output for training efficiency.

Reports:
    - GPU util mid-window mean (drops first/last 5%)
    - GPU util p50 / p90 / max
    - CPU util p50 / max
    - Peak GPU memory

Mid-window = sort util ascending, take [50%, 95%] slice — robust to startup + finish
phases that drag the simple mean down. This is the "efficiency anchor" we judge by
([[feedback-gpu-util-as-efficiency-anchor.md]]).

Usage:
    python analyze_gpu_csv.py <csv_path>
    python analyze_gpu_csv.py /tmp/gpu_baseline.csv /tmp/gpu_memmap.csv  # compare A/B
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def analyze_one(csv_path: Path) -> dict:
    rows = list(csv.reader(csv_path.open()))[1:]  # skip header
    util = [int(r[1]) for r in rows if r[1].isdigit()]
    mem = [int(r[2]) for r in rows if r[2].isdigit()]
    cpu = [float(r[4]) for r in rows if len(r) > 4]
    if not util:
        return {"path": str(csv_path), "n": 0}
    mid = sorted(util)[len(util) // 2 : int(len(util) * 0.95)]
    p90 = sorted(util)[max(0, int(0.9 * len(util)) - 1)]
    return {
        "path": str(csv_path),
        "n_samples": len(util),
        "gpu_util_p50": statistics.median(util),
        "gpu_util_p90": p90,
        "gpu_util_max": max(util),
        "gpu_util_mid": round(statistics.mean(mid), 1),
        "gpu_mem_peak_mib": max(mem),
        "cpu_p50": round(statistics.median(cpu), 1),
        "cpu_max": round(max(cpu), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="CSV files from sample_gpu.sh")
    args = p.parse_args()

    rows = [analyze_one(Path(p)) for p in args.paths]
    if not rows:
        return

    cols = [
        ("name", lambda r: Path(r["path"]).stem),
        ("n", "n_samples"),
        ("gpu_mid%", "gpu_util_mid"),
        ("gpu_p50", "gpu_util_p50"),
        ("gpu_p90", "gpu_util_p90"),
        ("gpu_max", "gpu_util_max"),
        ("mem_MB", "gpu_mem_peak_mib"),
        ("cpu_p50", "cpu_p50"),
        ("cpu_max", "cpu_max"),
    ]
    print(" | ".join(f"{c[0]:>10}" for c in cols))
    print("-" * (13 * len(cols)))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c[1]) if isinstance(c[1], str) else c[1](r)
            cells.append(f"{str(v):>10}")
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
